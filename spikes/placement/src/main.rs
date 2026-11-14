#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, unsafe_op_in_unsafe_fn, clippy::all)]
//! Placement spike: load a GGUF through llama.cpp's C API, route chosen tensors
//! into a Rust-implemented buffer type, and check from the kernel's side where
//! every tensor actually lives — before and after asking the kernel to evict.

mod pinned;
mod residency;
mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_void};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

struct Args {
    model: PathBuf,
    pins: Vec<String>,
    numa: Option<u32>,
    mlock: bool,
    load_mode: String,
    tokens: usize,
    evict: bool,
    verbose: bool,
    extra_bufts: bool,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let mut a = Args { model: PathBuf::new(), pins: vec![], numa: None, mlock: true, load_mode: "mmap".into(), tokens: 16, evict: false, verbose: false, extra_bufts: false };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--pin" => a.pins.push(it.next().expect("--pin REGEX")),
            "--numa" => a.numa = Some(it.next().expect("--numa NODE").parse().unwrap()),
            "--no-mlock" => a.mlock = false,
            "--load-mode" => a.load_mode = it.next().expect("--load-mode mmap|none"),
            "--tokens" => a.tokens = it.next().expect("--tokens N").parse().unwrap(),
            "--evict" => a.evict = true,
            "--verbose" => a.verbose = true,
            // llama.cpp's repacked CPU buffer types (AMX, aarch64, ...). They silently
            // copy some weights out of the mmap, so they are off by default here.
            "--extra-bufts" => a.extra_bufts = true,
            s if !s.starts_with("--") && a.model.as_os_str().is_empty() => a.model = s.into(),
            s => panic!("unknown argument {s}"),
        }
    }
    assert!(!a.model.as_os_str().is_empty(), "usage: placement-spike MODEL.gguf [--pin REGEX]... [--numa N] [--no-mlock] [--load-mode mmap|none] [--tokens N] [--evict] [--extra-bufts] [--verbose]");
    a
}

static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());
static VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

unsafe extern "C" fn log_cb(_level: sys::ggml_log_level, text: *const libc::c_char, _: *mut c_void) {
    let s = CStr::from_ptr(text).to_string_lossy().into_owned();
    if VERBOSE.load(std::sync::atomic::Ordering::Relaxed) {
        eprint!("{s}");
    }
    LOG.lock().unwrap().push(s);
}

/// Which "subsystem" box in the UI a tensor belongs to.
fn subsystem(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("blk.") {
        let layer = rest.split('.').next().unwrap();
        if name.contains("_exps") {
            return format!("experts  blk.{layer}");
        }
        if name.contains("ffn_gate_inp") {
            return "router   (all layers)".into();
        }
        return "attn+norms (all layers)".into();
    }
    if name.starts_with("token_embd") {
        return "embeddings".into();
    }
    "output head".into()
}

#[derive(Default)]
struct Group {
    bytes: usize,
    resident_bytes: f64,
    homes: BTreeMap<&'static str, usize>,
}

/// Where each tensor lives and how much of it is in RAM, grouped by subsystem.
fn report(title: &str, layout: &[residency::FileTensor], mapping: Option<(usize, usize, usize)>) {
    let reg = pinned::REGISTRY.lock().unwrap();
    let pinned_at: BTreeMap<&str, (usize, usize)> =
        reg.iter().filter(|b| !b.freed).flat_map(|b| b.tensors.iter().map(|(n, a, s)| (n.as_str(), (*a, *s)))).collect();

    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    for t in layout {
        let g = groups.entry(subsystem(&t.name)).or_default();
        g.bytes += t.size;
        let (home, frac) = if let Some((addr, size)) = pinned_at.get(t.name.as_str()) {
            ("rust-pinned RAM", residency::resident_fraction(*addr, *size))
        } else if let Some((origin, _, _)) = mapping {
            ("mmap'd file (page cache)", residency::resident_fraction(origin + t.file_offset, t.size))
        } else {
            ("ggml-owned RAM", f64::NAN)
        };
        *g.homes.entry(home).or_default() += t.size;
        if !frac.is_nan() {
            g.resident_bytes += frac * t.size as f64;
        }
    }

    println!("\n== {title} ==");
    println!("{:<26} {:>9} {:>10}  lives in", "subsystem", "MiB", "in RAM");
    for (name, g) in &groups {
        let homes: Vec<String> = g.homes.iter().map(|(h, b)| if g.homes.len() == 1 { h.to_string() } else { format!("{h} ({:.1} MiB)", *b as f64 / 1048576.0) }).collect();
        let resident = if g.homes.contains_key("ggml-owned RAM") { "n/a".to_string() } else { format!("{:.0}%", 100.0 * g.resident_bytes / g.bytes as f64) };
        println!("{:<26} {:>9.1} {:>10}  {}", name, g.bytes as f64 / 1048576.0, resident, homes.join(" + "));
    }
    println!("process: VmRSS={}  VmLck={}", residency::proc_status("VmRSS:"), residency::proc_status("VmLck:"));
}

struct Generation {
    tokens: Vec<i32>,
    logits_fnv: u64,
    ms_per_token: f64,
}

unsafe fn generate(ctx: *mut sys::llama_context, n_vocab: usize, n_new: usize) -> Generation {
    sys::llama_memory_clear(sys::llama_get_memory(ctx), true);
    // Byte-fallback vocab: token id = byte + 3, BOS = 1.
    let mut prompt: Vec<i32> = std::iter::once(1).chain(b"Where should the experts live?".iter().map(|b| *b as i32 + 3)).collect();
    let mut fnv: u64 = 0xcbf29ce484222325;
    let mut out = vec![];
    let start = Instant::now();
    let mut batch = sys::llama_batch_get_one(prompt.as_mut_ptr(), prompt.len() as i32);
    for _ in 0..n_new {
        assert_eq!(sys::llama_decode(ctx, batch), 0, "llama_decode failed");
        let logits = std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, -1), n_vocab);
        for l in logits {
            for b in l.to_bits().to_le_bytes() {
                fnv = (fnv ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as i32;
        out.push(next);
        // batch_get_one keeps a pointer; give it storage that outlives the call.
        prompt = vec![next];
        batch = sys::llama_batch_get_one(prompt.as_mut_ptr(), 1);
    }
    Generation { tokens: out, logits_fnv: fnv, ms_per_token: start.elapsed().as_secs_f64() * 1000.0 / n_new as f64 }
}

fn main() {
    let a = parse_args();
    VERBOSE.store(a.verbose, std::sync::atomic::Ordering::Relaxed);
    unsafe {
        sys::llama_log_set(Some(log_cb), std::ptr::null_mut());
        sys::llama_backend_init();
    }

    let layout = residency::gguf_layout(&a.model);

    // --- placement: pattern -> buffer type, exactly what the UI would emit ---
    let pin_buft = (!a.pins.is_empty()).then(|| pinned::new_buffer_type("RustPinned", pinned::PinOptions { numa_node: a.numa, mlock: a.mlock, hugepages: true }));
    let patterns: Vec<CString> = a.pins.iter().map(|p| CString::new(p.as_str()).unwrap()).collect();
    let mut overrides: Vec<sys::llama_model_tensor_buft_override> =
        patterns.iter().map(|p| sys::llama_model_tensor_buft_override { pattern: p.as_ptr(), buft: pin_buft.unwrap() }).collect();
    overrides.push(sys::llama_model_tensor_buft_override { pattern: std::ptr::null(), buft: std::ptr::null_mut() });

    let model_path = CString::new(a.model.to_str().unwrap()).unwrap();
    let t0 = Instant::now();
    let model = unsafe {
        let mut mp = sys::llama_model_default_params();
        mp.tensor_buft_overrides = overrides.as_ptr();
        mp.use_extra_bufts = a.extra_bufts;
        mp.load_mode = match a.load_mode.as_str() {
            "mmap" => sys::llama_load_mode_LLAMA_LOAD_MODE_MMAP,
            "none" => sys::llama_load_mode_LLAMA_LOAD_MODE_NONE,
            m => panic!("unknown load mode {m}"),
        };
        sys::llama_model_load_from_file(model_path.as_ptr(), mp)
    };
    assert!(!model.is_null(), "model load failed; rerun with --verbose");
    println!("loaded {} in {:.0} ms (load_mode={}, pins={:?})", a.model.display(), t0.elapsed().as_secs_f64() * 1000.0, a.load_mode, a.pins);
    for l in LOG.lock().unwrap().iter().filter(|l| l.contains("buffer size")) {
        print!("  llama.cpp: {}", l.trim_start_matches(|c: char| c != ' ').trim_start());
    }

    {
        let reg = pinned::REGISTRY.lock().unwrap();
        for (i, b) in reg.iter().enumerate() {
            let nodes = residency::numa_nodes(b.base, b.len);
            println!(
                "  RustPinned buffer #{i}: {:.1} MiB @ {:#x}, {} tensors, mlock={:?}, mbind={:?}, page NUMA nodes={:?}",
                b.len as f64 / 1048576.0, b.base, b.tensors.len(), b.mlock_result, b.mbind_result, nodes
            );
        }
    }

    let mapping = if a.load_mode == "mmap" { residency::find_file_mapping(&a.model) } else { None };
    report("after load", &layout, mapping);

    let ctx = unsafe {
        let mut cp = sys::llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 512;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        sys::llama_init_from_model(model, cp)
    };
    assert!(!ctx.is_null());
    let n_vocab = unsafe { sys::llama_vocab_n_tokens(sys::llama_model_get_vocab(model)) } as usize;

    let g = unsafe { generate(ctx, n_vocab, a.tokens) };
    report("after generating (warm)", &layout, mapping);
    println!("generation: {:.2} ms/token", g.ms_per_token);
    println!("RESULT tokens={:?} logits_fnv={:016x}", g.tokens, g.logits_fnv);

    if a.evict {
        // Ask the kernel to drop everything it can: the whole file mapping and our buffers.
        if let Some((_, s, e)) = mapping {
            residency::page_out(s, e - s);
        }
        for b in pinned::REGISTRY.lock().unwrap().iter().filter(|b| !b.freed) {
            residency::page_out(b.base, b.len);
        }
        report("after MADV_PAGEOUT on everything", &layout, mapping);

        let g2 = unsafe { generate(ctx, n_vocab, a.tokens) };
        report("after generating again (cold)", &layout, mapping);
        println!("generation after eviction: {:.2} ms/token", g2.ms_per_token);
        println!("RESULT-AFTER-EVICT tokens={:?} logits_fnv={:016x} identical={}", g2.tokens, g2.logits_fnv, g2.logits_fnv == g.logits_fnv);
    }

    unsafe {
        sys::llama_free(ctx);
        sys::llama_model_free(model);
    }
}
