#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code, unsafe_op_in_unsafe_fn, clippy::all)]
//! Placement spike: load a GGUF through llama.cpp's C API, route tensors to
//! devices / a Rust-implemented RAM buffer / the mmap'd file by regex, and
//! check from the kernel's side where every tensor actually lives. Measures
//! prefill and decode throughput so placements can be compared.

mod pinned;
mod residency;
mod verify;
mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString, c_void};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

const USAGE: &str = "usage: placement-spike MODEL.gguf [options]

placement (rules are tried in order; first match wins, like llama.cpp -ot):
  --place REGEX=DEST   DEST = pinned | CPU | <device name, e.g. CUDA0>
                       pinned = Rust buffer (mlock, mbind, hugepages)
                       CPU    = llama.cpp default CPU buffer (mmap'd file if --load-mode mmap)
  --pin REGEX          same as --place REGEX=pinned
  --gpu-layers N       layers llama.cpp offloads by itself (default 0; -1 = all)
  --tensor-split A,B   layer split ratio across GPUs
  --numa N             mbind pinned memory to NUMA node N
  --no-mlock           don't mlock pinned memory
  --cuda-register      cudaHostRegister pinned memory (faster GPU offload of those ops)
  --load-mode M        mmap (default) | none
  --extra-bufts        allow llama.cpp's repacked CPU buffer types (AMX, CPU_REPACK, ...)
  --moe-cache MIB      llama.cpp moe_cache_size: GPU cache for experts kept in host memory
  --no-op-offload      don't let the scheduler ship host weights to a GPU for big batches
  --no-host            don't let llama.cpp put CPU tensors in the GPU's pinned host buffer type

workload:
  --prompt-tokens N    prefill length (default 512)
  --tokens N           tokens to generate (default 64)
  --ctx N              context size (default prompt+tokens+64)
  --batch N            n_batch / n_ubatch (default 512)
  --threads N          CPU threads (default: available cores)
  --flash-attn on|off|auto
  --repeat N           timed runs; median is reported (default 3)
  --evict              MADV_PAGEOUT everything after the timed runs, then run once more

output:
  --label NAME         name for the RESULT line
  --per-layer          break the placement table down per layer
  --no-verify          skip the ground-truth placement check (eval-callback pass)
  --verbose            echo llama.cpp's log";

struct Args {
    model: PathBuf,
    rules: Vec<(String, String)>,
    gpu_layers: i32,
    tensor_split: Vec<f32>,
    numa: Option<u32>,
    mlock: bool,
    cuda_register: bool,
    load_mode: String,
    extra_bufts: bool,
    moe_cache_mib: usize,
    prompt_tokens: usize,
    tokens: usize,
    ctx: Option<u32>,
    batch: u32,
    threads: i32,
    flash_attn: String,
    evict: bool,
    repeat: usize,
    label: String,
    per_layer: bool,
    verbose: bool,
    verify: bool,
    op_offload: bool,
    no_host: bool,
}

fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
    let mut a = Args {
        model: PathBuf::new(), rules: vec![], gpu_layers: 0, tensor_split: vec![], numa: None, mlock: true,
        cuda_register: false, load_mode: "mmap".into(), extra_bufts: false, moe_cache_mib: 0, prompt_tokens: 512,
        tokens: 64, ctx: None, batch: 512, threads, flash_attn: "auto".into(), evict: false, repeat: 3, label: String::new(),
        per_layer: false, verbose: false, verify: true, op_offload: true, no_host: false,
    };
    let val = |name: &str, it: &mut dyn Iterator<Item = String>| it.next().unwrap_or_else(|| panic!("{name} needs a value\n\n{USAGE}"));
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--place" => {
                let v = val("--place", &mut it);
                let (re, dest) = v.rsplit_once('=').unwrap_or_else(|| panic!("--place wants REGEX=DEST, got {v}"));
                a.rules.push((re.into(), dest.into()));
            }
            "--pin" => a.rules.push((val("--pin", &mut it), "pinned".into())),
            "--gpu-layers" | "-ngl" => a.gpu_layers = val("--gpu-layers", &mut it).parse().unwrap(),
            "--tensor-split" => a.tensor_split = val("--tensor-split", &mut it).split(',').map(|x| x.parse().unwrap()).collect(),
            "--numa" => a.numa = Some(val("--numa", &mut it).parse().unwrap()),
            "--no-mlock" => a.mlock = false,
            "--cuda-register" => a.cuda_register = true,
            "--load-mode" => a.load_mode = val("--load-mode", &mut it),
            "--extra-bufts" => a.extra_bufts = true,
            "--moe-cache" => a.moe_cache_mib = val("--moe-cache", &mut it).parse().unwrap(),
            "--prompt-tokens" => a.prompt_tokens = val("--prompt-tokens", &mut it).parse().unwrap(),
            "--tokens" => a.tokens = val("--tokens", &mut it).parse().unwrap(),
            "--ctx" => a.ctx = Some(val("--ctx", &mut it).parse().unwrap()),
            "--batch" => a.batch = val("--batch", &mut it).parse().unwrap(),
            "--threads" => a.threads = val("--threads", &mut it).parse().unwrap(),
            "--flash-attn" => a.flash_attn = val("--flash-attn", &mut it),
            "--evict" => a.evict = true,
            "--repeat" => a.repeat = val("--repeat", &mut it).parse::<usize>().unwrap().max(1),
            "--label" => a.label = val("--label", &mut it),
            "--per-layer" => a.per_layer = true,
            "--verbose" => a.verbose = true,
            "--no-verify" => a.verify = false,
            "--no-op-offload" => a.op_offload = false,
            "--no-host" => a.no_host = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s if !s.starts_with('-') && a.model.as_os_str().is_empty() => a.model = s.into(),
            s => panic!("unknown argument {s}\n\n{USAGE}"),
        }
    }
    if a.model.as_os_str().is_empty() {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    a
}

// ---------------------------------------------------------------- logging --

static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());
static VERBOSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

unsafe extern "C" fn log_cb(_level: sys::ggml_log_level, text: *const libc::c_char, _: *mut c_void) {
    let s = CStr::from_ptr(text).to_string_lossy().into_owned();
    if VERBOSE.load(std::sync::atomic::Ordering::Relaxed) {
        eprint!("{s}");
    }
    LOG.lock().unwrap().push(s);
}

// -------------------------------------------------------------- placement --

/// UI-level subsystem a tensor belongs to (layer, kind).
fn subsystem(name: &str) -> (Option<u32>, &'static str) {
    if let Some(rest) = name.strip_prefix("blk.") {
        let layer = rest.split('.').next().and_then(|l| l.parse().ok());
        let kind = if name.contains("_exps") {
            "routed experts"
        } else if name.contains("_shexp") || name.contains("ffn_gate_inp_shexp") {
            "shared expert"
        } else if name.contains("ffn_gate_inp") || name.contains("exp_probs") {
            "router"
        } else if name.contains("attn_") && !name.contains("norm") {
            "attention"
        } else if name.contains("norm") {
            "norms"
        } else if name.contains("ffn_") {
            "dense ffn"
        } else {
            "other (layer)"
        };
        return (layer, kind);
    }
    if name.starts_with("token_embd") {
        (None, "embeddings")
    } else if name.starts_with("output") {
        (None, "output head")
    } else {
        (None, "other")
    }
}

/// Where llama.cpp put each tensor, reconstructed from its debug log:
/// explicit overrides win, otherwise the tensor follows its layer's device.
struct Placement {
    overridden: BTreeMap<String, String>,
    layer_dev: BTreeMap<u32, String>,
    output_dev: String,
}

fn parse_placement_log(n_layer: u32) -> Placement {
    let log = LOG.lock().unwrap();
    let mut p = Placement { overridden: BTreeMap::new(), layer_dev: BTreeMap::new(), output_dev: "CPU".into() };
    for l in log.iter() {
        // "tensor blk.0.ffn_up_exps.weight (48 MiB f16) buffer type overridden to RustPinned"
        if let Some(rest) = l.strip_prefix("tensor ")
            && let Some((name, tail)) = rest.split_once(' ')
            && let Some((_, buft)) = tail.split_once("buffer type overridden to ")
        {
            p.overridden.insert(name.to_string(), buft.trim().to_string());
        }
        // "load_tensors: layer   3 assigned to device CUDA0, is_swa = 0"
        if let Some((_, rest)) = l.split_once("layer ")
            && let Some((num, tail)) = rest.split_once(" assigned to device ")
            && let Ok(il) = num.trim().parse::<u32>()
        {
            let dev = tail.split(',').next().unwrap_or("?").trim().to_string();
            if il == n_layer {
                p.output_dev = dev;
            } else {
                p.layer_dev.insert(il, dev);
            }
        }
    }
    p
}

fn home_of(name: &str, p: &Placement, mmap: bool) -> String {
    if let Some(b) = p.overridden.get(name) {
        return match b.as_str() {
            "CPU" if mmap => "CPU (mmap'd file)".into(),
            "RustPinned" => "RustPinned RAM".into(),
            other => other.into(),
        };
    }
    let dev = match subsystem(name) {
        (Some(l), _) => p.layer_dev.get(&l).cloned().unwrap_or("CPU".into()),
        (None, "embeddings") => "CPU".into(),
        _ => p.output_dev.clone(),
    };
    if dev == "CPU" && mmap { "CPU (mmap'd file)".into() } else { dev }
}

fn layer_ranges(layers: &BTreeSet<u32>) -> String {
    let mut out: Vec<String> = vec![];
    let v: Vec<u32> = layers.iter().copied().collect();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if i == j { format!("{}", v[i]) } else { format!("{}-{}", v[i], v[j]) });
        i = j + 1;
    }
    out.join(",")
}

#[derive(Default)]
struct Row {
    bytes: usize,
    resident: f64,
    measurable: usize,
    layers: BTreeSet<u32>,
}

/// Placement table: subsystem x home, with mincore residency where measurable.
/// Returns (home -> (bytes, resident fraction)) for the JSON summary.
fn report(title: &str, layout: &[residency::FileTensor], p: &Placement, mmap: bool, mapping: Option<(usize, usize, usize)>, per_layer: bool) -> BTreeMap<String, (usize, f64)> {
    let reg = pinned::REGISTRY.lock().unwrap();
    let pinned_at: BTreeMap<&str, (usize, usize)> =
        reg.iter().filter(|b| !b.freed).flat_map(|b| b.tensors.iter().map(|(n, a, s)| (n.as_str(), (*a, *s)))).collect();

    let mut rows: BTreeMap<(String, String), Row> = BTreeMap::new();
    let mut homes: BTreeMap<String, Row> = BTreeMap::new();
    for t in layout {
        let (layer, kind) = subsystem(&t.name);
        let home = if pinned_at.contains_key(t.name.as_str()) { "RustPinned RAM".to_string() } else { home_of(&t.name, p, mmap) };
        let frac = if let Some((addr, size)) = pinned_at.get(t.name.as_str()) {
            Some(residency::resident_fraction(*addr, *size))
        } else if home == "CPU (mmap'd file)" {
            mapping.map(|(origin, _, _)| residency::resident_fraction(origin + t.file_offset, t.size))
        } else {
            None
        };
        let kind = match (per_layer, layer) {
            (true, Some(l)) => format!("{kind} blk.{l}"),
            _ => kind.to_string(),
        };
        for r in [rows.entry((kind, home.clone())).or_default(), homes.entry(home).or_default()] {
            r.bytes += t.size;
            if let Some(l) = layer {
                r.layers.insert(l);
            }
            if let Some(f) = frac.filter(|f| !f.is_nan()) {
                r.resident += f * t.size as f64;
                r.measurable += t.size;
            }
        }
    }

    let pct = |r: &Row| if r.measurable == 0 { "-".to_string() } else { format!("{:.0}%", 100.0 * r.resident / r.measurable as f64) };
    println!("\n== {title} ==");
    println!("{:<24} {:<20} {:>10} {:>7}  layers", "subsystem", "lives in", "MiB", "in RAM");
    for ((kind, home), r) in &rows {
        println!("{:<24} {:<20} {:>10.1} {:>7}  {}", kind, home, r.bytes as f64 / 1048576.0, pct(r), layer_ranges(&r.layers));
    }
    println!("{:-<24} {:-<20} {:->10} {:->7}", "", "", "", "");
    for (home, r) in &homes {
        println!("{:<24} {:<20} {:>10.1} {:>7}", "total", home, r.bytes as f64 / 1048576.0, pct(r));
    }
    println!("process: VmRSS={}  VmLck={}  (\"in RAM\" = mincore; '-' = not host memory or not measurable)", residency::proc_status("VmRSS:"), residency::proc_status("VmLck:"));
    homes.into_iter().map(|(h, r)| {
        let f = if r.measurable == 0 { f64::NAN } else { r.resident / r.measurable as f64 };
        (h, (r.bytes, f))
    }).collect()
}

// ----------------------------------------------------------- ground truth --

/// Where the rules say a tensor should go, in the placement table's vocabulary.
/// `None` = no rule matched, so llama.cpp's own layer assignment decides.
fn intended_home(name: &str, rules: &[(regex::Regex, String)], mmap: bool) -> Option<String> {
    let (_, dest) = rules.iter().find(|(re, _)| re.is_match(name))?;
    Some(match dest.as_str() {
        "pinned" => "RustPinned RAM".into(),
        "CPU" | "cpu" if mmap => "CPU (mmap'd file)".into(),
        "CPU" | "cpu" => "CPU".into(),
        dev => dev.into(),
    })
}

/// Compare what the graph actually read against the rules and against the
/// log-derived table. Prints a table and returns a JSON object.
fn verify_report(
    obs: &verify::Observed, layout: &[residency::FileTensor], p: &Placement, rules: &[(String, String)], mmap: bool,
    mapping: Option<(usize, usize, usize)>, prefill_tokens: usize,
) -> String {
    let rules: Vec<(regex::Regex, String)> = rules.iter().map(|(r, d)| (regex::Regex::new(r).expect("bad --place regex"), d.clone())).collect();
    let reg = pinned::REGISTRY.lock().unwrap();
    let pinned_ranges: Vec<(usize, usize)> = reg.iter().filter(|b| !b.freed).map(|b| (b.base, b.len)).collect();
    let pinned_names: BTreeSet<&str> = reg.iter().filter(|b| !b.freed).flat_map(|b| b.tensors.iter().map(|(n, _, _)| n.as_str())).collect();
    let size_of: BTreeMap<&str, usize> = layout.iter().map(|t| (t.name.as_str(), t.size)).collect();
    let mib = |b: usize| b as f64 / 1048576.0;

    // (subsystem, intended, actual) -> (bytes, tensors, claim mismatches)
    let mut rows: BTreeMap<(String, String, String), (usize, usize, usize)> = BTreeMap::new();
    let mut actual_tot: BTreeMap<String, usize> = BTreeMap::new();
    let (mut rule_bad, mut rule_bad_b, mut claim_bad, mut claim_bad_b) = (0, 0usize, 0, 0usize);
    let mut examples: Vec<String> = vec![];
    let mut unseen: Vec<&str> = vec![];
    for t in layout {
        let Some(seen) = obs.weights.get(&t.name) else {
            unseen.push(&t.name);
            continue;
        };
        let actual = verify::actual_home(seen, mapping, &pinned_ranges);
        let claimed = if pinned_names.contains(t.name.as_str()) { "RustPinned RAM".to_string() } else { home_of(&t.name, p, mmap) };
        let intended = intended_home(&t.name, &rules, mmap);
        let rule_ok = intended.as_deref().is_none_or(|i| i == actual);
        if !rule_ok {
            rule_bad += 1;
            rule_bad_b += t.size;
            if examples.len() < 5 {
                examples.push(format!("{} wanted {} got {}", t.name, intended.as_deref().unwrap_or("-"), actual));
            }
        }
        let claim_ok = claimed == actual;
        if !claim_ok {
            claim_bad += 1;
            claim_bad_b += t.size;
        }
        let r = rows.entry((subsystem(&t.name).1.to_string(), intended.unwrap_or_else(|| "(llama.cpp default)".into()), actual.clone())).or_default();
        r.0 += t.size;
        r.1 += 1;
        r.2 += (!claim_ok) as usize;
        *actual_tot.entry(actual).or_default() += t.size;
    }

    println!("\n== ground truth: buffer and address of every weight the graph read ==");
    println!("{:<16} {:<20} {:<20} {:>10} {:>5}  note", "subsystem", "rule says", "actually in", "MiB", "n");
    for ((kind, intended, actual), (b, n, cbad)) in &rows {
        let mut note = vec![];
        if intended != "(llama.cpp default)" && intended != actual {
            note.push("RULE NOT HONOURED".to_string());
        }
        if *cbad > 0 {
            note.push(format!("table claimed otherwise for {cbad}"));
        }
        println!("{:<16} {:<20} {:<20} {:>10.1} {:>5}  {}", kind, intended, actual, mib(*b), n, note.join("; "));
    }
    println!(
        "rules honoured: {}  ({} tensors / {:.1} MiB not where the rule said)",
        if rule_bad == 0 { "yes" } else { "NO" }, rule_bad, mib(rule_bad_b)
    );
    for e in &examples {
        println!("  e.g. {e}");
    }
    println!("log-derived table accurate: {}  ({} tensors / {:.1} MiB misreported)", if claim_bad == 0 { "yes" } else { "NO" }, claim_bad, mib(claim_bad_b));
    if !unseen.is_empty() {
        println!("not read by the graph (unverified): {} tensors, e.g. {}", unseen.len(), unseen[..unseen.len().min(3)].join(", "));
    }

    let mut copies_json = vec![];
    for (phase, by_w) in &obs.copies {
        let mut per_dev: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for (w, devs) in by_w {
            for d in devs {
                let e = per_dev.entry(d.as_str()).or_default();
                e.0 += size_of.get(w.as_str()).copied().unwrap_or(0);
                e.1 += 1;
            }
        }
        let tokens = if *phase == "prefill" { prefill_tokens } else { 1 };
        for (d, (b, n)) in &per_dev {
            println!("run-time copy ({phase}, {tokens}-token batch): {n} weights / {:.1} MiB shipped host -> {d} per batch", mib(*b));
        }
        copies_json.push(format!(
            "{}:{{{}}}",
            json_str(phase),
            per_dev.iter().map(|(d, (b, _))| format!("{}:{:.1}", json_str(d), mib(*b))).collect::<Vec<_>>().join(",")
        ));
    }
    for (phase, subs) in &obs.substitutes {
        for (buf, (names, bufs)) in subs {
            println!(
                "substituted ({phase}): {} expert tensors read from {buf} (MoE cache, {:.1} MiB in {} buffers) instead of the placed weight",
                names.len(), mib(bufs.values().sum()), bufs.len()
            );
        }
    }

    format!(
        "{{\"rules_honoured\":{},\"rule_mismatch_tensors\":{},\"rule_mismatch_mib\":{:.1},\"table_accurate\":{},\"table_mismatch_mib\":{:.1},\"unseen\":{},\"actual\":{{{}}},\"copied_mib\":{{{}}}}}",
        rule_bad == 0,
        rule_bad,
        mib(rule_bad_b),
        claim_bad == 0,
        mib(claim_bad_b),
        unseen.len(),
        actual_tot.iter().map(|(h, b)| format!("{}:{:.1}", json_str(h), mib(*b))).collect::<Vec<_>>().join(","),
        copies_json.join(",")
    )
}

// ---------------------------------------------------------------- devices --

fn devices() -> Vec<(String, String, usize, usize)> {
    unsafe {
        (0..sys::ggml_backend_dev_count())
            .map(|i| {
                let d = sys::ggml_backend_dev_get(i);
                let (mut free, mut total) = (0usize, 0usize);
                sys::ggml_backend_dev_memory(d, &mut free, &mut total);
                (
                    CStr::from_ptr(sys::ggml_backend_dev_name(d)).to_string_lossy().into_owned(),
                    CStr::from_ptr(sys::ggml_backend_dev_description(d)).to_string_lossy().into_owned(),
                    free,
                    total,
                )
            })
            .collect()
    }
}

fn buft_for(dest: &str, pinned: &mut Option<sys::ggml_backend_buffer_type_t>, opts: &pinned::PinOptions) -> sys::ggml_backend_buffer_type_t {
    match dest {
        "pinned" => *pinned.get_or_insert_with(|| pinned::new_buffer_type("RustPinned", opts.clone())),
        "CPU" | "cpu" => unsafe { sys::ggml_backend_cpu_buffer_type() },
        dev => unsafe {
            let c = CString::new(dev).unwrap();
            let d = sys::ggml_backend_dev_by_name(c.as_ptr());
            if d.is_null() {
                let names: Vec<String> = devices().into_iter().map(|d| d.0).collect();
                panic!("no device named {dev}; available: {names:?} (plus 'pinned' and 'CPU')");
            }
            sys::ggml_backend_dev_buffer_type(d)
        },
    }
}

// --------------------------------------------------------------- workload --

struct Run {
    prefill_tps: f64,
    decode_tps: f64,
    tokens: Vec<i32>,
    logits_fnv: u64,
}

const PROMPT_SEED: &str = "The engine decides where each part of the model lives: attention on the fastest GPU, \
hot experts in local RAM, cold experts streamed from NVMe. Placement is a plan the operator can see and edit. ";

unsafe fn tokenize(vocab: *const sys::llama_vocab, n: usize) -> Vec<i32> {
    let text = PROMPT_SEED.repeat(n / 8 + 4);
    let mut toks = vec![0i32; text.len() + 8];
    let mut got = sys::llama_tokenize(vocab, text.as_ptr() as *const _, text.len() as i32, toks.as_mut_ptr(), toks.len() as i32, true, false);
    if got < 0 {
        // negative = buffer too small; -got is the size needed
        toks.resize((-got) as usize, 0);
        got = sys::llama_tokenize(vocab, text.as_ptr() as *const _, text.len() as i32, toks.as_mut_ptr(), toks.len() as i32, true, false);
    }
    assert!(got > 0, "tokenize failed");
    toks.truncate((got as usize).min(n.max(1)));
    toks
}

unsafe fn run(ctx: *mut sys::llama_context, vocab: *const sys::llama_vocab, prompt: &[i32], n_new: usize, batch: usize) -> Run {
    let n_vocab = sys::llama_vocab_n_tokens(vocab) as usize;
    sys::llama_memory_clear(sys::llama_get_memory(ctx), true);
    let mut fnv: u64 = 0xcbf29ce484222325;
    let mut hash = |logits: &[f32]| {
        for l in logits {
            for b in l.to_bits().to_le_bytes() {
                fnv = (fnv ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
    };

    let mut buf = prompt.to_vec();
    let t0 = Instant::now();
    for chunk in buf.chunks_mut(batch) {
        assert_eq!(sys::llama_decode(ctx, sys::llama_batch_get_one(chunk.as_mut_ptr(), chunk.len() as i32)), 0, "prefill decode failed");
    }
    sys::llama_synchronize(ctx);
    let prefill_s = t0.elapsed().as_secs_f64();

    let mut out = vec![];
    let t1 = Instant::now();
    for i in 0..n_new {
        let logits = std::slice::from_raw_parts(sys::llama_get_logits_ith(ctx, -1), n_vocab);
        hash(logits);
        let mut next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as i32;
        out.push(next);
        if i + 1 < n_new {
            assert_eq!(sys::llama_decode(ctx, sys::llama_batch_get_one(&mut next, 1)), 0, "decode failed");
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    Run { prefill_tps: prompt.len() as f64 / prefill_s, decode_tps: (n_new.saturating_sub(1)) as f64 / decode_s, tokens: out, logits_fnv: fnv }
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

// ------------------------------------------------------------------- main --

fn main() {
    let a = parse_args();
    VERBOSE.store(a.verbose, std::sync::atomic::Ordering::Relaxed);
    unsafe {
        sys::llama_log_set(Some(log_cb), std::ptr::null_mut());
        sys::llama_backend_init();
    }

    println!("devices:");
    for (name, desc, free, total) in devices() {
        println!("  {name:<8} {desc:<40} {:>8.0} / {:>8.0} MiB free", free as f64 / 1048576.0, total as f64 / 1048576.0);
    }

    let layout = residency::gguf_layout(&a.model);
    let n_layer = layout.iter().filter_map(|t| subsystem(&t.name).0).max().map(|l| l + 1).unwrap_or(0);

    // --- placement rules -> tensor_buft_overrides, exactly what the UI would emit ---
    let pin_opts = pinned::PinOptions { numa_node: a.numa, mlock: a.mlock, hugepages: true, cuda_register: a.cuda_register };
    let mut pin_buft = None;
    let patterns: Vec<CString> = a.rules.iter().map(|(re, _)| CString::new(re.as_str()).unwrap()).collect();
    let mut overrides: Vec<sys::llama_model_tensor_buft_override> = a
        .rules
        .iter()
        .zip(&patterns)
        .map(|((_, dest), pat)| sys::llama_model_tensor_buft_override { pattern: pat.as_ptr(), buft: buft_for(dest, &mut pin_buft, &pin_opts) })
        .collect();
    overrides.push(sys::llama_model_tensor_buft_override { pattern: std::ptr::null(), buft: std::ptr::null_mut() });
    let mut split = vec![0f32; unsafe { sys::llama_max_devices() }];
    for (i, s) in a.tensor_split.iter().enumerate() {
        split[i] = *s;
    }

    let model_path = CString::new(a.model.to_str().unwrap()).unwrap();
    let t0 = Instant::now();
    let model = unsafe {
        let mut mp = sys::llama_model_default_params();
        mp.tensor_buft_overrides = overrides.as_ptr();
        mp.use_extra_bufts = a.extra_bufts;
        mp.no_host = a.no_host;
        mp.n_gpu_layers = a.gpu_layers;
        if !a.tensor_split.is_empty() {
            mp.tensor_split = split.as_ptr();
        }
        mp.load_mode = match a.load_mode.as_str() {
            "mmap" => sys::llama_load_mode_LLAMA_LOAD_MODE_MMAP,
            "none" => sys::llama_load_mode_LLAMA_LOAD_MODE_NONE,
            m => panic!("unknown load mode {m}"),
        };
        sys::llama_model_load_from_file(model_path.as_ptr(), mp)
    };
    assert!(!model.is_null(), "model load failed; rerun with --verbose");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("\nloaded {} in {:.0} ms (load_mode={}, gpu_layers={}, rules={:?})", a.model.display(), load_ms, a.load_mode, a.gpu_layers, a.rules);
    for l in LOG.lock().unwrap().iter().filter(|l| l.contains("model buffer size") || l.contains("cannot be used with preferred")) {
        print!("  llama.cpp: {}", l.split_once(": ").map(|x| x.1).unwrap_or(l).trim_start());
    }
    for (i, b) in pinned::REGISTRY.lock().unwrap().iter().enumerate() {
        println!(
            "  RustPinned buffer #{i}: {:.1} MiB, {} tensors, mlock={:?} mbind={:?} cuda_register={:?} page NUMA nodes={:?}",
            b.len as f64 / 1048576.0, b.tensors.len(), b.mlock_result, b.mbind_result, b.cuda_register_result, residency::numa_nodes(b.base, b.len)
        );
    }

    let mmap = a.load_mode == "mmap";
    let placement = parse_placement_log(n_layer);
    let mapping = if mmap { residency::find_file_mapping(&a.model) } else { None };
    report("placement after load", &layout, &placement, mmap, mapping, a.per_layer);

    let make_ctx = |n_ctx: u32, observe: bool| unsafe {
        let mut cp = sys::llama_context_default_params();
        cp.n_ctx = n_ctx;
        if observe {
            cp.cb_eval = Some(verify::eval_cb);
        }
        cp.n_batch = a.batch;
        cp.n_ubatch = a.batch;
        cp.n_threads = a.threads;
        cp.n_threads_batch = a.threads;
        cp.moe_cache_size = a.moe_cache_mib * 1024 * 1024;
        cp.op_offload = a.op_offload;
        cp.flash_attn_type = match a.flash_attn.as_str() {
            "on" => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED,
            "off" => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
            _ => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
        };
        sys::llama_init_from_model(model, cp)
    };
    let vocab = unsafe { sys::llama_model_get_vocab(model) };
    let prompt = unsafe { tokenize(vocab, a.prompt_tokens) };

    // Ground truth, in its own context: the eval callback makes the scheduler
    // run node by node, so it must not touch the timed runs.
    let verified = a.verify.then(|| {
        let vctx = make_ctx((a.batch as usize + 64) as u32, true);
        assert!(!vctx.is_null(), "verify context creation failed; rerun with --verbose");
        verify::begin(layout.iter().map(|t| t.name.clone()));
        unsafe {
            let mut toks: Vec<i32> = prompt[..prompt.len().min(a.batch as usize)].to_vec();
            verify::set_phase("prefill");
            assert_eq!(sys::llama_decode(vctx, sys::llama_batch_get_one(toks.as_mut_ptr(), toks.len() as i32)), 0, "verify prefill failed");
            verify::set_phase("decode");
            let mut next = toks[0];
            assert_eq!(sys::llama_decode(vctx, sys::llama_batch_get_one(&mut next, 1)), 0, "verify decode failed");
            sys::llama_synchronize(vctx);
            sys::llama_free(vctx);
        }
        verify_report(&verify::finish(), &layout, &placement, &a.rules, mmap, mapping, prompt.len().min(a.batch as usize))
    });

    let ctx = make_ctx(a.ctx.unwrap_or((a.prompt_tokens + a.tokens + 64) as u32), false);
    assert!(!ctx.is_null(), "context creation failed; rerun with --verbose");

    // Warm-up (CUDA graph capture, kernel loading, first page faults) so it isn't billed to the timed run.
    unsafe { run(ctx, vocab, &prompt[..prompt.len().min(32)], 4, a.batch as usize) };
    let runs: Vec<Run> = (0..a.repeat).map(|_| unsafe { run(ctx, vocab, &prompt, a.tokens, a.batch as usize) }).collect();
    assert!(runs.iter().all(|r| r.logits_fnv == runs[0].logits_fnv), "repeated runs disagree: nondeterministic output");
    let median = |f: &dyn Fn(&Run) -> f64| {
        let mut v: Vec<f64> = runs.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let (prefill_med, decode_med) = (median(&|r| r.prefill_tps), median(&|r| r.decode_tps));
    let spread = |f: &dyn Fn(&Run) -> f64| {
        let v: Vec<f64> = runs.iter().map(f).collect();
        format!("{:.1}-{:.1}", v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(0.0, f64::max))
    };
    let warm = &runs[0];
    let homes = report("after run (warm)", &layout, &placement, mmap, mapping, a.per_layer);
    println!(
        "prefill {:.1} tok/s ({} tokens), decode {:.2} tok/s ({} tokens)  [median of {}; ranges {} / {}]",
        prefill_med, prompt.len(), decode_med, a.tokens, a.repeat, spread(&|r| r.prefill_tps), spread(&|r| r.decode_tps)
    );
    println!("first tokens: {:?}", &warm.tokens[..warm.tokens.len().min(12)]);

    let mut cold = None;
    if a.evict {
        if let Some((_, s, e)) = mapping {
            residency::page_out(s, e - s);
        }
        for b in pinned::REGISTRY.lock().unwrap().iter().filter(|b| !b.freed) {
            residency::page_out(b.base, b.len);
        }
        report("after MADV_PAGEOUT on everything", &layout, &placement, mmap, mapping, a.per_layer);
        let r = unsafe { run(ctx, vocab, &prompt, a.tokens, a.batch as usize) };
        report("after run (cold)", &layout, &placement, mmap, mapping, a.per_layer);
        println!("cold: prefill {:.1} tok/s, decode {:.2} tok/s, identical logits: {}", r.prefill_tps, r.decode_tps, r.logits_fnv == warm.logits_fnv);
        cold = Some(r);
    }

    // One machine-readable line per run, for bench scripts.
    let homes_json: Vec<String> = homes
        .iter()
        .map(|(h, (b, f))| format!("{}:{{\"mib\":{:.1},\"in_ram\":{}}}", json_str(h), *b as f64 / 1048576.0, if f.is_nan() { "null".into() } else { format!("{f:.3}") }))
        .collect();
    let rules_json: Vec<String> = a.rules.iter().map(|(r, d)| format!("[{},{}]", json_str(r), json_str(d))).collect();
    println!(
        "RESULT {{\"label\":{},\"model\":{},\"rules\":[{}],\"gpu_layers\":{},\"extra_bufts\":{},\"load_mode\":{},\"load_ms\":{:.0},\"prompt_tokens\":{},\"gen_tokens\":{},\"prefill_tps\":{:.2},\"decode_tps\":{:.3},\"logits_fnv\":\"{:016x}\",\"homes\":{{{}}}{}{}}}",
        json_str(&a.label),
        json_str(&a.model.display().to_string()),
        rules_json.join(","),
        a.gpu_layers,
        a.extra_bufts,
        json_str(&a.load_mode),
        load_ms,
        prompt.len(),
        a.tokens,
        prefill_med,
        decode_med,
        warm.logits_fnv,
        homes_json.join(","),
        cold.map(|c| format!(",\"cold_prefill_tps\":{:.2},\"cold_decode_tps\":{:.3},\"cold_identical\":{}", c.prefill_tps, c.decode_tps, c.logits_fnv == warm.logits_fnv)).unwrap_or_default(),
        verified.map(|v| format!(",\"verify\":{v}")).unwrap_or_default()
    );

    unsafe {
        sys::llama_free(ctx);
        sys::llama_model_free(model);
    }
}
