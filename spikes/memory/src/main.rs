//! Memory spike: load a MoE model with attention on the GPUs and a share of
//! the routed experts in the RAM tier (the Rust pinned buffer), with chosen
//! lock / huge-page options; report what the kernel actually did with the
//! buffer and how fast inference runs.
//!
//!   memory-spike MODEL.gguf [--ram-share F] [--no-mlock] [--thp on|off]
//!                [--hugetlb 2M|1G] [--cuda-register] [--threads N]
//!                [--prompt N] [--tokens N] [--repeat N] [--label L]
//!                [--hold SECONDS]
//!
//! --ram-share: fraction of the experts in RAM; the rest stay in the mapped
//! file (disk tier). Prints a RESULT {json} line.

use aie_anatomy::{Anatomy, Subsystem};
use aie_llama::context::{Context, ContextParams, Sampling};
use aie_llama::pinned::{PinOptions, REGISTRY};
use aie_llama::{LoadOptions, Model};
use aie_plan::{Amount, Intent, Share, Tier};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Instant;

struct Args {
    model: String,
    ram_share: f64,
    pin: PinOptions,
    threads: i32,
    prompt: usize,
    tokens: usize,
    repeat: usize,
    label: String,
    hold: u64,
}

fn parse() -> Args {
    let mut a = Args {
        model: String::new(),
        ram_share: 1.0,
        pin: PinOptions { mlock: true, hugepages: true, ..Default::default() },
        threads: 8,
        prompt: 512,
        tokens: 64,
        repeat: 3,
        label: String::new(),
        hold: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        let mut val = || it.next().expect("missing value");
        match x.as_str() {
            "--ram-share" => a.ram_share = val().parse().unwrap(),
            "--no-mlock" => a.pin.mlock = false,
            "--thp" => a.pin.hugepages = val() == "on",
            "--hugetlb" => {
                a.pin.hugetlb = Some(match val().as_str() {
                    "2M" => 2 << 20,
                    "1G" => 1 << 30,
                    v => panic!("--hugetlb 2M|1G, got {v}"),
                })
            }
            "--cuda-register" => a.pin.cuda_register = true,
            "--threads" => a.threads = val().parse().unwrap(),
            "--prompt" => a.prompt = val().parse().unwrap(),
            "--tokens" => a.tokens = val().parse().unwrap(),
            "--repeat" => a.repeat = val().parse().unwrap(),
            "--label" => a.label = val(),
            "--hold" => a.hold = val().parse().unwrap(),
            m if !m.starts_with('-') => a.model = m.into(),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(!a.model.is_empty(), "usage: memory-spike MODEL.gguf [options]");
    a
}

/// smaps fields summed over the VMAs inside [base, base+len).
fn smaps(base: usize, len: usize) -> Value {
    let text = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
    let mut inside = false;
    let mut kib = std::collections::BTreeMap::<String, u64>::new();
    let mut page_kib = 0;
    for line in text.lines() {
        if let Some((range, _)) = line.split_once(' ')
            && let Some((s, e)) = range.split_once('-')
            && let (Ok(s), Ok(e)) = (usize::from_str_radix(s, 16), usize::from_str_radix(e, 16))
        {
            inside = s >= base && e <= base + len;
            continue;
        }
        if !inside {
            continue;
        }
        let mut f = line.split_whitespace();
        if let (Some(k), Some(v)) = (f.next(), f.next())
            && let Ok(v) = v.parse::<u64>()
        {
            let k = k.trim_end_matches(':').to_string();
            if k == "KernelPageSize" {
                page_kib = page_kib.max(v);
            } else if ["Rss", "AnonHugePages", "Locked", "Swap", "Private_Hugetlb", "Shared_Hugetlb"].contains(&k.as_str()) {
                *kib.entry(k).or_default() += v;
            }
        }
    }
    let mib = |k: &str| kib.get(k).copied().unwrap_or(0) as f64 / 1024.0;
    json!({
        "rss_mib": mib("Rss") + mib("Private_Hugetlb") + mib("Shared_Hugetlb"),
        "thp_mib": mib("AnonHugePages"),
        "hugetlb_mib": mib("Private_Hugetlb") + mib("Shared_Hugetlb"),
        "locked_mib": mib("Locked"),
        "swap_mib": mib("Swap"),
        "page_kib": page_kib,
    })
}

fn buffers() -> Vec<Value> {
    REGISTRY
        .lock()
        .unwrap()
        .iter()
        .filter(|b| !b.freed && b.len > 0)
        .map(|b| {
            json!({
                "mib": b.len as f64 / 1048576.0,
                "mlock": format!("{:?}", b.mlock_result),
                "hugetlb": format!("{:?}", b.hugetlb_result),
                "cuda_register": format!("{:?}", b.cuda_register_result),
                "smaps": smaps(b.base, b.map_len),
            })
        })
        .collect()
}

fn proc_status(field: &str) -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with(field))
        .map(|l| l[field.len()..].trim().to_string())
        .unwrap_or_default()
}

fn main() {
    let a = parse();
    let path = std::path::Path::new(&a.model);
    let anatomy = Anatomy::from_gguf(&aie_gguf::Gguf::open(path).unwrap()).unwrap();
    let gpus = aie_llama::devices().iter().filter(|d| d.is_gpu).count() as u32;

    // layers split across the GPUs (or RAM without GPUs); experts: a share
    // in RAM, the rest on disk
    let mut intent = Intent::all_on(if gpus > 0 { Tier::Gpu(0) } else { Tier::Ram }, (a.prompt + a.tokens + 64) as u64);
    if gpus > 1 {
        intent.layers = (0..gpus)
            .map(|g| if g + 1 == gpus { Share::all(Tier::Gpu(g)) } else { Share::new(Tier::Gpu(g), Amount::Fraction(1.0 / gpus as f64)) })
            .collect();
    }
    let experts = if a.ram_share >= 1.0 {
        vec![Share::all(Tier::Ram)]
    } else {
        vec![Share::new(Tier::Ram, Amount::Fraction(a.ram_share)), Share::all(Tier::Disk)]
    };
    let bulk = if anatomy.is_moe() { Subsystem::RoutedExperts } else { Subsystem::DenseFfn };
    intent.subsystems.insert(bulk, experts);
    let plan = aie_plan::plan(&anatomy, &intent, gpus).unwrap();
    let settings = aie_plan::llama::compile(&anatomy, &plan, gpus).unwrap();
    let by_tier = plan.subsystem_by_tier(&anatomy, bulk);

    let t0 = Instant::now();
    let model = Arc::new(Model::load(path, &LoadOptions { settings, pin: a.pin.clone() }).expect("load"));
    let load_s = t0.elapsed().as_secs_f64();
    let after_load = buffers();
    eprintln!("loaded in {load_s:.1} s; buffers: {after_load:#?}");

    let cp = ContextParams { n_ctx: (a.prompt + a.tokens + 64) as u32, n_batch: 512, threads: a.threads, ..Default::default() };
    let mut ctx = Context::new(model.clone(), cp).unwrap();
    let text = "The memory spike measures how the RAM tier behaves under pressure. ".repeat(a.prompt / 10 + 8);
    let mut prompt = model.tokenize(&text, true).unwrap();
    prompt.truncate(a.prompt);
    ctx.generate(&prompt[..32.min(prompt.len())], 4, &Sampling::greedy(), |_| true).unwrap(); // warm-up
    let mut runs = Vec::new();
    for _ in 0..a.repeat {
        let g = ctx.generate(&prompt, a.tokens, &Sampling::greedy(), |_| true).unwrap();
        runs.push((g.prefill_tps(), g.decode_tps()));
        eprintln!("prefill {:.1} tok/s, decode {:.2} tok/s", g.prefill_tps(), g.decode_tps());
    }
    let median = |f: fn(&(f64, f64)) -> f64| {
        let mut v: Vec<f64> = runs.iter().map(f).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    if a.hold > 0 {
        eprintln!("holding {} s for outside measurement", a.hold);
        std::thread::sleep(std::time::Duration::from_secs(a.hold));
    }
    println!(
        "RESULT {}",
        json!({
            "label": a.label,
            "pin": { "mlock": a.pin.mlock, "thp": a.pin.hugepages, "hugetlb": a.pin.hugetlb, "cuda_register": a.pin.cuda_register },
            "experts_mib": by_tier.iter().map(|(t, b)| (format!("{t}"), *b as f64 / 1048576.0)).collect::<std::collections::BTreeMap<_, _>>(),
            "load_s": load_s,
            "prefill_tps": median(|r| r.0),
            "decode_tps": median(|r| r.1),
            "runs": runs,
            "buffers_after_load": after_load,
            "buffers_after_runs": buffers(),
            "VmLck": proc_status("VmLck:"),
            "VmSwap": proc_status("VmSwap:"),
            "VmRSS": proc_status("VmRSS:"),
        })
    );
}
