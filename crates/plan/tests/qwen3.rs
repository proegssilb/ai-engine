#![forbid(unsafe_code)]

use aie_anatomy::{Anatomy, Subsystem};
use aie_gguf::Gguf;
use aie_plan::fit::{self, Capacity, Machine, Overheads};
use aie_plan::llama::{self, BufDest};
use aie_plan::{Amount, Intent, PlanError, Share, Tier, plan};
use std::collections::BTreeMap;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

fn model(name: &str) -> Anatomy {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name);
    Anatomy::from_gguf(&Gguf::open_header(p).unwrap()).unwrap()
}

fn moe() -> Anatomy {
    model("qwen3-30b-a3b-q4_k_m.header.gguf")
}

/// The GPU test box: 2x 32 GB, 47 GB RAM.
fn test_box() -> Machine {
    let gpu = |i| Capacity {
        name: format!("CUDA{i}"),
        total: 32 * GIB,
        free: 32 * GIB - 300 * MIB,
    };
    Machine {
        gpus: vec![gpu(0), gpu(1)],
        ram: Capacity {
            name: "RAM".into(),
            total: 47 * GIB,
            free: 44 * GIB,
        },
    }
}

/// plan -> compile -> what llama.cpp will do must equal the plan, tensor by tensor.
fn assert_round_trip(a: &Anatomy, intent: &Intent, gpus: u32) -> llama::Settings {
    let p = plan(a, intent, gpus).unwrap();
    let s = llama::compile(a, &p, gpus).unwrap();
    let got = llama::resolve(a, &s).unwrap();
    for (name, want) in &p.tensors {
        assert_eq!(got[name], *want, "{name}");
    }
    assert_eq!(llama::resolve_layer_homes(a, &s), p.layer_home);
    s
}

fn experts_split(gpu: f64, ram: f64) -> Intent {
    let mut i = Intent::all_on(Tier::Gpu(0), 4096);
    i.layers = vec![
        Share::new(Tier::Gpu(0), Amount::Fraction(0.5)),
        Share::all(Tier::Gpu(1)),
    ];
    i.subsystems.insert(
        Subsystem::RoutedExperts,
        vec![
            Share::new(Tier::Gpu(1), Amount::Fraction(gpu)),
            Share::new(Tier::Ram, Amount::Fraction(ram)),
            Share::all(Tier::Disk),
        ],
    );
    i
}

#[test]
fn all_on_one_gpu_needs_no_rules() {
    let a = moe();
    let s = assert_round_trip(&a, &Intent::all_on(Tier::Gpu(0), 4096), 2);
    assert_eq!(s.n_gpu_layers, 49);
    assert_eq!(s.tensor_split, vec![49.0, 0.0]);
    // only the embeddings stay on the CPU (mapped), which is llama.cpp's default
    assert!(s.overrides.is_empty(), "{:?}", s.overrides);
}

#[test]
fn even_split_across_two_gpus() {
    let a = moe();
    let mut i = Intent::all_on(Tier::Gpu(0), 4096);
    i.layers = vec![
        Share::new(Tier::Gpu(0), Amount::Fraction(0.5)),
        Share::new(Tier::Gpu(1), Amount::Fraction(0.5)),
    ];
    let s = assert_round_trip(&a, &i, 2);
    assert_eq!(s.tensor_split, vec![24.0, 25.0]);
    assert!(s.overrides.is_empty());
}

#[test]
fn experts_ten_eighty_ten() {
    let a = moe();
    let i = experts_split(0.1, 0.8);
    let s = assert_round_trip(&a, &i, 2);
    let p = plan(&a, &i, 2).unwrap();

    let by = p.subsystem_by_tier(&a, Subsystem::RoutedExperts);
    let total: u64 = by.values().sum();
    let share = |t| by.get(&t).copied().unwrap_or(0) as f64 / total as f64;
    // whole layers of 324-373.5 MiB out of 16740 MiB: within about one layer (~2.2%)
    assert!(
        (share(Tier::Gpu(1)) - 0.1).abs() < 0.025,
        "gpu {}",
        share(Tier::Gpu(1))
    );
    assert!(
        (share(Tier::Ram) - 0.8).abs() < 0.025,
        "ram {}",
        share(Tier::Ram)
    );
    assert!(
        (share(Tier::Disk) - 0.1).abs() < 0.025,
        "disk {}",
        share(Tier::Disk)
    );

    // GPU experts come from layers already homed on GPU 1 (no extra hop)
    for part in a.parts_of(Subsystem::RoutedExperts) {
        if p.tensors[&part.name] == Tier::Gpu(1) {
            assert_eq!(
                p.layer_home[part.layer.unwrap() as usize],
                Tier::Gpu(1),
                "{}",
                part.name
            );
        }
    }
    // rules are grouped: a handful, not one per tensor
    assert!(s.overrides.len() <= 12, "{} rules", s.overrides.len());
    assert!(s.overrides.iter().any(|o| o.dest == BufDest::Pinned));
    assert!(s.overrides.iter().any(|o| o.dest == BufDest::Cpu));
}

#[test]
fn ram_homed_layers_go_first_and_are_pinned() {
    let a = moe();
    let mut i = Intent::all_on(Tier::Gpu(0), 4096);
    i.layers = vec![
        Share::new(Tier::Gpu(0), Amount::Fraction(0.75)),
        Share::all(Tier::Ram),
    ];
    let s = assert_round_trip(&a, &i, 1);
    let p = plan(&a, &i, 1).unwrap();
    assert_eq!(p.layer_counts()[&Tier::Ram], 12);
    assert!(p.layer_home[..12].iter().all(|t| *t == Tier::Ram));
    assert_eq!(s.n_gpu_layers, 37);
    // everything in those 12 layers needs an explicit pinned rule
    assert!(
        p.tensors
            .iter()
            .filter(|(n, _)| n.starts_with("blk.0."))
            .all(|(_, t)| *t == Tier::Ram)
    );
}

#[test]
fn dense_model_ffn_split() {
    let a = model("qwen3-8b-q4_k_m.header.gguf");
    let mut i = Intent::all_on(Tier::Gpu(0), 8192);
    i.subsystems.insert(
        Subsystem::DenseFfn,
        vec![
            Share::new(Tier::Gpu(0), Amount::Fraction(0.5)),
            Share::all(Tier::Ram),
        ],
    );
    i.embeddings = Some(Tier::Ram);
    assert_round_trip(&a, &i, 1);
}

#[test]
fn bytes_amounts() {
    let a = moe();
    let mut i = experts_split(0.0, 0.0);
    i.subsystems.insert(
        Subsystem::RoutedExperts,
        vec![
            Share::new(Tier::Gpu(1), Amount::Bytes(4 * GIB)),
            Share::all(Tier::Ram),
        ],
    );
    let p = plan(&a, &i, 2).unwrap();
    let on_gpu = p.subsystem_by_tier(&a, Subsystem::RoutedExperts)[&Tier::Gpu(1)];
    assert!(on_gpu.abs_diff(4 * GIB) < 380 * MIB, "{} MiB", on_gpu / MIB);
    assert_round_trip(&a, &i, 2);
}

#[test]
fn errors() {
    let a = moe();
    let mut i = Intent::all_on(Tier::Disk, 4096);
    assert_eq!(plan(&a, &i, 1), Err(PlanError::LayersOnDisk));
    i = Intent::all_on(Tier::Gpu(2), 4096);
    assert_eq!(
        plan(&a, &i, 2),
        Err(PlanError::NoSuchGpu { index: 2, gpus: 2 })
    );
    i = Intent::all_on(Tier::Gpu(0), 4096);
    i.layers = vec![Share::new(Tier::Gpu(0), Amount::Fraction(0.5))];
    assert!(matches!(plan(&a, &i, 1), Err(PlanError::Uncovered { .. })));
    i = experts_split(0.7, 0.7);
    assert!(matches!(
        plan(&a, &i, 2),
        Err(PlanError::OverCommitted { .. })
    ));
    i = experts_split(0.1, 0.1);
    i.subsystems
        .get_mut(&Subsystem::RoutedExperts)
        .unwrap()
        .push(Share::all(Tier::Ram));
    assert!(matches!(
        plan(&a, &i, 2),
        Err(PlanError::MoreThanOneRest(_))
    ));
    i = Intent::all_on(Tier::Gpu(0), 4096);
    i.subsystems
        .insert(Subsystem::Embeddings, vec![Share::all(Tier::Ram)]);
    assert_eq!(
        plan(&a, &i, 1),
        Err(PlanError::NotPerLayer(Subsystem::Embeddings))
    );
    i = Intent::all_on(Tier::Gpu(0), 4096);
    i.layers = vec![Share::new(Tier::Gpu(0), Amount::Fraction(1.5))];
    assert_eq!(plan(&a, &i, 1), Err(PlanError::BadFraction(1.5)));
}

#[test]
fn fit_check() {
    let a = moe();
    let m = test_box();
    let o = Overheads::default();

    // all on one 32 GB GPU: 17.6 GiB of weights + KV fits
    let p = plan(&a, &Intent::all_on(Tier::Gpu(0), 4096), 2).unwrap();
    let r = fit::check(&a, &p, &m, &o);
    assert!(r.fits(), "{r}");
    let g0 = &r.devices[0];
    assert_eq!(g0.need.kv_cache, 384 * MIB); // 48 layers x 4 KV heads x 256 x 2 B x 4096
    assert_eq!(r.disk_bytes, a.subsystem_bytes()[&Subsystem::Embeddings]);

    // a 1 GiB GPU doesn't fit, and the report says so
    let mut small = m.clone();
    small.gpus[0].free = GIB;
    assert!(!fit::check(&a, &p, &small, &o).fits());

    // experts in RAM: the RAM tier carries them, GPU 0 gets offload scratch
    let p = plan(&a, &experts_split(0.1, 0.8), 2).unwrap();
    let r = fit::check(&a, &p, &m, &o);
    assert!(r.fits(), "{r}");
    let ram = r.devices.iter().find(|d| d.tier == Tier::Ram).unwrap();
    assert!(ram.need.weights > 13 * GIB);
    let g0 = r.devices.iter().find(|d| d.tier == Tier::Gpu(0)).unwrap();
    assert_eq!(
        g0.need.overhead,
        o.cuda_context + o.gpu_compute + o.op_offload
    );
    // disk-tier experts don't count against RAM
    assert!(r.disk_bytes > GIB);
}

#[test]
fn expert_cache_only_when_some_layer_is_all_host() {
    let a = moe();
    let mut i = experts_split(0.1, 0.8);
    i.expert_cache_bytes = 4 * GIB;
    let p = plan(&a, &i, 2).unwrap();
    assert!(fit::expert_cache_active(&a, &p));
    let r = fit::check(&a, &p, &test_box(), &Overheads::default());
    let cache: u64 = r.devices.iter().map(|d| d.need.expert_cache).sum();
    assert_eq!(cache, 4 * GIB);
    let split = fit::expert_cache_split(&a, &p);
    assert_eq!(split.iter().map(|(_, b)| b).sum::<u64>(), 4 * GIB);
    let s = llama::compile(&a, &p, 2).unwrap();
    assert_eq!(s.moe_cache_size, 4 * GIB);

    let mut all_gpu = Intent::all_on(Tier::Gpu(0), 4096);
    all_gpu.expert_cache_bytes = 4 * GIB;
    let p = plan(&a, &all_gpu, 2).unwrap();
    assert!(!fit::expert_cache_active(&a, &p));
    assert!(fit::expert_cache_split(&a, &p).is_empty());
}

#[test]
fn repack_choice_sets_both_flags() {
    let a = moe();
    let mut i = experts_split(0.1, 0.8);
    let s = llama::compile(&a, &plan(&a, &i, 2).unwrap(), 2).unwrap();
    assert!(!s.use_extra_bufts && !s.no_host && s.use_mmap && s.op_offload);
    i.allow_repack = true;
    let s = llama::compile(&a, &plan(&a, &i, 2).unwrap(), 2).unwrap();
    assert!(s.use_extra_bufts && s.no_host);
}

#[test]
fn rules_are_anchored_and_disjoint() {
    let a = moe();
    let s = llama::compile(&a, &plan(&a, &experts_split(0.1, 0.8), 2).unwrap(), 2).unwrap();
    let res: Vec<regex::Regex> = s
        .overrides
        .iter()
        .map(|o| regex::Regex::new(&o.pattern).unwrap())
        .collect();
    let mut hits: BTreeMap<&str, usize> = BTreeMap::new();
    for p in &a.parts {
        let n = res.iter().filter(|r| r.is_match(&p.name)).count();
        assert!(n <= 1, "{} matched {n} rules", p.name);
        *hits.entry(&p.name).or_default() += n;
    }
    // anchoring: blk.1 must not match a rule for blk.11
    for o in &s.overrides {
        assert!(
            o.pattern.starts_with('^') && o.pattern.ends_with('$'),
            "{}",
            o.pattern
        );
    }
}

#[test]
#[ignore = "prints an example; run with --ignored --nocapture"]
fn show_example() {
    let a = moe();
    let p = plan(&a, &experts_split(0.1, 0.8), 2).unwrap();
    let s = llama::compile(&a, &p, 2).unwrap();
    println!(
        "n_gpu_layers {} tensor_split {:?}",
        s.n_gpu_layers, s.tensor_split
    );
    for o in &s.overrides {
        println!("{} = {}", o.pattern, o.dest);
    }
    println!("{}", fit::check(&a, &p, &test_box(), &Overheads::default()));
}
