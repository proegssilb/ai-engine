//! Placement planning, from what users say to what llama.cpp needs.
//!
//! 1. [`Intent`]: per subsystem, a share of it per tier ("experts: 10% GPU 1,
//!    80% RAM, rest on disk"), as in decision D6.
//! 2. [`plan()`] turns shares into a concrete [`Plan`]: a home device for each
//!    layer (where its attention and KV cache live) and a tier for every
//!    tensor. Shares become whole layers, so they're approximate.
//! 3. [`fit::check`] says whether the plan fits each device, and why not.
//! 4. [`llama::compile`] turns the plan into llama.cpp settings: layer split,
//!    and grouped override rules for whatever doesn't follow its layer.
//!    [`llama::resolve`] predicts what llama.cpp will do with those settings,
//!    so the compiler can be checked without loading a model.

pub mod fit;
pub mod llama;
mod planner;

pub use planner::{PlanError, plan};

use aie_anatomy::Subsystem;
use std::collections::BTreeMap;
use std::fmt;

/// Where bytes can live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// A GPU, by index (`CUDA0` = 0).
    Gpu(u32),
    /// Host RAM we own: the Rust buffer (pinned, mlocked).
    Ram,
    /// The model file, memory-mapped: may be dropped from RAM and re-read.
    Disk,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tier::Gpu(i) => write!(f, "GPU {i}"),
            Tier::Ram => f.write_str("RAM"),
            Tier::Disk => f.write_str("disk"),
        }
    }
}

/// How much of a subsystem goes to a tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Amount {
    /// A share of the subsystem's bytes, 0.0–1.0.
    Fraction(f64),
    /// About this many bytes.
    Bytes(u64),
    /// Whatever the other shares leave.
    Rest,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Share {
    pub tier: Tier,
    pub amount: Amount,
}

impl Share {
    pub fn new(tier: Tier, amount: Amount) -> Self {
        Share { tier, amount }
    }
    pub fn all(tier: Tier) -> Self {
        Share { tier, amount: Amount::Rest }
    }
}

/// What the user asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct Intent {
    /// Context length the KV cache is sized for.
    pub n_ctx: u64,
    /// Where layers live: their attention, norms, router and KV cache, and any
    /// per-layer subsystem without its own shares. Tiers must be GPUs or RAM.
    pub layers: Vec<Share>,
    /// Per-layer subsystems placed independently of their layer, e.g. routed
    /// experts or a dense model's FFN.
    pub subsystems: BTreeMap<Subsystem, Vec<Share>>,
    /// Embeddings' tier; `None` = llama.cpp's choice (the mapped file).
    pub embeddings: Option<Tier>,
    /// Output head's tier; `None` = llama.cpp's choice (with the last layer).
    pub output: Option<Tier>,
    /// GPU memory for llama.cpp's expert cache, split across GPUs like the layers.
    pub expert_cache_bytes: u64,
    /// Let llama.cpp repack CPU weights. Faster CPU kernels, but repacked
    /// weights leave the tier they were placed in (decision D4).
    pub allow_repack: bool,
}

impl Intent {
    /// Everything in one tier, for `n_ctx` tokens of context.
    pub fn all_on(tier: Tier, n_ctx: u64) -> Intent {
        Intent {
            n_ctx,
            layers: vec![Share::all(tier)],
            subsystems: BTreeMap::new(),
            embeddings: None,
            output: None,
            expert_cache_bytes: 0,
            allow_repack: false,
        }
    }
}

/// A concrete placement.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// Home of each layer (index = layer): GPU or RAM.
    pub layer_home: Vec<Tier>,
    /// Tier of every tensor, by name.
    pub tensors: BTreeMap<String, Tier>,
    pub n_ctx: u64,
    pub expert_cache_bytes: u64,
    pub allow_repack: bool,
}

impl Plan {
    /// Bytes per tier, from an anatomy's tensor sizes.
    pub fn bytes_by_tier(&self, a: &aie_anatomy::Anatomy) -> BTreeMap<Tier, u64> {
        let mut m = BTreeMap::new();
        for p in &a.parts {
            if let Some(t) = self.tensors.get(&p.name) {
                *m.entry(*t).or_insert(0) += p.bytes;
            }
        }
        m
    }

    /// Bytes of one subsystem per tier.
    pub fn subsystem_by_tier(&self, a: &aie_anatomy::Anatomy, s: Subsystem) -> BTreeMap<Tier, u64> {
        let mut m = BTreeMap::new();
        for p in a.parts_of(s) {
            if let Some(t) = self.tensors.get(&p.name) {
                *m.entry(*t).or_insert(0) += p.bytes;
            }
        }
        m
    }

    /// Number of layers homed on each tier.
    pub fn layer_counts(&self) -> BTreeMap<Tier, u32> {
        let mut m = BTreeMap::new();
        for t in &self.layer_home {
            *m.entry(*t).or_insert(0) += 1;
        }
        m
    }
}
