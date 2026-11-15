//! Compile a [`Plan`] to llama.cpp settings, and predict what llama.cpp will
//! do with them.
//!
//! How llama.cpp places weights (src/llama-model.cpp at the pinned version):
//! - The last `n_gpu_layers` of the layers plus the output layer are
//!   offloaded; earlier layers stay on the CPU. Offloaded layers are split
//!   across GPUs in order by cumulative `tensor_split` ratios, so exact layer
//!   counts as the split give exact boundaries.
//! - Token embeddings always start on the CPU.
//! - Each tensor follows its layer's device unless an override rule matches
//!   first. Rules are tried in order and the first `regex_search` match wins.
//! - A layer's KV cache lives on the layer's device, so a layer's home is where
//!   its attention goes, and rules move whatever lives elsewhere.
//! - On the CPU with mmap on, weights stay in the mapped file (our disk tier).
//!   RAM-tier weights need an explicit rule to the Rust buffer.
//!
//! llama.cpp compiles every rule's regex once per tensor, so rules are grouped:
//! one per destination and tensor kind, alternating over layer numbers.

use crate::{Plan, Tier};
use aie_anatomy::{Anatomy, Subsystem};
use std::collections::BTreeMap;
use std::fmt;

/// A buffer an override can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BufDest {
    /// A GPU's device buffer (`CUDA{i}`).
    Device(u32),
    /// The Rust-implemented pinned RAM buffer.
    Pinned,
    /// llama.cpp's CPU buffer: the mapped file when mmap is on.
    Cpu,
}

impl BufDest {
    fn of(t: Tier) -> BufDest {
        match t {
            Tier::Gpu(i) => BufDest::Device(i),
            Tier::Ram => BufDest::Pinned,
            Tier::Disk => BufDest::Cpu,
        }
    }

    /// The tier a tensor ends up in, assuming mmap on and no repacking.
    pub fn tier(self) -> Tier {
        match self {
            BufDest::Device(i) => Tier::Gpu(i),
            BufDest::Pinned => Tier::Ram,
            BufDest::Cpu => Tier::Disk,
        }
    }
}

impl fmt::Display for BufDest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufDest::Device(i) => write!(f, "CUDA{i}"),
            BufDest::Pinned => f.write_str("pinned"),
            BufDest::Cpu => f.write_str("CPU"),
        }
    }
}

/// One `tensor_buft_overrides` entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Override {
    pub pattern: String,
    pub dest: BufDest,
}

/// The llama.cpp model and context parameters a plan sets.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub n_gpu_layers: i32,
    /// One entry per GPU; all zero when nothing is offloaded.
    pub tensor_split: Vec<f32>,
    pub overrides: Vec<Override>,
    pub n_ctx: u64,
    pub moe_cache_size: u64,
    pub use_mmap: bool,
    pub op_offload: bool,
    /// Set together: repacking only happens with both on (decision D4).
    pub use_extra_bufts: bool,
    pub no_host: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CompileError {
    /// Layer homes must be RAM first, then one contiguous block per GPU in
    /// index order: the only layouts llama.cpp's layer split can express.
    UnsupportedLayout(String),
    /// The plan has no tier for a tensor in the anatomy.
    MissingTensor(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::UnsupportedLayout(m) => {
                write!(f, "layer layout llama.cpp can't express: {m}")
            }
            CompileError::MissingTensor(t) => write!(f, "plan has no placement for tensor {t}"),
        }
    }
}

impl std::error::Error for CompileError {}

/// Turn a plan into llama.cpp settings for a machine with `gpus` GPUs.
pub fn compile(a: &Anatomy, plan: &Plan, gpus: u32) -> Result<Settings, CompileError> {
    let counts = layer_counts(&plan.layer_home, gpus)?;
    let offloaded: u32 = counts.iter().sum();
    let (n_gpu_layers, tensor_split) = if offloaded == 0 {
        (0, vec![0.0; gpus as usize])
    } else {
        // Offload the output layer too; it lands on the last GPU with layers.
        let mut split: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
        let last = counts.iter().rposition(|&c| c > 0).unwrap();
        split[last] += 1.0;
        (offloaded as i32 + 1, split)
    };

    let defaults = default_devices(a, n_gpu_layers, &tensor_split);
    // (dest, tensor kind) -> layers (None for non-layer tensors)
    let mut groups: BTreeMap<(BufDest, String), Vec<Option<u32>>> = BTreeMap::new();
    for p in &a.parts {
        let want = BufDest::of(
            *plan
                .tensors
                .get(&p.name)
                .ok_or_else(|| CompileError::MissingTensor(p.name.clone()))?,
        );
        if defaults[&p.name] == want {
            continue;
        }
        let key = match p.layer {
            Some(l) => p
                .name
                .strip_prefix(&format!("blk.{l}."))
                .unwrap_or(&p.name)
                .to_string(),
            None => p.name.clone(),
        };
        groups.entry((want, key)).or_default().push(p.layer);
    }

    let overrides = groups
        .into_iter()
        .map(|((dest, key), layers)| {
            let k = escape(&key);
            let pattern = if layers[0].is_some() {
                let ls: Vec<String> = layers.iter().map(|l| l.unwrap().to_string()).collect();
                format!("^blk\\.(?:{})\\.{k}$", ls.join("|"))
            } else {
                format!("^{k}$")
            };
            Override { pattern, dest }
        })
        .collect();

    Ok(Settings {
        n_gpu_layers,
        tensor_split,
        overrides,
        n_ctx: plan.n_ctx,
        moe_cache_size: plan.expert_cache_bytes,
        use_mmap: true,
        op_offload: true,
        use_extra_bufts: plan.allow_repack,
        no_host: plan.allow_repack,
    })
}

/// GPU layer counts by index, checking the homes are laid out as llama.cpp
/// requires: RAM block, then GPU blocks in ascending index order.
fn layer_counts(homes: &[Tier], gpus: u32) -> Result<Vec<u32>, CompileError> {
    let mut counts = vec![0u32; gpus as usize];
    let rank = |t: &Tier| match t {
        Tier::Ram => Ok(-1i64),
        Tier::Gpu(i) if *i < gpus => Ok(i64::from(*i)),
        t => Err(CompileError::UnsupportedLayout(format!(
            "a layer is homed on {t}"
        ))),
    };
    let mut prev = -1i64;
    for (l, h) in homes.iter().enumerate() {
        let r = rank(h)?;
        if r < prev {
            return Err(CompileError::UnsupportedLayout(format!(
                "layer {l} on {h} comes after a later device"
            )));
        }
        prev = r;
        if let Tier::Gpu(i) = h {
            counts[*i as usize] += 1;
        }
    }
    Ok(counts)
}

fn escape(name: &str) -> String {
    // GGUF tensor names are [a-z0-9_.]; only '.' needs escaping, but be safe.
    regex::escape(name)
}

/// Device of each layer index 0..=n_layers (the last is the output layer), as
/// llama.cpp computes it. `None` = CPU.
fn layer_devices(n_layers: u32, n_gpu_layers: i32, tensor_split: &[f32]) -> Vec<Option<u32>> {
    let n_all = n_layers as i32;
    let n_dev = tensor_split.len();
    let mut splits = tensor_split.to_vec();
    if splits.iter().all(|&x| x == 0.0) {
        // llama.cpp would split by free memory; callers of resolve always pass a split
        splits = vec![1.0; n_dev];
    }
    let mut sum = 0.0f32;
    for s in splits.iter_mut() {
        sum += *s;
        *s = sum;
    }
    for (i, s) in splits.iter_mut().enumerate() {
        *s = if sum != 0.0 {
            *s / sum
        } else {
            (i + 1) as f32 / n_dev as f32
        };
    }
    let start = (n_all + 1 - n_gpu_layers).max(0);
    let act = if n_dev == 0 {
        0
    } else {
        n_gpu_layers.min(n_all + 1)
    };
    (0..=n_all)
        .map(|il| {
            if il < start || il - start >= act {
                None
            } else {
                let x = (il - start) as f32 / act as f32;
                Some(splits.iter().position(|&s| s > x).unwrap_or(n_dev) as u32)
            }
        })
        .collect()
}

fn default_devices(
    a: &Anatomy,
    n_gpu_layers: i32,
    tensor_split: &[f32],
) -> BTreeMap<String, BufDest> {
    let devs = layer_devices(a.layers, n_gpu_layers, tensor_split);
    let to = |d: Option<u32>| d.map(BufDest::Device).unwrap_or(BufDest::Cpu);
    a.parts
        .iter()
        .map(|p| {
            let d = match (p.layer, p.subsystem) {
                (Some(l), _) => to(devs[l as usize]),
                (None, Subsystem::OutputHead) => to(devs[a.layers as usize]),
                // input layer and anything else outside the layers: CPU
                (None, _) => BufDest::Cpu,
            };
            (p.name.clone(), d)
        })
        .collect()
}

/// Predict where llama.cpp puts each tensor under `s`, assuming mmap on and no
/// repacking (as compiled unless the plan allows repacking).
pub fn resolve(a: &Anatomy, s: &Settings) -> Result<BTreeMap<String, Tier>, regex::Error> {
    let rules: Vec<(regex::Regex, BufDest)> = s
        .overrides
        .iter()
        .map(|o| Ok((regex::Regex::new(&o.pattern)?, o.dest)))
        .collect::<Result<_, regex::Error>>()?;
    let defaults = default_devices(a, s.n_gpu_layers, &s.tensor_split);
    Ok(a.parts
        .iter()
        .map(|p| {
            let d = rules
                .iter()
                .find(|(re, _)| re.is_match(&p.name))
                .map(|(_, d)| *d)
                .unwrap_or(defaults[&p.name]);
            (p.name.clone(), d.tier())
        })
        .collect())
}

/// Each layer's device under `s` (`Tier::Ram` for the CPU, where the KV cache
/// then lives in host memory).
pub fn resolve_layer_homes(a: &Anatomy, s: &Settings) -> Vec<Tier> {
    layer_devices(a.layers, s.n_gpu_layers, &s.tensor_split)
        .into_iter()
        .take(a.layers as usize)
        .map(|d| d.map(Tier::Gpu).unwrap_or(Tier::Ram))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_split_matches_llama_cpp_boundaries() {
        // 48 layers: 8 on CPU, 20 on GPU0, 20 + output on GPU1
        let d = layer_devices(48, 41, &[20.0, 21.0]);
        assert_eq!(d.iter().filter(|x| x.is_none()).count(), 8);
        assert_eq!(d.iter().filter(|x| **x == Some(0)).count(), 20);
        assert_eq!(d.iter().filter(|x| **x == Some(1)).count(), 21);
        assert_eq!(d[7], None);
        assert_eq!(d[8], Some(0));
        assert_eq!(d[27], Some(0));
        assert_eq!(d[28], Some(1));
        assert_eq!(d[48], Some(1)); // output layer
    }

    #[test]
    fn exact_boundaries_for_every_split_of_48() {
        // float arithmetic must never move a boundary by one
        for a in 0..=48u32 {
            let b = 48 - a;
            let mut split = vec![a as f32, b as f32];
            let last = if b > 0 { 1 } else { 0 };
            split[last] += 1.0;
            let d = layer_devices(48, 49, &split);
            assert_eq!(
                d.iter().take(48).filter(|x| **x == Some(0)).count() as u32,
                a,
                "split {a}/{b}"
            );
        }
    }

    #[test]
    fn layout_rules() {
        use Tier::*;
        assert_eq!(
            layer_counts(&[Ram, Gpu(0), Gpu(0), Gpu(1)], 2),
            Ok(vec![2, 1])
        );
        assert!(layer_counts(&[Gpu(0), Ram], 2).is_err());
        assert!(layer_counts(&[Gpu(1), Gpu(0)], 2).is_err());
        assert!(layer_counts(&[Disk], 2).is_err());
        assert!(layer_counts(&[Gpu(2)], 2).is_err());
    }
}
