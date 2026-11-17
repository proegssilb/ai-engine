//! Where the weights really are.
//!
//! llama.cpp's log misreports placement (decision D4), so this doesn't read
//! it. It evaluates a short batch and one token in a separate context with a
//! scheduler eval callback, and for every weight a node reads records the
//! tensor's buffer and data address:
//! - inside the model file's mapping: the disk tier;
//! - inside a Rust pinned allocation: the RAM tier;
//! - a GPU's device buffer: that GPU;
//! - anything else (CUDA_Host, CPU_REPACK, plain CPU memory) by buffer name.
//!
//! The scheduler sometimes hands a node a copy of a host weight instead of
//! the weight (op offload, `BACKEND#weight#N`); those are recorded as run-time
//! copies, and a weight only ever seen through copies is attributed from the
//! pinned buffer's records, or as "host, via copy". The MoE cache's stand-in
//! tensors are counted as substitutions.
//!
//! Lessons from the spike's review (Fable): count cache stand-ins once, not
//! their backing tensors too; attribute copies to the weight; and don't treat
//! a weight that's only read through a copy as missing.

use crate::context::{Context, ContextParams};
use crate::model::{Model, gpu_device};
use crate::pinned::REGISTRY;
use crate::{Error, Token, sys};
use aie_plan::{Plan, Tier};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ffi::{CStr, c_void};
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

/// Where a weight was found.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Where {
    /// A tier the planner knows: a GPU, the pinned RAM buffer, the mapped file.
    Tier(Tier),
    /// A buffer outside the planner's tiers, by name (`CUDA_Host`, `CPU_REPACK`, `CPU`).
    Buffer(String),
    /// Host memory, only ever read through a run-time copy, not in the pinned buffer.
    HostViaCopy,
}

impl fmt::Display for Where {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Where::Tier(t) => write!(f, "{t}"),
            Where::Buffer(b) => write!(f, "{b} buffer"),
            Where::HostViaCopy => f.write_str("host memory (seen only via copies)"),
        }
    }
}

/// What the check found.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Report {
    /// Every weight the graph read, directly or through a copy.
    pub weights: BTreeMap<String, Where>,
    /// Weights the scheduler copied at run time, and to which backends.
    pub copied: BTreeMap<String, BTreeSet<String>>,
    /// MoE-cache stand-ins read instead of the placed experts, by buffer name.
    pub cache_reads: BTreeMap<String, usize>,
    /// Model tensors the graph never read (can't be verified this way).
    pub unseen: Vec<String>,
}

/// A weight not where the plan put it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Mismatch {
    pub tensor: String,
    pub planned: Tier,
    pub actual: Where,
}

impl Report {
    /// Weights that aren't where the plan says. Unseen tensors aren't counted.
    pub fn compare(&self, plan: &Plan) -> Vec<Mismatch> {
        self.weights
            .iter()
            .filter_map(|(name, actual)| {
                let planned = *plan.tensors.get(name)?;
                (actual != &Where::Tier(planned)).then(|| Mismatch {
                    tensor: name.clone(),
                    planned,
                    actual: actual.clone(),
                })
            })
            .collect()
    }

    /// Bytes per actual location, given tensor sizes.
    pub fn bytes_by_location(&self, sizes: &BTreeMap<String, u64>) -> BTreeMap<Where, u64> {
        let mut m = BTreeMap::new();
        for (n, w) in &self.weights {
            *m.entry(w.clone()).or_insert(0) += sizes.get(n).copied().unwrap_or(0);
        }
        m
    }
}

#[derive(Clone)]
struct Seen {
    buffer: String,
    data: usize,
}

struct State {
    names: HashSet<String>,
    seen: BTreeMap<String, Seen>,
    copied: BTreeMap<String, BTreeSet<String>>,
    cache: BTreeMap<String, BTreeSet<String>>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// One check at a time: the callback state is process-global.
static RUNNING: Mutex<()> = Mutex::new(());

unsafe fn name_of(t: *const sys::ggml_tensor) -> String {
    // SAFETY: Callers pass a non-null tensor from ggml; `name` is a NUL-terminated array inside
    // it.
    unsafe { CStr::from_ptr((*t).name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

unsafe fn observe_src(s: &mut State, mut t: *const sys::ggml_tensor) {
    // Walk views back towards storage, stopping at the first thing we recognise.
    while !t.is_null() {
        // SAFETY: `t` is non-null (loop condition) and points at a tensor in the graph ggml is
        // evaluating, alive for the callback.
        let name = unsafe { name_of(t) };
        // SAFETY: As above.
        let buf = unsafe { (*t).buffer };
        if s.names.contains(&name) {
            if !buf.is_null() {
                // SAFETY: `buf` is non-null (checked); a buffer's name is a NUL-terminated
                // string owned by its buffer type, alive for the callback.
                let buffer = unsafe { CStr::from_ptr(sys::ggml_backend_buffer_name(buf)) }
                    .to_string_lossy()
                    .into_owned();
                s.seen.entry(name).or_insert(Seen {
                    buffer,
                    // SAFETY: As above: `t` is a live, non-null tensor.
                    data: unsafe { (*t).data } as usize,
                });
            }
            return;
        }
        let mut parts = name.splitn(3, '#');
        if let (Some(backend), Some(inner), Some(_)) = (parts.next(), parts.next(), parts.next()) {
            if s.names.contains(inner) {
                s.copied
                    .entry(inner.to_string())
                    .or_default()
                    .insert(backend.to_string());
            }
            return;
        }
        if name.starts_with("moe_cache") && !buf.is_null() {
            // SAFETY: As above: `buf` is non-null and live.
            let buffer = unsafe { CStr::from_ptr(sys::ggml_backend_buffer_name(buf)) }
                .to_string_lossy()
                .into_owned();
            s.cache.entry(buffer).or_default().insert(name);
            return;
        }
        // SAFETY: As above: `t` is non-null; `view_src` is null or another live tensor of the
        // same graph.
        t = unsafe { (*t).view_src };
    }
}

unsafe extern "C" fn eval_cb(t: *mut sys::ggml_tensor, ask: bool, _: *mut c_void) -> bool {
    if !ask {
        return true;
    }
    let mut g = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(s) = g.as_mut() {
        // SAFETY: ggml passes a live, non-null tensor; `src` is its fixed-size array of source
        // pointers, each null or a live tensor.
        for src in unsafe { (*t).src.iter() } {
            // SAFETY: Each source is null (handled in observe_src) or a live tensor of the
            // graph being evaluated.
            unsafe { observe_src(s, *src) };
        }
    }
    false
}

/// Address ranges where the model file is mapped in this process. Kept as
/// separate ranges: two mappings of the same file (two loads) can have other
/// memory, such as pinned buffers, between them.
fn file_mappings(path: &std::path::Path) -> Vec<(usize, usize)> {
    let Some(canon) = std::fs::canonicalize(path)
        .ok()
        .and_then(|c| c.to_str().map(String::from))
    else {
        return vec![];
    };
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    maps.lines()
        .filter(|l| l.ends_with(canon.as_str()))
        .filter_map(|line| {
            let (s, e) = line.split_whitespace().next()?.split_once('-')?;
            Some((
                usize::from_str_radix(s, 16).ok()?,
                usize::from_str_radix(e, 16).ok()?,
            ))
        })
        .collect()
}

/// Run the check: one batch of up to `n_batch` prompt tokens, then one
/// token. `names` are the model's tensor names (from its anatomy).
pub fn check(
    model: &Arc<Model>,
    params: &ContextParams,
    prompt: &[Token],
    names: impl IntoIterator<Item = String>,
) -> Result<Report, Error> {
    let _one_at_a_time = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
    let names: HashSet<String> = names.into_iter().collect();
    *STATE.lock().unwrap() = Some(State {
        names: names.clone(),
        seen: BTreeMap::new(),
        copied: BTreeMap::new(),
        cache: BTreeMap::new(),
    });

    let p = ContextParams {
        n_ctx: params.n_batch + 64,
        ..params.clone()
    };
    let run = (|| {
        let mut ctx = Context::with_eval_callback(model.clone(), p.clone(), Some(eval_cb))?;
        let batch = &prompt[..prompt.len().min(p.n_batch as usize).max(1)];
        ctx.decode(batch)?;
        ctx.decode(&[batch[0]])?;
        Ok::<(), Error>(())
    })();
    let state = STATE.lock().unwrap().take().expect("verify state");
    run?;

    let mappings = file_mappings(model.path());
    let pinned: Vec<(usize, usize, BTreeSet<String>)> = REGISTRY
        .lock()
        .unwrap()
        .iter()
        .filter(|b| !b.freed)
        .map(|b| {
            (
                b.base,
                b.len,
                b.tensors.iter().map(|(n, _, _)| n.clone()).collect(),
            )
        })
        .collect();
    let gpu_names: Vec<String> = (0..)
        .map_while(gpu_device)
        .map(|d| {
            // SAFETY: `d` is a registered GPU device from `gpu_device`; its name is a NUL-
            // terminated string owned by the device, alive for the process, copied here.
            unsafe { CStr::from_ptr(sys::ggml_backend_dev_name(d)) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    let locate = |seen: &Seen| -> Where {
        if mappings.iter().any(|(s, e)| (*s..*e).contains(&seen.data)) {
            return Where::Tier(Tier::Disk);
        }
        if pinned
            .iter()
            .any(|(b, l, _)| (*b..b + l).contains(&seen.data))
        {
            return Where::Tier(Tier::Ram);
        }
        if let Some(i) = gpu_names.iter().position(|g| *g == seen.buffer) {
            return Where::Tier(Tier::Gpu(i as u32));
        }
        Where::Buffer(seen.buffer.clone())
    };

    let mut report = Report::default();
    for name in &names {
        let w = if let Some(seen) = state.seen.get(name) {
            locate(seen)
        } else if state.copied.contains_key(name) {
            if pinned.iter().any(|(_, _, ns)| ns.contains(name)) {
                Where::Tier(Tier::Ram)
            } else {
                Where::HostViaCopy
            }
        } else {
            report.unseen.push(name.clone());
            continue;
        };
        report.weights.insert(name.clone(), w);
    }
    report.unseen.sort();
    report.copied = state.copied;
    report.cache_reads = state.cache.into_iter().map(|(b, s)| (b, s.len())).collect();
    Ok(report)
}
