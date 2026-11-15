//! Ground truth for "do the weights live where the rules said?".
//!
//! The placement table in main.rs is reconstructed from llama.cpp's debug log,
//! and that log is not reliable: for a `CPU` override, llama.cpp logs the
//! buffer type it picked first (e.g. `CUDA_Host`) and then silently swaps it
//! for the mmap'd CPU buffer. So this module doesn't read the log at all.
//! It evaluates the graph once with a scheduler eval callback and, for every
//! weight a node actually reads, records the tensor's real buffer and data
//! address. An address inside the GGUF mapping means the mmap'd file; inside
//! a RustPinned allocation means our buffer; otherwise the buffer's own name
//! (`CUDA0`, `CUDA_Host`, `CPU_REPACK`, ...).
//!
//! It also records what the scheduler copies at run time (`CUDA0#blk.0.ffn_up_exps.weight#0`)
//! and what the MoE cache substitutes, since a weight that lives in RAM but is
//! shipped to a GPU every batch is a placement fact too.

use crate::sys;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::ffi::{CStr, c_void};
use std::sync::Mutex;

#[derive(Clone)]
pub struct Seen {
    pub buffer: String,
    pub data: usize,
    pub is_host: bool,
}

#[derive(Default)]
pub struct Observed {
    pub weights: BTreeMap<String, Seen>,
    /// phase -> weight name -> backends it was copied to during that phase
    pub copies: BTreeMap<&'static str, BTreeMap<String, BTreeSet<String>>>,
    /// tensors read in place of a weight that aren't a scheduler copy (e.g. MoE cache):
    /// phase -> buffer name -> (tensor names, distinct backing buffers as (ptr, size))
    pub substitutes: BTreeMap<&'static str, BTreeMap<String, (BTreeSet<String>, BTreeMap<usize, usize>)>>,
}

struct State {
    names: HashSet<String>,
    phase: &'static str,
    obs: Observed,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

pub fn begin(names: impl IntoIterator<Item = String>) {
    *STATE.lock().unwrap() = Some(State { names: names.into_iter().collect(), phase: "prefill", obs: Observed::default() });
}

pub fn set_phase(phase: &'static str) {
    if let Some(s) = STATE.lock().unwrap().as_mut() {
        s.phase = phase;
    }
}

pub fn finish() -> Observed {
    STATE.lock().unwrap().take().map(|s| s.obs).unwrap_or_default()
}

unsafe fn name_of(t: *const sys::ggml_tensor) -> String {
    CStr::from_ptr((*t).name.as_ptr()).to_string_lossy().into_owned()
}

unsafe fn record(s: &mut State, t: *const sys::ggml_tensor) {
    if t.is_null() || (*t).buffer.is_null() {
        return;
    }
    let name = name_of(t);
    let buf = (*t).buffer;
    let buffer = CStr::from_ptr(sys::ggml_backend_buffer_name(buf)).to_string_lossy().into_owned();
    if s.names.contains(&name) {
        s.obs.weights.entry(name).or_insert(Seen { buffer, data: (*t).data as usize, is_host: sys::ggml_backend_buffer_is_host(buf) });
        return;
    }
    // Scheduler copy: "<backend>#<tensor>#<n>"
    let mut parts = name.splitn(3, '#');
    if let (Some(backend), Some(inner), Some(_)) = (parts.next(), parts.next(), parts.next()) {
        if s.names.contains(inner) {
            s.obs.copies.entry(s.phase).or_default().entry(inner.to_string()).or_default().insert(backend.to_string());
        }
        return;
    }
    // Anything else that looks like expert weights but isn't the original tensor (MoE cache slots).
    if name.contains("exps") && sys::ggml_backend_buffer_get_usage(buf) != sys::ggml_backend_buffer_usage_GGML_BACKEND_BUFFER_USAGE_COMPUTE {
        // The cache's stand-ins have the full [n_embd, n_ff, n_expert] shape, so their nbytes
        // overstate what's cached; the backing buffer's size is the real footprint.
        let e = s.obs.substitutes.entry(s.phase).or_default().entry(buffer).or_default();
        e.0.insert(name);
        e.1.insert(buf as usize, sys::ggml_backend_buffer_get_size(buf));
    }
}

/// ggml_backend_sched_eval_callback: observe every source of every node, never ask for data.
pub unsafe extern "C" fn eval_cb(t: *mut sys::ggml_tensor, ask: bool, _: *mut c_void) -> bool {
    if !ask {
        return true;
    }
    let mut g = STATE.lock().unwrap();
    let Some(s) = g.as_mut() else { return false };
    for src in (*t).src.iter() {
        let mut p = *src as *const sys::ggml_tensor;
        // walk views back to the tensor that owns the memory
        while !p.is_null() {
            record(s, p);
            p = (*p).view_src;
        }
    }
    false
}

/// Normalise an observed tensor to the same vocabulary as main.rs's placement table.
pub fn actual_home(seen: &Seen, mapping: Option<(usize, usize, usize)>, pinned: &[(usize, usize)]) -> String {
    if let Some((_, s, e)) = mapping
        && seen.data >= s
        && seen.data < e
    {
        return "CPU (mmap'd file)".into();
    }
    if pinned.iter().any(|(b, l)| seen.data >= *b && seen.data < b + l) {
        return "RustPinned RAM".into();
    }
    if seen.is_host && seen.buffer == "CPU_Mapped" {
        // mapped but not our file mapping: shouldn't happen, say so
        return "CPU_Mapped (unknown mapping)".into();
    }
    seen.buffer.clone()
}
