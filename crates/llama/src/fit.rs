//! Automatic placement: llama.cpp's own fit (`common_fit_params`, the code
//! behind `--fit`), through the C++ shim. Its result is ordinary llama.cpp
//! settings, which `aie_plan::llama::resolve` turns into a plan to show and
//! verify like any other (features.md: auto-place is llama.cpp's choice).

use crate::model::gpu_device;
use crate::{Error, init, sys};
use aie_plan::llama::{BufDest, Override, Settings};
use std::ffi::{CStr, CString};
use std::path::Path;

unsafe extern "C" {
    fn aie_fit(
        model_path: *const libc::c_char,
        mparams: *mut sys::llama_model_params,
        cparams: *mut sys::llama_context_params,
        tensor_split: *mut f32,
        overrides: *mut sys::llama_model_tensor_buft_override,
        margins: *mut usize,
        n_ctx_min: u32,
    ) -> i32;
    fn aie_measure(
        model_path: *const libc::c_char,
        mparams: *const sys::llama_model_params,
        cparams: *const sys::llama_context_params,
        rows: *mut MemRow,
        max_rows: i32,
    ) -> i32;
    fn aie_ctx_breakdown(ctx: *const sys::llama_context, rows: *mut MemRow, max_rows: i32) -> i32;
}

/// llama.cpp's fit and dry run swap the global logger and restore whatever
/// they found; two at once restore each other's temporary logger, which
/// points into a finished stack frame. One at a time per process.
static FIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[repr(C)]
#[derive(Clone, Copy)]
struct MemRow {
    name: [libc::c_char; 128],
    kind: i32,
    total: u64,
    free: u64,
    model: u64,
    context: u64,
    compute: u64,
}

/// One row of llama.cpp's dry-run memory breakdown.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Measured {
    /// Device name (`CUDA0`), `Host`, or a buffer type name.
    pub name: String,
    /// "device", "host" or "buffer_type".
    pub kind: String,
    /// Device memory (0 unless a device).
    pub total: u64,
    pub free: u64,
    /// Weights, KV cache/state, and compute scratch llama.cpp would allocate here.
    pub model: u64,
    pub context: u64,
    pub compute: u64,
}

impl Measured {
    pub fn need(&self) -> u64 {
        self.model + self.context + self.compute
    }
}

/// llama.cpp's dry run (`no_alloc`): what each device would hold with these
/// load options and context, without allocating anything. The authoritative
/// check before applying a plan (architecture.md: memory estimation).
pub fn measure(
    path: &Path,
    load: &crate::LoadOptions,
    ctx: &crate::ContextParams,
) -> Result<Vec<Measured>, Error> {
    let c_path = crate::model::c_path(path).map_err(|e| Error::Fit(e.to_string()))?;
    let params = crate::model::ModelParams::new(load)?;
    let cp = ctx.raw();
    let mut rows = vec![
        MemRow {
            name: [0; 128],
            kind: 0,
            total: 0,
            free: 0,
            model: 0,
            context: 0,
            compute: 0
        };
        64
    ];
    let _one = FIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = unsafe {
        aie_measure(
            c_path.as_ptr(),
            &params.mp,
            &cp,
            rows.as_mut_ptr(),
            rows.len() as i32,
        )
    };
    if n < 0 {
        return Err(Error::Fit(format!(
            "llama.cpp's dry run failed: {}",
            crate::log::tail(6)
        )));
    }
    Ok(convert(&rows[..n as usize]))
}

fn convert(rows: &[MemRow]) -> Vec<Measured> {
    rows.iter()
        .map(|r| Measured {
            name: unsafe { CStr::from_ptr(r.name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            kind: match r.kind {
                0 => "device",
                1 => "host",
                _ => "buffer_type",
            }
            .into(),
            total: r.total,
            free: r.free,
            model: r.model,
            context: r.context,
            compute: r.compute,
        })
        .collect()
}

/// What a live context really allocated, per device and on the host: the
/// same rows as [`measure`], but read from the allocated buffers.
pub fn breakdown(ctx: *const sys::llama_context) -> Vec<Measured> {
    let mut rows = vec![
        MemRow {
            name: [0; 128],
            kind: 0,
            total: 0,
            free: 0,
            model: 0,
            context: 0,
            compute: 0
        };
        64
    ];
    let n = unsafe { aie_ctx_breakdown(ctx, rows.as_mut_ptr(), rows.len() as i32) };
    if n < 0 {
        return Vec::new();
    }
    convert(&rows[..n as usize])
}

/// What llama.cpp's fit chose.
#[derive(Clone, Debug, PartialEq)]
pub struct Fitted {
    pub settings: Settings,
}

/// Fit a model to the memory free right now on each GPU, leaving `margin`
/// bytes per device. `n_ctx` 0 lets the fit pick the context (down to
/// `n_ctx_min`). Host memory is treated as unlimited, as llama.cpp does.
pub fn fit(path: &Path, n_ctx: u32, n_ctx_min: u32, margin: u64) -> Result<Fitted, Error> {
    init();
    let c_path = CString::new(
        path.to_str()
            .ok_or_else(|| Error::Fit("path is not UTF-8".into()))?,
    )
    .map_err(|_| Error::Fit("path contains NUL".into()))?;
    let max_dev = unsafe { sys::llama_max_devices() };
    let max_tbo = unsafe { sys::llama_max_tensor_buft_overrides() };
    let mut split = vec![0f32; max_dev];
    let mut tbo = vec![
        sys::llama_model_tensor_buft_override {
            pattern: std::ptr::null(),
            buft: std::ptr::null_mut()
        };
        max_tbo
    ];
    let mut margins = vec![margin as usize; max_dev];
    let mut mp = unsafe { sys::llama_model_default_params() };
    let mut cp = unsafe { sys::llama_context_default_params() };
    cp.n_ctx = n_ctx;

    let _one = FIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let status = unsafe {
        aie_fit(
            c_path.as_ptr(),
            &mut mp,
            &mut cp,
            split.as_mut_ptr(),
            tbo.as_mut_ptr(),
            margins.as_mut_ptr(),
            n_ctx_min,
        )
    };
    match status {
        0 => {}
        1 => {
            return Err(Error::Fit(
                "llama.cpp found no placement that fits the free GPU memory".into(),
            ));
        }
        _ => {
            return Err(Error::Fit(format!(
                "llama.cpp's fit failed: {}",
                crate::log::tail(6)
            )));
        }
    }

    let gpus: Vec<sys::ggml_backend_buffer_type_t> = (0..)
        .map_while(gpu_device)
        .map(|d| unsafe { sys::ggml_backend_dev_buffer_type(d) })
        .collect();
    let cpu = unsafe { sys::ggml_backend_cpu_buffer_type() };
    let mut overrides = Vec::new();
    let tbo_ptr = if mp.tensor_buft_overrides.is_null() {
        tbo.as_ptr()
    } else {
        mp.tensor_buft_overrides
    };
    for i in 0..max_tbo {
        let o = unsafe { *tbo_ptr.add(i) };
        if o.pattern.is_null() {
            break;
        }
        let dest = if o.buft == cpu {
            BufDest::Cpu
        } else if let Some(g) = gpus.iter().position(|b| *b == o.buft) {
            BufDest::Device(g as u32)
        } else {
            let name = unsafe { CStr::from_ptr(sys::ggml_backend_buft_name(o.buft)) }
                .to_string_lossy()
                .into_owned();
            return Err(Error::Fit(format!(
                "fit chose buffer type {name}, which placement can't express"
            )));
        };
        overrides.push(Override {
            pattern: unsafe { CStr::from_ptr(o.pattern) }
                .to_string_lossy()
                .into_owned(),
            dest,
        });
    }

    let split_src: &[f32] = if mp.tensor_split.is_null() {
        &split
    } else {
        unsafe { std::slice::from_raw_parts(mp.tensor_split, max_dev) }
    };
    let n_gpus = gpus.len();
    Ok(Fitted {
        settings: Settings {
            n_gpu_layers: mp.n_gpu_layers,
            tensor_split: split_src[..n_gpus].to_vec(),
            overrides,
            n_ctx: u64::from(cp.n_ctx),
            moe_cache_size: 0,
            use_mmap: true,
            op_offload: true,
            use_extra_bufts: false,
            no_host: false,
        },
    })
}
