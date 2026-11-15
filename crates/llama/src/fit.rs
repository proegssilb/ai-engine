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
