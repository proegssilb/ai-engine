//! Loading a model with a placement.

use crate::pinned::{self, PinOptions};
use crate::{Error, Token, init, sys};
use aie_plan::llama::{BufDest, Settings};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Mutex;

/// How to load: llama.cpp settings compiled from a plan, and how the RAM tier
/// (the Rust buffer) behaves.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoadOptions {
    pub settings: Settings,
    pub pin: PinOptions,
}

impl LoadOptions {
    /// Everything on the CPU, weights left in the mapped file.
    pub fn cpu() -> LoadOptions {
        LoadOptions {
            settings: Settings {
                n_gpu_layers: 0,
                tensor_split: vec![],
                overrides: vec![],
                n_ctx: 4096,
                moe_cache_size: 0,
                use_mmap: true,
                op_offload: true,
                use_extra_bufts: false,
                no_host: false,
            },
            pin: PinOptions {
                mlock: true,
                hugepages: true,
                ..PinOptions::default()
            },
        }
    }
}

/// A loaded model. Read-only after loading, so shareable across threads.
pub struct Model {
    ptr: NonNull<sys::llama_model>,
    path: PathBuf,
    /// Override patterns llama.cpp was given; kept alive with the model.
    _patterns: Vec<CString>,
}

// llama_model is immutable after load; llama.cpp shares it across contexts.
// SAFETY: llama.cpp treats a loaded llama_model as immutable and shares it between contexts on
// different threads; this type exposes no way to mutate it.
unsafe impl Send for Model {}
// SAFETY: As for Send: every method used through `&Model` only reads the model (llama.cpp's own
// server calls them concurrently).
unsafe impl Sync for Model {}

/// One pinned buffer type per distinct option set, alive for the process.
fn pinned_buft(opts: &PinOptions) -> sys::ggml_backend_buffer_type_t {
    static CACHE: Mutex<Vec<(PinOptions, usize)>> = Mutex::new(Vec::new());
    let mut c = CACHE.lock().unwrap();
    if let Some((_, p)) = c.iter().find(|(o, _)| o == opts) {
        return *p as sys::ggml_backend_buffer_type_t;
    }
    let b = pinned::new_buffer_type("RustPinned", opts.clone());
    c.push((opts.clone(), b as usize));
    b
}

/// The i-th GPU device, in llama.cpp's order.
pub(crate) fn gpu_device(i: u32) -> Option<sys::ggml_backend_dev_t> {
    // SAFETY: Every index is below ggml_backend_dev_count(), so each `d` is a registered
    // device; the registry lives for the process.
    unsafe {
        (0..sys::ggml_backend_dev_count())
            .map(|k| sys::ggml_backend_dev_get(k))
            .filter(|d| {
                sys::ggml_backend_dev_type(*d)
                    == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
            })
            .nth(i as usize)
    }
}

pub(crate) fn c_path(path: &Path) -> Result<CString, Error> {
    CString::new(
        path.to_str()
            .ok_or_else(|| Error::Load("path is not UTF-8".into()))?,
    )
    .map_err(|_| Error::Load("path contains NUL".into()))
}

/// llama_model_params for a load, with the memory its pointers refer to.
/// Shared by loading and llama.cpp's dry run, so they see the same placement.
pub(crate) struct ModelParams {
    pub mp: sys::llama_model_params,
    pub patterns: Vec<CString>,
    _overrides: Vec<sys::llama_model_tensor_buft_override>,
    _split: Vec<f32>,
}

impl ModelParams {
    pub fn new(opts: &LoadOptions) -> Result<ModelParams, Error> {
        init();
        let s = &opts.settings;
        let mut patterns = Vec::with_capacity(s.overrides.len());
        let mut overrides = Vec::with_capacity(s.overrides.len() + 1);
        for o in &s.overrides {
            let buft = match o.dest {
                // SAFETY: Returns a static buffer type; no preconditions.
                BufDest::Cpu => unsafe { sys::ggml_backend_cpu_buffer_type() },
                BufDest::Pinned => pinned_buft(&opts.pin),
                BufDest::Device(i) => {
                    let d = gpu_device(i).ok_or_else(|| {
                        Error::Device(format!(
                            "placement uses GPU {i}, which this machine doesn't have"
                        ))
                    })?;
                    // SAFETY: `d` is a registered GPU device from `gpu_device`.
                    unsafe { sys::ggml_backend_dev_buffer_type(d) }
                }
            };
            let p = CString::new(o.pattern.as_str())
                .map_err(|_| Error::Load("override pattern contains NUL".into()))?;
            overrides.push(sys::llama_model_tensor_buft_override {
                pattern: p.as_ptr(),
                buft,
            });
            patterns.push(p);
        }
        overrides.push(sys::llama_model_tensor_buft_override {
            pattern: std::ptr::null(),
            buft: std::ptr::null_mut(),
        });

        // llama.cpp reads the split by the real GPU count (see check_device_count).
        crate::check_device_count()?;
        // SAFETY: Returns a constant; no preconditions.
        let max_dev = unsafe { sys::llama_max_devices() };
        if s.tensor_split.len() > max_dev {
            return Err(Error::Load(format!(
                "{} GPUs in the split, llama.cpp supports {max_dev}",
                s.tensor_split.len()
            )));
        }
        let mut split = vec![0f32; max_dev];
        split[..s.tensor_split.len()].copy_from_slice(&s.tensor_split);

        // SAFETY: Returns a struct by value; no pointers or preconditions.
        let mut mp = unsafe { sys::llama_model_default_params() };
        mp.n_gpu_layers = s.n_gpu_layers;
        mp.tensor_split = split.as_ptr();
        if !s.overrides.is_empty() {
            mp.tensor_buft_overrides = overrides.as_ptr();
        }
        mp.use_extra_bufts = s.use_extra_bufts;
        mp.no_host = s.no_host;
        mp.load_mode = if s.use_mmap {
            sys::llama_load_mode_LLAMA_LOAD_MODE_MMAP
        } else {
            sys::llama_load_mode_LLAMA_LOAD_MODE_NONE
        };
        Ok(ModelParams {
            mp,
            patterns,
            _overrides: overrides,
            _split: split,
        })
    }
}

impl Model {
    pub fn load(path: &Path, opts: &LoadOptions) -> Result<Model, Error> {
        init();
        let c_path = c_path(path)?;
        let params = ModelParams::new(opts)?;
        let _log = crate::log_read();
        // SAFETY: `c_path` is NUL-terminated. `params.mp` points at the split array and
        // override list owned by `params`, alive for this call. llama.cpp copies the split into
        // the model, and reads the override list (whose patterns are kept in the Model) only
        // while loading, so it is fine that `params`' vectors are dropped after.
        let ptr = unsafe { sys::llama_model_load_from_file(c_path.as_ptr(), params.mp) };
        let ptr = NonNull::new(ptr).ok_or_else(|| Error::Load(crate::log::tail(8)))?;
        Ok(Model {
            ptr,
            path: path.to_path_buf(),
            _patterns: params.patterns,
        })
    }

    pub(crate) fn raw(&self) -> *mut sys::llama_model {
        self.ptr.as_ptr()
    }

    pub(crate) fn vocab(&self) -> *const sys::llama_vocab {
        // SAFETY: `self.raw()` is the llama_model this value owns (non-null, freed only in
        // Drop).
        unsafe { sys::llama_model_get_vocab(self.raw()) }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn n_layer(&self) -> u32 {
        // SAFETY: `self.raw()` is the llama_model this value owns (non-null, freed only in
        // Drop).
        unsafe { sys::llama_model_n_layer(self.raw()) as u32 }
    }

    pub fn n_vocab(&self) -> usize {
        // SAFETY: `self.vocab()` points into the model this value owns, which outlives the
        // call.
        unsafe { sys::llama_vocab_n_tokens(self.vocab()) as usize }
    }

    /// e.g. `qwen3moe 30B.A3B Q4_K - Medium`
    pub fn description(&self) -> String {
        let mut buf = vec![0u8; 256];
        let cap = buf.len() - 1;
        // SAFETY: `self.raw()` is the llama_model this value owns (non-null, freed only in
        // Drop). `buf` has `buf.len()` writable bytes, and llama_model_desc (snprintf) writes
        // at most that many.
        let n = unsafe { sys::llama_model_desc(self.raw(), buf.as_mut_ptr() as *mut _, buf.len()) };
        // snprintf returns the untruncated length; keep what fit (without the NUL).
        buf.truncate((n.max(0) as usize).min(cap));
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Weight bytes.
    pub fn size(&self) -> u64 {
        // SAFETY: `self.raw()` is the llama_model this value owns (non-null, freed only in
        // Drop).
        unsafe { sys::llama_model_size(self.raw()) }
    }

    /// The model's own chat template (Jinja source), if it has one.
    pub fn chat_template(&self) -> Option<String> {
        // SAFETY: `self.raw()` is the llama_model this value owns (non-null, freed only in
        // Drop). A null template name asks for the default template.
        let p = unsafe { sys::llama_model_chat_template(self.raw(), std::ptr::null()) };
        // SAFETY: `p` is non-null (checked) and points at a NUL-terminated string owned by the
        // model; it is copied immediately.
        (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }

    pub fn is_eog(&self, t: Token) -> bool {
        // SAFETY: `self.vocab()` points into the model this value owns, which outlives the
        // call.
        unsafe { sys::llama_vocab_is_eog(self.vocab(), t) }
    }

    /// Text to tokens. `special` adds BOS etc. as the model wants and parses
    /// special tokens written in the text (as chat templates produce).
    pub fn tokenize(&self, text: &str, special: bool) -> Result<Vec<Token>, Error> {
        let len = i32::try_from(text.len()).map_err(|_| Error::Tokenize("text too long".into()))?;
        let mut toks = vec![0 as Token; text.len() + 8];
        // The tokenizer can log warnings.
        let _log = crate::log_read();
        // SAFETY: `self.vocab()` points into the model this value owns, which outlives the
        // call. `text` is valid for `len` bytes (len = text.len(), checked to fit i32); `toks`
        // has `toks.len()` writable slots, and llama_tokenize writes at most n_tokens_max (a
        // negative max only returns the size needed).
        let mut n = unsafe {
            sys::llama_tokenize(
                self.vocab(),
                text.as_ptr() as *const _,
                len,
                toks.as_mut_ptr(),
                toks.len() as i32,
                special,
                special,
            )
        };
        if n < 0 {
            // buffer too small: -n is the size needed
            toks.resize(n.unsigned_abs() as usize, 0);
            // SAFETY: `self.vocab()` points into the model this value owns, which outlives the
            // call. As above, with `toks` resized to the size llama_tokenize asked for.
            n = unsafe {
                sys::llama_tokenize(
                    self.vocab(),
                    text.as_ptr() as *const _,
                    len,
                    toks.as_mut_ptr(),
                    toks.len() as i32,
                    special,
                    special,
                )
            };
        }
        if n < 0 {
            return Err(Error::Tokenize(format!("llama_tokenize returned {n}")));
        }
        toks.truncate(n as usize);
        Ok(toks)
    }

    /// One token's bytes. May be part of a UTF-8 character; see [`crate::context::Utf8Stream`].
    pub fn piece(&self, t: Token) -> Vec<u8> {
        // llama.cpp looks the token up with `.at()`, which throws for an id
        // outside the vocabulary; that exception would end the process.
        if t < 0 || t as usize >= self.n_vocab() {
            return Vec::new();
        }
        let mut buf = vec![0u8; 64];
        // SAFETY: `self.vocab()` points into the model this value owns, which outlives the
        // call. `buf` has `buf.len()` writable bytes, and llama_token_to_piece writes at most
        // that many (or returns minus the size needed).
        let mut n = unsafe {
            sys::llama_token_to_piece(
                self.vocab(),
                t,
                buf.as_mut_ptr() as *mut _,
                buf.len() as i32,
                0,
                false,
            )
        };
        if n < 0 {
            buf.resize(n.unsigned_abs() as usize, 0);
            // SAFETY: `self.vocab()` points into the model this value owns, which outlives the
            // call. As above, with `buf` resized to the size it asked for.
            n = unsafe {
                sys::llama_token_to_piece(
                    self.vocab(),
                    t,
                    buf.as_mut_ptr() as *mut _,
                    buf.len() as i32,
                    0,
                    false,
                )
            };
        }
        buf.truncate(n.max(0) as usize);
        buf
    }

    /// Format a conversation with the model's chat template. Uses llama.cpp's
    /// built-in template support (no Jinja); fails for templates it doesn't know.
    pub fn apply_chat_template(
        &self,
        messages: &[(String, String)],
        add_assistant: bool,
    ) -> Result<String, Error> {
        let tmpl = self
            .chat_template()
            .ok_or_else(|| Error::Tokenize("the model has no chat template".into()))?;
        let tmpl =
            CString::new(tmpl).map_err(|_| Error::Tokenize("template contains NUL".into()))?;
        let owned: Vec<(CString, CString)> = messages
            .iter()
            .map(|(r, c)| Ok((CString::new(r.as_str())?, CString::new(c.as_str())?)))
            .collect::<Result<_, std::ffi::NulError>>()
            .map_err(|_| Error::Tokenize("message contains NUL".into()))?;
        let msgs: Vec<sys::llama_chat_message> = owned
            .iter()
            .map(|(r, c)| sys::llama_chat_message {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();
        let total: usize = messages.iter().map(|(r, c)| r.len() + c.len()).sum();
        let mut buf = vec![0u8; total * 2 + 256];
        // SAFETY: `tmpl` and every role/content pointer in `msgs` are NUL-terminated strings
        // owned by `tmpl` and `owned`, alive for the call; `msgs` has `msgs.len()` entries;
        // `buf` has `buf.len()` writable bytes, and the function copies at most `length`
        // (= buf.len()) bytes with strncpy, and nothing when the length is negative.
        let mut n = unsafe {
            sys::llama_chat_apply_template(
                tmpl.as_ptr(),
                msgs.as_ptr(),
                msgs.len(),
                add_assistant,
                buf.as_mut_ptr() as *mut _,
                buf.len() as i32,
            )
        };
        if n > buf.len() as i32 {
            buf.resize(n as usize, 0);
            // SAFETY: As above, with `buf` resized to the length the first call asked for.
            n = unsafe {
                sys::llama_chat_apply_template(
                    tmpl.as_ptr(),
                    msgs.as_ptr(),
                    msgs.len(),
                    add_assistant,
                    buf.as_mut_ptr() as *mut _,
                    buf.len() as i32,
                )
            };
        }
        if n < 0 {
            return Err(Error::Tokenize(
                "llama.cpp doesn't support this model's chat template".into(),
            ));
        }
        buf.truncate(n as usize);
        String::from_utf8(buf)
            .map_err(|_| Error::Tokenize("template produced invalid UTF-8".into()))
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let _log = crate::log_read();
        // SAFETY: `self.raw()` is the llama_model this value owns (non-null, freed only in
        // Drop). This is the only place it is freed.
        unsafe { sys::llama_model_free(self.raw()) }
    }
}
