//! Inference contexts and generation.

use crate::model::Model;
use crate::{Error, Token, sys};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Context settings.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ContextParams {
    pub n_ctx: u32,
    /// Tokens per decode call (and per micro-batch).
    pub n_batch: u32,
    pub threads: i32,
    /// llama.cpp's GPU expert cache, bytes; 0 = off.
    pub moe_cache_bytes: u64,
    /// Let the scheduler run big batches of host weights on GPU 0 (D4: on by default).
    pub op_offload: bool,
    /// `None` = llama.cpp decides.
    pub flash_attn: Option<bool>,
}

impl Default for ContextParams {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        ContextParams {
            n_ctx: 4096,
            n_batch: 512,
            threads,
            moe_cache_bytes: 0,
            op_offload: true,
            flash_attn: None,
        }
    }
}

impl ContextParams {
    /// llama.cpp's context parameters (shared by contexts and the dry run).
    pub(crate) fn raw(&self) -> sys::llama_context_params {
        // SAFETY: Returns a struct by value; no pointers or preconditions.
        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = self.n_ctx;
        cp.n_batch = self.n_batch;
        cp.n_ubatch = self.n_batch;
        cp.n_threads = self.threads;
        cp.n_threads_batch = self.threads;
        cp.moe_cache_size = self.moe_cache_bytes as usize;
        cp.op_offload = self.op_offload;
        cp.flash_attn_type = match self.flash_attn {
            Some(true) => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED,
            Some(false) => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
            None => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
        };
        cp
    }
}

/// How to pick the next token.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sampling {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u32,
}

impl Sampling {
    pub fn greedy() -> Sampling {
        Sampling {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        }
    }
}

impl Default for Sampling {
    /// llama.cpp's CLI defaults.
    fn default() -> Self {
        Sampling {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: 0xFFFF_FFFF,
        }
    }
}

struct Sampler(NonNull<sys::llama_sampler>);

impl Sampler {
    fn new(s: &Sampling) -> Sampler {
        // SAFETY: Each sampler is created by its llama.cpp init function and immediately handed
        // to the chain, which takes ownership; the chain pointer is checked non-null below and
        // freed once in Sampler's Drop.
        unsafe {
            let chain = sys::llama_sampler_chain_init(sys::llama_sampler_chain_default_params());
            if s.temperature <= 0.0 {
                sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_greedy());
            } else {
                if s.top_k > 0 {
                    sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_k(s.top_k));
                }
                if s.top_p < 1.0 {
                    sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_p(s.top_p, 1));
                }
                if s.min_p > 0.0 {
                    sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_min_p(s.min_p, 1));
                }
                sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_temp(s.temperature));
                sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_dist(s.seed));
            }
            Sampler(NonNull::new(chain).expect("sampler chain"))
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        // SAFETY: `self.0` is the chain created in `Sampler::new`, owned by this value and
        // freed only here.
        unsafe { sys::llama_sampler_free(self.0.as_ptr()) }
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The model produced an end-of-generation token.
    EndOfText,
    /// `max_tokens` reached.
    Length,
    /// The token callback asked to stop (e.g. a stop string, a cancelled request).
    Caller,
    /// The context is full.
    ContextFull,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub prompt_tokens: usize,
    pub tokens: Vec<Token>,
    pub stop: Stop,
    pub prefill: Duration,
    pub decode: Duration,
}

impl Generation {
    pub fn prefill_tps(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill.as_secs_f64().max(1e-9)
    }
    pub fn decode_tps(&self) -> f64 {
        self.tokens.len().saturating_sub(1) as f64 / self.decode.as_secs_f64().max(1e-9)
    }
}

/// An inference context: KV cache and scratch for one conversation at a time.
pub struct Context {
    ptr: NonNull<sys::llama_context>,
    model: Arc<Model>,
}

// A context may move between threads but isn't shared.
// SAFETY: A llama_context is used by one thread at a time (Context is Send but not Sync, and
// every call takes `&self`/`&mut self`); llama.cpp has no thread affinity for contexts. The
// model it uses is kept alive by the Arc<Model> field.
unsafe impl Send for Context {}

impl Context {
    pub fn new(model: Arc<Model>, p: ContextParams) -> Result<Context, Error> {
        // SAFETY: no callback.
        unsafe { Self::with_eval_callback(model, p, None) }
    }

    /// # Safety
    ///
    /// `cb`, if given, must be sound to call from ggml's scheduler with any
    /// tensor of the graph and a null user-data pointer, for as long as the
    /// context lives.
    pub(crate) unsafe fn with_eval_callback(
        model: Arc<Model>,
        p: ContextParams,
        cb: sys::ggml_backend_sched_eval_callback,
    ) -> Result<Context, Error> {
        let _log = crate::log_read();
        // SAFETY: `model.raw()` is a live model (held by the Arc we store in the Context, so it
        // outlives the context); `cp` is passed by value; `cb` is None or sound to call (this
        // function's contract).
        let ptr = unsafe {
            let mut cp = p.raw();
            cp.cb_eval = cb;
            sys::llama_init_from_model(model.raw(), cp)
        };
        let ptr = NonNull::new(ptr).ok_or_else(|| Error::Context(crate::log::tail(8)))?;
        Ok(Context { ptr, model })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    /// What this context really allocated, per device and on the host.
    pub fn memory_breakdown(&self) -> Vec<crate::fit::Measured> {
        let _log = crate::log_read();
        // SAFETY: `self.ptr` is the live context this value owns, kept alive by `&self`.
        unsafe { crate::fit::breakdown(self.ptr.as_ptr()) }
    }

    pub fn n_ctx(&self) -> u32 {
        // SAFETY: `self.ptr` is the llama_context this value owns, created non-null in `new`
        // and freed only in Drop.
        unsafe { sys::llama_n_ctx(self.ptr.as_ptr()) }
    }

    /// Forget the conversation (clear the KV cache).
    pub fn clear(&mut self) {
        let _log = crate::log_read();
        // SAFETY: `self.ptr` is the llama_context this value owns, created non-null in `new`
        // and freed only in Drop. The memory handle belongs to the context.
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ptr.as_ptr()), true) }
    }

    /// Feed tokens, in batches of the context's batch size.
    pub fn decode(&mut self, tokens: &[Token]) -> Result<(), Error> {
        let _log = crate::log_read();
        // llama.cpp clamps n_batch to n_ctx and asserts each batch fits, so
        // chunk by its effective value, not the one we asked for.
        // SAFETY: `self.ptr` is the llama_context this value owns (freed only in Drop).
        let n_batch = unsafe { sys::llama_n_batch(self.ptr.as_ptr()) }.max(1);
        let mut buf = tokens.to_vec();
        for chunk in buf.chunks_mut(n_batch as usize) {
            // SAFETY: `self.ptr` is the llama_context this value owns, created non-null in
            // `new` and freed only in Drop. `chunk` is a live, mutable slice of `chunk.len()`
            // tokens (at most n_batch, so i32 can't overflow); llama_batch_get_one only borrows
            // it for the duration of llama_decode.
            let r = unsafe {
                sys::llama_decode(
                    self.ptr.as_ptr(),
                    sys::llama_batch_get_one(chunk.as_mut_ptr(), chunk.len() as i32),
                )
            };
            if r != 0 {
                return Err(Error::Decode(match r {
                    1 => "no room in the KV cache (context full)".into(),
                    2 => "aborted".into(),
                    r => format!("llama_decode returned {r}: {}", crate::log::tail(4)),
                }));
            }
        }
        Ok(())
    }

    /// Generate from a fresh conversation. `on_token` gets each token and
    /// returns false to stop.
    pub fn generate(
        &mut self,
        prompt: &[Token],
        max_tokens: usize,
        sampling: &Sampling,
        mut on_token: impl FnMut(Token) -> bool,
    ) -> Result<Generation, Error> {
        if prompt.is_empty() {
            return Err(Error::Decode("empty prompt".into()));
        }
        if prompt.len() >= self.n_ctx() as usize {
            return Err(Error::Decode(format!(
                "prompt is {} tokens, context is {}",
                prompt.len(),
                self.n_ctx()
            )));
        }
        self.clear();
        let sampler = Sampler::new(sampling);
        let t0 = Instant::now();
        self.decode(prompt)?;
        // After decode returns: it takes the same lock, and a thread mustn't
        // hold a read lock twice (a waiting writer would deadlock it).
        let _log = crate::log_read();
        // SAFETY: `self.ptr` is the llama_context this value owns, created non-null in `new`
        // and freed only in Drop.
        unsafe { sys::llama_synchronize(self.ptr.as_ptr()) };
        let prefill = t0.elapsed();

        let t1 = Instant::now();
        let mut out = Vec::new();
        let mut stop = Stop::Length;
        let mut pos = prompt.len();
        while out.len() < max_tokens {
            let mut t =
                // SAFETY: `self.ptr` is the llama_context this value owns, created non-null in
                // `new` and freed only in Drop. The sampler chain is live (owned by `sampler`),
                // and the prompt was decoded above, so the last position has logits (an empty
                // prompt is rejected at the top).
                unsafe { sys::llama_sampler_sample(sampler.0.as_ptr(), self.ptr.as_ptr(), -1) };
            if self.model.is_eog(t) {
                stop = Stop::EndOfText;
                break;
            }
            out.push(t);
            if !on_token(t) {
                stop = Stop::Caller;
                break;
            }
            if out.len() == max_tokens {
                break;
            }
            if pos + 1 >= self.n_ctx() as usize {
                stop = Stop::ContextFull;
                break;
            }
            // SAFETY: `self.ptr` is the llama_context this value owns, created non-null in
            // `new` and freed only in Drop. `t` is a local token that outlives the call; the
            // batch only borrows it.
            let r = unsafe {
                sys::llama_decode(self.ptr.as_ptr(), sys::llama_batch_get_one(&mut t, 1))
            };
            if r != 0 {
                return Err(Error::Decode(format!("llama_decode returned {r}")));
            }
            pos += 1;
        }
        Ok(Generation {
            prompt_tokens: prompt.len(),
            tokens: out,
            stop,
            prefill,
            decode: t1.elapsed(),
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let _log = crate::log_read();
        // SAFETY: `self.ptr` is the llama_context this value owns, created non-null in `new`
        // and freed only in Drop. This is the only place it is freed; the Arc<Model> field is
        // dropped after, so the model outlives the context.
        unsafe { sys::llama_free(self.ptr.as_ptr()) }
    }
}

/// Turns token bytes into text without splitting UTF-8 characters: bytes of
/// an incomplete character wait for the next token.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    /// Add bytes; returns whatever is complete text so far.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) => {
                let good = e.valid_up_to();
                // An invalid (not just incomplete) sequence: emit it lossily.
                let cut = if e.error_len().is_some() {
                    self.pending.len()
                } else {
                    good
                };
                let s = String::from_utf8_lossy(&self.pending[..cut]).into_owned();
                self.pending.drain(..cut);
                s
            }
        }
    }

    /// Whatever's left at the end (lossily).
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_stream_holds_partial_characters() {
        let mut u = Utf8Stream::default();
        let euro = "€".as_bytes(); // 3 bytes
        assert_eq!(u.push(b"a"), "a");
        assert_eq!(u.push(&euro[..2]), "");
        assert_eq!(u.push(&euro[2..]), "€");
        assert_eq!(u.push(&[0xff, b'b']), "\u{fffd}b");
        assert_eq!(u.push(&euro[..1]), "");
        assert_eq!(u.finish(), "\u{fffd}");
    }
}
