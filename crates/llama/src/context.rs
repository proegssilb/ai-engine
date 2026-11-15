//! Inference contexts and generation.

use crate::model::Model;
use crate::{Error, Token, sys};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Context settings.
#[derive(Clone, Debug, PartialEq)]
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

/// How to pick the next token.
#[derive(Clone, Debug, PartialEq)]
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
    params: ContextParams,
}

// A context may move between threads but isn't shared.
unsafe impl Send for Context {}

impl Context {
    pub fn new(model: Arc<Model>, p: ContextParams) -> Result<Context, Error> {
        Self::with_eval_callback(model, p, None)
    }

    pub(crate) fn with_eval_callback(
        model: Arc<Model>,
        p: ContextParams,
        cb: sys::ggml_backend_sched_eval_callback,
    ) -> Result<Context, Error> {
        let ptr = unsafe {
            let mut cp = sys::llama_context_default_params();
            cp.n_ctx = p.n_ctx;
            cp.n_batch = p.n_batch;
            cp.n_ubatch = p.n_batch;
            cp.n_threads = p.threads;
            cp.n_threads_batch = p.threads;
            cp.moe_cache_size = p.moe_cache_bytes as usize;
            cp.op_offload = p.op_offload;
            cp.flash_attn_type = match p.flash_attn {
                Some(true) => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_ENABLED,
                Some(false) => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_DISABLED,
                None => sys::llama_flash_attn_type_LLAMA_FLASH_ATTN_TYPE_AUTO,
            };
            cp.cb_eval = cb;
            sys::llama_init_from_model(model.raw(), cp)
        };
        let ptr = NonNull::new(ptr).ok_or_else(|| Error::Context(crate::log::tail(8)))?;
        Ok(Context {
            ptr,
            model,
            params: p,
        })
    }

    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }

    pub fn n_ctx(&self) -> u32 {
        unsafe { sys::llama_n_ctx(self.ptr.as_ptr()) }
    }

    /// Forget the conversation (clear the KV cache).
    pub fn clear(&mut self) {
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ptr.as_ptr()), true) }
    }

    /// Feed tokens, in batches of `n_batch`.
    pub fn decode(&mut self, tokens: &[Token]) -> Result<(), Error> {
        let mut buf = tokens.to_vec();
        for chunk in buf.chunks_mut(self.params.n_batch.max(1) as usize) {
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

    /// The last position's logits.
    pub fn logits(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                sys::llama_get_logits_ith(self.ptr.as_ptr(), -1),
                self.model.n_vocab(),
            )
        }
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
        unsafe { sys::llama_synchronize(self.ptr.as_ptr()) };
        let prefill = t0.elapsed();

        let t1 = Instant::now();
        let mut out = Vec::new();
        let mut stop = Stop::Length;
        let mut pos = prompt.len();
        while out.len() < max_tokens {
            let mut t =
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
