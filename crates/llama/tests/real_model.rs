//! End-to-end on a real model, CPU only: anatomy -> intent -> plan -> compile
//! -> load -> verify -> generate.
//!
//! Uses `AIE_TEST_MODEL` (a GGUF path), defaulting to Qwen3-0.6B Q8_0 pulled
//! into ~/aie-store by `cargo run -p aie-store --example pull -- ~/aie-store
//! hf.co/Qwen/Qwen3-0.6B-GGUF`. Skips if the file isn't there.

use aie_anatomy::{Anatomy, Subsystem};
use aie_gguf::Gguf;
use aie_llama::context::{Stop, Utf8Stream};
use aie_llama::pinned::PinOptions;
use aie_llama::verify::{self, Where};
use aie_llama::{Context, ContextParams, LoadOptions, Model, Sampling};
use aie_plan::llama::compile;
use aie_plan::{Amount, Intent, Share, Tier, plan};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var_os("AIE_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap())
                .join("aie-store/models/hf.co/Qwen/Qwen3-0.6B-GGUF/Q8_0/Qwen3-0.6B-Q8_0.gguf")
        });
    if p.exists() {
        Some(p)
    } else {
        eprintln!(
            "skipping: no test model at {} (set AIE_TEST_MODEL)",
            p.display()
        );
        None
    }
}

fn params() -> ContextParams {
    ContextParams {
        n_ctx: 1024,
        n_batch: 256,
        threads: 4,
        ..ContextParams::default()
    }
}

/// Load with a plan, check every weight is where the plan put it, return the model.
fn load_verified(path: &Path, a: &Anatomy, intent: &Intent) -> Arc<Model> {
    let p = plan(a, intent, 0).unwrap();
    let settings = compile(a, &p, 0).unwrap();
    let opts = LoadOptions {
        settings,
        pin: PinOptions {
            mlock: false,
            hugepages: true,
            ..Default::default()
        },
    };
    let m = Arc::new(Model::load(path, &opts).unwrap());
    let prompt = m
        .tokenize(
            "The quick brown fox jumps over the lazy dog. "
                .repeat(8)
                .as_str(),
            true,
        )
        .unwrap();
    let r = verify::check(
        &m,
        &params(),
        &prompt,
        a.parts.iter().map(|p| p.name.clone()),
    )
    .unwrap();
    let mismatches = r.compare(&p);
    assert!(
        mismatches.is_empty(),
        "{} tensors misplaced, e.g. {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(3)]
    );
    assert!(r.weights.len() + r.unseen.len() == a.parts.len());
    assert!(r.unseen.len() <= 2, "unverified: {:?}", r.unseen);
    m
}

fn greedy(m: &Arc<Model>, text: &str, n: usize) -> (String, Vec<i32>) {
    let mut ctx = Context::new(m.clone(), params()).unwrap();
    let prompt = m.tokenize(text, true).unwrap();
    let mut out = Utf8Stream::default();
    let mut s = String::new();
    let g = ctx
        .generate(&prompt, n, &Sampling::greedy(), |t| {
            s.push_str(&out.push(&m.piece(t)));
            true
        })
        .unwrap();
    s.push_str(&out.finish());
    assert!(matches!(g.stop, Stop::Length | Stop::EndOfText));
    (s, g.tokens)
}

#[test]
fn placements_are_honoured_and_output_is_identical() {
    let Some(path) = model_path() else { return };
    let a = Anatomy::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    assert!(a.is_supported());

    // 1. Everything in the mapped file except layers, which must be in RAM:
    //    the layers' tensors go to the Rust pinned buffer.
    let all_ram = Intent::all_on(Tier::Ram, 1024);
    // 2. Half the FFN stays in the mapped file (disk tier).
    let mut half = all_ram.clone();
    half.subsystems.insert(
        Subsystem::DenseFfn,
        vec![
            Share::new(Tier::Ram, Amount::Fraction(0.5)),
            Share::all(Tier::Disk),
        ],
    );
    // 3. Embeddings pinned too.
    let mut emb = half.clone();
    emb.embeddings = Some(Tier::Ram);

    let chat = "<|im_start|>user\nName three primary colours.<|im_end|>\n<|im_start|>assistant\n";
    let mut outputs = Vec::new();
    for intent in [&all_ram, &half, &emb] {
        let m = load_verified(&path, &a, intent);
        outputs.push(greedy(&m, chat, 24));
    }
    // CPU kernels are the same wherever the bytes live: identical tokens.
    assert_eq!(outputs[0].1, outputs[1].1);
    assert_eq!(outputs[0].1, outputs[2].1);
    assert!(!outputs[0].0.trim().is_empty());
    eprintln!("generated: {:?}", outputs[0].0);
}

#[test]
fn verify_catches_a_wrong_plan() {
    let Some(path) = model_path() else { return };
    let a = Anatomy::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    // Load everything in the mapped file, then compare against a plan that
    // claims the layers are pinned: every layer tensor must be reported.
    let m = Arc::new(Model::load(&path, &LoadOptions::cpu()).unwrap());
    let claimed = plan(&a, &Intent::all_on(Tier::Ram, 1024), 0).unwrap();
    let prompt = m.tokenize("hello world", true).unwrap();
    let r = verify::check(
        &m,
        &params(),
        &prompt,
        a.parts.iter().map(|p| p.name.clone()),
    )
    .unwrap();
    let mm = r.compare(&claimed);
    let layer_tensors = a.parts.iter().filter(|p| p.layer.is_some()).count();
    assert!(
        mm.len() >= layer_tensors - 2,
        "{} of {layer_tensors}",
        mm.len()
    );
    assert!(
        mm.iter()
            .all(|x| x.planned == Tier::Ram && x.actual == Where::Tier(Tier::Disk))
    );
}

#[test]
fn chat_template_and_tokens() {
    let Some(path) = model_path() else { return };
    let m = Model::load(&path, &LoadOptions::cpu()).unwrap();
    assert!(m.chat_template().is_some());
    let text = m
        .apply_chat_template(&[("user".into(), "hi".into())], true)
        .unwrap();
    assert!(
        text.contains("<|im_start|>user") && text.ends_with("<|im_start|>assistant\n"),
        "{text:?}"
    );
    let toks = m.tokenize(&text, true).unwrap();
    assert!(
        toks.len() < 16,
        "special tokens parsed as single tokens: {toks:?}"
    );
    let mut u = Utf8Stream::default();
    let round: String = toks.iter().map(|&t| u.push(&m.piece(t))).collect();
    assert!(round.contains("hi"));
    assert!(m.description().contains("qwen3"));
}

#[test]
fn context_full_and_errors() {
    let Some(path) = model_path() else { return };
    let m = Arc::new(Model::load(&path, &LoadOptions::cpu()).unwrap());
    let mut ctx = Context::new(
        m.clone(),
        ContextParams {
            n_ctx: 64,
            n_batch: 64,
            threads: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let prompt = m.tokenize("one two three", true).unwrap();
    let g = ctx
        .generate(&prompt, 1000, &Sampling::greedy(), |_| true)
        .unwrap();
    assert!(
        matches!(g.stop, Stop::ContextFull | Stop::EndOfText),
        "{:?}",
        g.stop
    );
    assert!(ctx.generate(&[], 4, &Sampling::greedy(), |_| true).is_err());
    // llama.cpp rounds n_ctx up (64 asked, more given); use what it gave
    let long = vec![prompt[0]; ctx.n_ctx() as usize + 1];
    assert!(
        ctx.generate(&long, 4, &Sampling::greedy(), |_| true)
            .is_err()
    );
    // a stop from the caller
    let g = ctx
        .generate(&prompt, 50, &Sampling::default(), |_| false)
        .unwrap();
    assert_eq!((g.stop, g.tokens.len()), (Stop::Caller, 1));
}

#[test]
fn llama_cpp_fit_runs_and_resolves_to_a_plan() {
    let Some(path) = model_path() else { return };
    // No GPUs here: the fit has nothing to offload, but must run cleanly and
    // produce settings our resolver can read.
    let f = aie_llama::fit::fit(&path, 2048, 512, 512 << 20).unwrap();
    let a = Anatomy::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    let tiers = aie_plan::llama::resolve(&a, &f.settings).unwrap();
    assert_eq!(tiers.len(), a.parts.len());
    assert!(f.settings.n_ctx >= 512);
}
