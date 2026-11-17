//! A real worker on a Unix socket, with a real model (CPU). Uses
//! `AIE_TEST_MODEL`, defaulting to Qwen3-0.6B in ~/aie-store; skips if missing.

#![forbid(unsafe_code)]

use aie_anatomy::Anatomy;
use aie_llama::{ContextParams, LoadOptions};
use aie_plan::{Intent, Tier, llama::compile, plan};
use aie_proto::worker::generate_event::Event;
use aie_proto::worker::generate_request::Input;
use aie_proto::worker::{
    Chat, ChatMessage, GenerateRequest, Sampling, StatusRequest, done::Reason,
};
use aie_worker::{Config, Placement, connect, run};
use std::path::PathBuf;
use std::time::Duration;

fn model_path() -> Option<PathBuf> {
    let p = std::env::var_os("AIE_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap())
                .join("aie-store/models/hf.co/Qwen/Qwen3-0.6B-GGUF/Q8_0/Qwen3-0.6B-Q8_0.gguf")
        });
    p.exists().then_some(p)
}

fn chat(text: &str, max: u32, stop: Vec<String>) -> GenerateRequest {
    GenerateRequest {
        input: Some(Input::Chat(Chat {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: text.into(),
            }],
        })),
        sampling: Some(Sampling {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        }),
        max_tokens: max,
        stop,
    }
}

async fn collect(
    client: &mut aie_proto::worker::worker_client::WorkerClient<tonic::transport::Channel>,
    req: GenerateRequest,
) -> (String, aie_proto::worker::Done) {
    let mut stream = client.generate(req).await.unwrap().into_inner();
    let mut text = String::new();
    while let Some(ev) = stream.message().await.unwrap() {
        match ev.event.unwrap() {
            Event::Text(t) => text.push_str(&t),
            Event::Done(d) => return (text, d),
        }
    }
    panic!("stream ended without Done")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn serves_a_model_over_a_unix_socket() {
    let Some(path) = model_path() else {
        eprintln!("skipping: no test model");
        return;
    };
    let anatomy = Anatomy::from_gguf(&aie_gguf::Gguf::open(&path).unwrap()).unwrap();
    let p = plan(&anatomy, &Intent::all_on(Tier::Ram, 1024), 0).unwrap();
    let settings = compile(&anatomy, &p, 0).unwrap();
    let mut pin = LoadOptions::cpu().pin;
    pin.mlock = false;
    let cfg = Config {
        model: path,
        placement: Placement::Planned { plan: p, settings },
        pin,
        context: ContextParams {
            n_ctx: 1024,
            n_batch: 256,
            threads: 4,
            ..Default::default()
        },
    };

    let socket = std::env::temp_dir().join(format!("aie-worker-test-{}.sock", std::process::id()));
    let s2 = socket.clone();
    tokio::spawn(async move { run(cfg, &s2).await.unwrap() });

    // wait for the socket and a successful Status (load takes a few seconds)
    let mut client = None;
    for _ in 0..300 {
        if let Ok(mut c) = connect(&socket).await
            && c.status(StatusRequest {}).await.is_ok()
        {
            client = Some(c);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let mut client = client.expect("worker never became ready");
    let st = client.status(StatusRequest {}).await.unwrap().into_inner();
    assert_eq!(st.misplaced, 0, "placement not honoured");
    assert!(st.description.contains("qwen3") && st.n_ctx >= 1024 && !st.placement_json.is_empty());
    let plan: aie_plan::Plan = serde_json::from_str(&st.plan_json).unwrap();
    assert!(plan.layer_home.iter().all(|t| *t == Tier::Ram));

    // a whole answer
    let (text, done) = collect(
        &mut client,
        chat("What is 2+2? Answer with just the number.", 200, vec![]),
    )
    .await;
    assert!(text.contains('4'), "{text:?}");
    assert!(matches!(
        Reason::try_from(done.reason).unwrap(),
        Reason::EndOfText | Reason::Length
    ));
    assert!(done.prompt_tokens > 5 && done.completion_tokens > 0);

    // a stop string: Qwen3 starts with "<think>", stop right there
    let (text, done) = collect(&mut client, chat("Say hi.", 50, vec!["<think>".into()])).await;
    assert_eq!(text, "");
    assert_eq!(Reason::try_from(done.reason).unwrap(), Reason::StopString);

    // max_tokens
    let (_, done) = collect(&mut client, chat("Count to one hundred.", 5, vec![])).await;
    assert_eq!(
        (
            Reason::try_from(done.reason).unwrap(),
            done.completion_tokens
        ),
        (Reason::Length, 5)
    );

    // dropping the stream cancels; the worker is free for the next request
    let mut s = client
        .generate(chat("Write a long story.", 500, vec![]))
        .await
        .unwrap()
        .into_inner();
    let _ = s.message().await.unwrap();
    drop(s);
    let (_, done) = tokio::time::timeout(
        Duration::from_secs(60),
        collect(&mut client, chat("Hi", 3, vec![])),
    )
    .await
    .expect("worker stuck");
    assert_eq!(done.completion_tokens, 3);

    let toks = client
        .tokenize(aie_proto::worker::TokenizeRequest {
            text: "hello world".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!toks.tokens.is_empty());
    let _ = std::fs::remove_file(&socket);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn auto_placement_is_llama_cpp_fit_and_checked() {
    let Some(path) = model_path() else {
        eprintln!("skipping: no test model");
        return;
    };
    let cfg = Config {
        model: path,
        placement: Placement::Auto {
            n_ctx_min: 512,
            margin: 512 << 20,
        },
        pin: LoadOptions::cpu().pin,
        context: ContextParams {
            n_ctx: 1024,
            n_batch: 256,
            threads: 4,
            ..Default::default()
        },
    };
    let socket = std::env::temp_dir().join(format!("aie-worker-auto-{}.sock", std::process::id()));
    let s2 = socket.clone();
    tokio::spawn(async move { run(cfg, &s2).await.unwrap() });
    let mut client = None;
    for _ in 0..300 {
        if let Ok(mut c) = connect(&socket).await
            && c.status(StatusRequest {}).await.is_ok()
        {
            client = Some(c);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let mut client = client.expect("worker never became ready");
    let st = client.status(StatusRequest {}).await.unwrap().into_inner();
    assert_eq!(st.misplaced, 0);
    // No GPU here: llama.cpp keeps everything on the CPU, i.e. the mapped file.
    let plan: aie_plan::Plan = serde_json::from_str(&st.plan_json).unwrap();
    assert!(plan.tensors.values().all(|t| *t == Tier::Disk));
    let (_, done) = collect(&mut client, chat("Hi", 3, vec![])).await;
    assert_eq!(done.completion_tokens, 3);
    let _ = std::fs::remove_file(&socket);
}
