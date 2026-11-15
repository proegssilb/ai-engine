//! The real binary end to end: `ai-engine serve` starts workers for requests
//! through the ollama and OpenAI APIs, placement can be set and is verified.
//! Needs Qwen3-0.6B Q8_0 pulled into ~/aie-store (or AIE_TEST_STORE); skips
//! otherwise.

use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

const MODEL: &str = "hf.co/Qwen/Qwen3-0.6B-GGUF";

struct Daemon(Child, PathBuf);

impl Drop for Daemon {
    fn drop(&mut self) {
        unsafe { libc_kill(self.0.id() as i32, 15) };
        let _ = self.0.wait();
        let _ = std::fs::remove_dir_all(&self.1);
    }
}

unsafe extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

fn store() -> Option<PathBuf> {
    let s = std::env::var_os("AIE_TEST_STORE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join("aie-store"));
    s.join("manifests/hf.co/Qwen/Qwen3-0.6B-GGUF/Q8_0.json")
        .exists()
        .then_some(s)
}

async fn post(c: &reqwest::Client, base: &str, path: &str, body: Value) -> (u16, Value) {
    let r = c
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap();
    let code = r.status().as_u16();
    (code, r.json().await.unwrap_or(Value::Null))
}

#[tokio::test(flavor = "multi_thread")]
async fn serve_load_chat_place() {
    let Some(store) = store() else {
        eprintln!("skipping: Qwen3-0.6B not in the test store");
        return;
    };
    let home = std::env::temp_dir().join(format!("aie-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        home.join("config.toml"),
        format!(
            "stores = [{:?}, {:?}]\nn_ctx = 2048\nthreads = 4\n",
            store,
            home.join("store2")
        ),
    )
    .unwrap();
    let port = 20000 + (std::process::id() % 20000) as u16;
    let base = format!("http://127.0.0.1:{port}");
    let child = Command::new(env!("CARGO_BIN_EXE_ai-engine"))
        .arg("serve")
        .env("AIE_HOME", &home)
        .env("AIE_HOST", format!("127.0.0.1:{port}"))
        .spawn()
        .unwrap();
    let _d = Daemon(child, home.clone());
    let c = reqwest::Client::new();
    for _ in 0..100 {
        if c.get(&base).send().await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // ollama chat, non-streaming: loads the model with llama.cpp's fit
    let (code, v) = post(&c, &base, "/api/chat", json!({
        "model": MODEL, "stream": false,
        "messages": [{ "role": "user", "content": "What is 2+2? Reply with the number only. /no_think" }],
        "options": { "temperature": 0, "num_predict": 30 }
    })).await;
    assert_eq!(code, 200, "{v}");
    assert!(
        v["message"]["content"].as_str().unwrap().contains('4'),
        "{v}"
    );
    assert_eq!(v["done"], true);

    // OpenAI chat
    let (code, v) = post(
        &c,
        &base,
        "/v1/chat/completions",
        json!({
            "model": MODEL, "messages": [{ "role": "user", "content": "Say hello. /no_think" }],
            "temperature": 0, "max_tokens": 5
        }),
    )
    .await;
    assert_eq!(code, 200, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 5);

    // placement: auto, verified
    let p: Value = c
        .get(format!("{base}/api/aie/placement?model={MODEL}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(p["loaded"]["misplaced"], 0, "{p}");
    assert!(p["intent"].is_null());

    // save an intent (layers pinned in RAM, half the FFN on disk), reload, verify
    let intent = json!({
        "n_ctx": 2048, "layers": [{ "tier": "Ram", "amount": "Rest" }],
        "subsystems": { "DenseFfn": [{ "tier": "Ram", "amount": { "Fraction": 0.5 } }, { "tier": "Disk", "amount": "Rest" }] },
        "embeddings": null, "output": null, "expert_cache_bytes": 0, "allow_repack": false
    });
    let r = c
        .put(format!("{base}/api/aie/placement"))
        .json(&json!({ "model": MODEL, "intent": intent, "reload": true }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let (code, _) = post(
        &c,
        &base,
        "/api/generate",
        json!({ "model": MODEL, "prompt": "Hi", "stream": false, "options": { "num_predict": 3 } }),
    )
    .await;
    assert_eq!(code, 200);
    let p: Value = c
        .get(format!("{base}/api/aie/placement?model={MODEL}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(p["loaded"]["misplaced"], 0, "{p}");
    let ffn = &p["loaded"]["planned"]["dense ffn"];
    assert!(
        ffn["ram"].as_u64().unwrap() > 0 && ffn["disk"].as_u64().unwrap() > 0,
        "{ffn}"
    );
    assert!(p["loaded"]["planned"]["attention"]["ram"].as_u64().unwrap() > 0);

    // live stats: the requests above are counted; the disk-tier FFN half
    // was just read, so it's in RAM
    let stats: Value = c
        .get(format!("{base}/api/aie/placement?model={MODEL}"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let s = &stats["loaded"]["stats"];
    assert!(
        s["requests"].as_u64().unwrap() >= 1 && s["completion_tokens"].as_u64().unwrap() >= 1,
        "{s}"
    );
    assert!(s["disk_tier_bytes"].as_u64().unwrap() > 0, "{s}");
    assert!(s["disk_tier_resident"].as_u64().unwrap() > 0, "{s}");

    // a placement that can't work is refused before it's saved
    let bad = json!({ "n_ctx": 2048, "layers": [{ "tier": "Disk", "amount": "Rest" }], "subsystems": {}, "embeddings": null, "output": null, "expert_cache_bytes": 0, "allow_repack": false });
    let r = c
        .put(format!("{base}/api/aie/placement"))
        .json(&json!({ "model": MODEL, "intent": bad }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 400);

    // keep_alive 0 unloads after the request
    post(&c, &base, "/api/generate", json!({ "model": MODEL, "prompt": "Hi", "stream": false, "keep_alive": 0, "options": { "num_predict": 1 } })).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let ps: Value = c
        .get(format!("{base}/api/ps"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ps["models"].as_array().unwrap().len(), 0, "{ps}");

    // the machine view: devices from the probe subprocess, the store's disk
    let m: Value = c
        .get(format!("{base}/api/aie/machine"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        m["devices"].as_array().is_some_and(|d| !d.is_empty()),
        "{m}"
    );
    assert!(
        m["ram"]["total"].as_u64().unwrap() > 0 && m["disk"]["total"].as_u64().unwrap() > 0,
        "{m}"
    );

    // two stores: both disks shown; pulling into one that isn't configured is refused
    assert_eq!(m["disks"].as_array().unwrap().len(), 2, "{m}");
    let (code, v) = post(
        &c,
        &base,
        "/api/pull",
        json!({ "model": "hf.co/a/b:Q4_K_M", "store": "/nowhere", "stream": false }),
    )
    .await;
    assert_eq!(code, 400, "{v}");

    // unknown models are a clean 404
    let (code, v) = post(&c, &base, "/api/chat", json!({ "model": "hf.co/nobody/nothing:Q4_K_M", "messages": [{ "role": "user", "content": "x" }] })).await;
    assert_eq!(code, 404, "{v}");
}
