//! ollama-compatible API, so existing ollama clients work unchanged
//! (features.md). Shapes follow ollama's docs/api.md.

use super::{Body, State, content_text, default_sampling};
use crate::{ApiError, GenOptions, rfc3339};
use aie_proto::worker::done::Reason;
use aie_proto::worker::generate_event::Event;
use aie_proto::worker::generate_request::Input;
use aie_proto::worker::{Chat, ChatMessage};
use aie_store::ModelRef;
use axum::Json;
use axum::body::Body as HttpBody;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::{Duration, SystemTime};

pub async fn version() -> Json<Value> {
    Json(json!({ "version": env!("CARGO_PKG_VERSION") }))
}

#[derive(Deserialize, Default)]
pub struct Options {
    temperature: Option<f32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    seed: Option<i64>,
    num_predict: Option<i64>,
    num_ctx: Option<u32>,
    stop: Option<Vec<String>>,
}

/// ollama's `keep_alive`: a duration string or seconds; negative = forever.
fn keep_alive(v: &Option<Value>, default: Duration) -> Result<Option<Duration>, ApiError> {
    match v {
        None | Some(Value::Null) => Ok(Some(default)),
        Some(Value::Number(n)) => {
            let s = n.as_f64().unwrap_or(0.0);
            Ok((s >= 0.0).then(|| Duration::from_secs_f64(s)))
        }
        Some(Value::String(s)) if s.trim_start().starts_with('-') => Ok(None),
        Some(Value::String(s)) => crate::config::parse_duration(s)
            .map(Some)
            .ok_or_else(|| ApiError::bad_request(format!("bad keep_alive {s:?}"))),
        _ => Err(ApiError::bad_request(
            "keep_alive must be a duration or a number of seconds",
        )),
    }
}

fn gen_options(
    d: &crate::Daemon,
    o: Option<Options>,
    ka: &Option<Value>,
) -> Result<GenOptions, ApiError> {
    let o = o.unwrap_or_default();
    let mut s = default_sampling();
    if let Some(v) = o.temperature {
        s.temperature = v;
    }
    if let Some(v) = o.top_k {
        s.top_k = v;
    }
    if let Some(v) = o.top_p {
        s.top_p = v;
    }
    if let Some(v) = o.min_p {
        s.min_p = v;
    }
    if let Some(v) = o.seed
        && v >= 0
    {
        s.seed = v as u32;
    }
    Ok(GenOptions {
        sampling: s,
        max_tokens: o
            .num_predict
            .filter(|n| *n > 0)
            .map(|n| n as u32)
            .unwrap_or(0),
        stop: o.stop.unwrap_or_default(),
        n_ctx: o.num_ctx,
        keep_alive: keep_alive(ka, d.cfg.keep_alive)?,
    })
}

fn done_reason(r: i32) -> &'static str {
    match Reason::try_from(r).unwrap_or(Reason::Unspecified) {
        Reason::Length | Reason::ContextFull => "length",
        _ => "stop",
    }
}

fn ndjson(stream: impl futures_util::Stream<Item = Value> + Send + 'static) -> Response {
    let body =
        HttpBody::from_stream(stream.map(|v| Ok::<_, std::convert::Infallible>(format!("{v}\n"))));
    ([("content-type", "application/x-ndjson")], body).into_response()
}

/// Shared by /api/chat and /api/generate: `wrap` builds each line from a
/// piece of text (or the final line with stats).
async fn run(
    st: State,
    model: String,
    input: Input,
    o: GenOptions,
    stream: bool,
    wrap: fn(&str, &str, Option<&Value>) -> Value,
) -> Result<Response, ApiError> {
    let t0 = std::time::Instant::now();
    let (resolved, events) = st.0.generate(&model, input, o).await?;
    let name = resolved.name;
    let load_ns = t0.elapsed().as_nanos() as u64;
    let finish = move |d: &aie_proto::worker::Done, total_ns: u64| {
        json!({
            "done": true,
            "done_reason": done_reason(d.reason),
            "total_duration": total_ns,
            "load_duration": load_ns.saturating_sub(d.prefill_ns + d.decode_ns),
            "prompt_eval_count": d.prompt_tokens,
            "prompt_eval_duration": d.prefill_ns,
            "eval_count": d.completion_tokens,
            "eval_duration": d.decode_ns,
        })
    };
    if stream {
        let name2 = name.clone();
        let lines = events.map(move |ev| match ev {
            Ok(e) => match e.event {
                Some(Event::Text(t)) => wrap(&name2, &t, None),
                Some(Event::Done(d)) => wrap(
                    &name2,
                    "",
                    Some(&finish(&d, t0.elapsed().as_nanos() as u64)),
                ),
                None => json!({}),
            },
            Err(e) => json!({ "error": e.1 }),
        });
        return Ok(ndjson(lines));
    }
    let mut text = String::new();
    let mut events = Box::pin(events);
    while let Some(ev) = events.next().await {
        match ev?.event {
            Some(Event::Text(t)) => text.push_str(&t),
            Some(Event::Done(d)) => {
                return Ok(Json(wrap(
                    &name,
                    &text,
                    Some(&finish(&d, t0.elapsed().as_nanos() as u64)),
                ))
                .into_response());
            }
            None => {}
        }
    }
    Err(ApiError::internal("generation ended without a result"))
}

#[derive(Deserialize)]
pub struct ChatReq {
    model: String,
    #[serde(default)]
    messages: Vec<Value>,
    stream: Option<bool>,
    options: Option<Options>,
    keep_alive: Option<Value>,
}

pub async fn chat(st: State, Body(r): Body<ChatReq>) -> Result<Response, ApiError> {
    let mut messages = Vec::new();
    for m in &r.messages {
        let role = m
            .get("role")
            .and_then(|x| x.as_str())
            .ok_or_else(|| ApiError::bad_request("message without a role"))?;
        messages.push(ChatMessage {
            role: role.into(),
            content: content_text(m.get("content").unwrap_or(&Value::Null))?,
        });
    }
    if messages.is_empty() {
        // ollama: an empty chat just loads the model
        let d = &st.0;
        let res = d.resolve(&r.model)?;
        let spec = d.load_spec(&res, None)?;
        let ka = keep_alive(&r.keep_alive, d.cfg.keep_alive)?;
        if ka == Some(Duration::ZERO) {
            d.workers.unload(&res.name).await;
        } else {
            drop(
                d.workers
                    .lease(&spec, ka)
                    .await
                    .map_err(ApiError::internal)?,
            );
        }
        return Ok(Json(json!({ "model": res.name, "created_at": rfc3339(SystemTime::now()), "message": { "role": "assistant", "content": "" }, "done": true, "done_reason": if ka == Some(Duration::ZERO) { "unload" } else { "load" } })).into_response());
    }
    let o = gen_options(&st.0, r.options, &r.keep_alive)?;
    let wrap: fn(&str, &str, Option<&Value>) -> Value = |model, text, done| {
        let mut v = json!({ "model": model, "created_at": rfc3339(SystemTime::now()), "message": { "role": "assistant", "content": text }, "done": false });
        if let Some(Value::Object(d)) = done {
            v.as_object_mut().unwrap().extend(d.clone());
        }
        v
    };
    run(
        st,
        r.model,
        Input::Chat(Chat { messages }),
        o,
        r.stream.unwrap_or(true),
        wrap,
    )
    .await
}

#[derive(Deserialize)]
pub struct GenerateReq {
    model: String,
    #[serde(default)]
    prompt: String,
    system: Option<String>,
    #[serde(default)]
    raw: bool,
    stream: Option<bool>,
    options: Option<Options>,
    keep_alive: Option<Value>,
}

pub async fn generate(st: State, Body(r): Body<GenerateReq>) -> Result<Response, ApiError> {
    if r.prompt.is_empty() {
        return chat(
            st,
            Body(ChatReq {
                model: r.model,
                messages: vec![],
                stream: r.stream,
                options: r.options,
                keep_alive: r.keep_alive,
            }),
        )
        .await;
    }
    let input = if r.raw {
        Input::Prompt(r.prompt)
    } else {
        let mut messages = Vec::new();
        if let Some(s) = r.system {
            messages.push(ChatMessage {
                role: "system".into(),
                content: s,
            });
        }
        messages.push(ChatMessage {
            role: "user".into(),
            content: r.prompt,
        });
        Input::Chat(Chat { messages })
    };
    let o = gen_options(&st.0, r.options, &r.keep_alive)?;
    let wrap: fn(&str, &str, Option<&Value>) -> Value = |model, text, done| {
        let mut v = json!({ "model": model, "created_at": rfc3339(SystemTime::now()), "response": text, "done": false });
        if let Some(Value::Object(d)) = done {
            v.as_object_mut().unwrap().extend(d.clone());
        }
        v
    };
    run(st, r.model, input, o, r.stream.unwrap_or(true), wrap).await
}

/// ollama's `details` for a model, from its GGUF header.
pub fn details(path: &std::path::Path) -> Value {
    let Ok(g) = aie_gguf::Gguf::open_header(path) else {
        return json!({ "format": "gguf" });
    };
    let params: u64 = g.tensors.iter().map(|t| t.elements()).sum();
    let mut by_type: BTreeMap<String, u64> = BTreeMap::new();
    for t in &g.tensors {
        *by_type.entry(t.ty.to_string().to_uppercase()).or_default() += t.size;
    }
    let quant = by_type
        .into_iter()
        .max_by_key(|(_, b)| *b)
        .map(|(t, _)| t)
        .unwrap_or_default();
    let size = if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else {
        format!("{:.0}M", params as f64 / 1e6)
    };
    let arch = g.architecture().unwrap_or("").to_string();
    json!({ "format": "gguf", "family": arch, "families": [arch], "parameter_size": size, "quantization_level": quant })
}

pub async fn tags(st: State) -> Result<Json<Value>, ApiError> {
    let models =
        st.0.store
            .list()
            .map_err(|e| ApiError::internal(e.to_string()))?;
    let list: Vec<Value> = models
        .iter()
        .map(|m| {
            let t = SystemTime::UNIX_EPOCH + Duration::from_secs(m.manifest.pulled_at);
            json!({
                "name": m.manifest.name,
                "model": m.manifest.name,
                "modified_at": rfc3339(t),
                "size": m.manifest.size(),
                "digest": m.manifest.files.first().map(|f| f.sha256.clone()).unwrap_or_default(),
                "details": details(&m.path),
            })
        })
        .collect();
    Ok(Json(json!({ "models": list })))
}

#[derive(Deserialize)]
pub struct ModelReq {
    #[serde(alias = "name")]
    model: String,
    stream: Option<bool>,
}

pub async fn show(st: State, Body(r): Body<ModelReq>) -> Result<Json<Value>, ApiError> {
    let res = st.0.resolve(&r.model)?;
    let g =
        aie_gguf::Gguf::open_header(&res.path).map_err(|e| ApiError::internal(e.to_string()))?;
    let mut info = serde_json::Map::new();
    for (k, v) in &g.metadata {
        if k.starts_with("tokenizer.") {
            continue;
        }
        let jv = match v {
            aie_gguf::Value::String(s) => json!(s),
            aie_gguf::Value::Bool(b) => json!(b),
            aie_gguf::Value::Array(..) => continue,
            other => other.as_f64().map(|x| json!(x)).unwrap_or(Value::Null),
        };
        info.insert(k.clone(), jv);
    }
    let template = g
        .get_str("tokenizer.chat_template")
        .unwrap_or("")
        .to_string();
    Ok(Json(json!({
        "modelfile": "",
        "parameters": "",
        "template": template,
        "details": details(&res.path),
        "model_info": info,
        "capabilities": ["completion"],
        "modified_at": rfc3339(SystemTime::now()),
    })))
}

pub async fn pull(st: State, Body(r): Body<ModelReq>) -> Result<Response, ApiError> {
    let target = ModelRef::parse(&r.model).map_err(|e| ApiError::bad_request(e.to_string()))?;
    if matches!(target, ModelRef::Local(_)) {
        return Err(ApiError::bad_request("local files don't need pulling"));
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Value>(256);
    let d = st.0.clone();
    tokio::spawn(async move {
        let _one = d.pulling.lock().await;
        let _ = tx.send(json!({ "status": "pulling manifest" })).await;
        let tx2 = tx.clone();
        let mut last = std::time::Instant::now() - Duration::from_secs(1);
        let mut progress = |p: aie_store::Progress| {
            if p.done == p.total || last.elapsed() > Duration::from_millis(250) {
                last = std::time::Instant::now();
                let _ = tx2.try_send(json!({ "status": format!("pulling {}", p.file), "digest": p.file, "total": p.total, "completed": p.done }));
            }
        };
        let result = match &target {
            ModelRef::Hf(h) => d.store.pull(&d.hub, h, &mut progress).await,
            ModelRef::Oci(o) => d.store.pull_oci(o, &mut progress).await,
            ModelRef::Local(_) => unreachable!(),
        };
        let _ = match result {
            Ok(_) => {
                let _ = tx
                    .send(json!({ "status": "verifying sha256 digest" }))
                    .await;
                let _ = tx.send(json!({ "status": "writing manifest" })).await;
                tx.send(json!({ "status": "success" })).await
            }
            Err(e) => tx.send(json!({ "error": e.to_string() })).await,
        };
    });
    let events = tokio_stream::wrappers::ReceiverStream::new(rx);
    if r.stream.unwrap_or(true) {
        return Ok(ndjson(events));
    }
    let all: Vec<Value> = events.collect().await;
    match all.last() {
        Some(v) if v.get("error").is_some() => Err(ApiError::internal(
            v["error"].as_str().unwrap_or("").to_string(),
        )),
        _ => Ok(Json(json!({ "status": "success" })).into_response()),
    }
}

pub async fn delete(st: State, Body(r): Body<ModelReq>) -> Result<Response, ApiError> {
    let res = st.0.resolve(&r.model)?;
    st.0.workers.unload(&res.name).await;
    let target = ModelRef::parse(&res.name).map_err(|e| ApiError::bad_request(e.to_string()))?;
    if matches!(target, ModelRef::Local(_)) {
        return Err(ApiError::bad_request(
            "local files aren't deleted; remove them yourself",
        ));
    }
    st.0.store
        .remove(&target)
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(json!({})).into_response())
}

pub async fn ps(st: State) -> Json<Value> {
    let mut list = Vec::new();
    for w in st.0.workers.list().await {
        let vram = super::aie::gpu_bytes(&w);
        list.push(json!({
            "name": w.name,
            "model": w.name,
            "size": w.status.weight_bytes,
            "size_vram": vram,
            "digest": "",
            "details": details(&w.model_path),
            "expires_at": w.expires_at().map(rfc3339).unwrap_or_else(|| "9999-12-31T23:59:59Z".into()),
            "context_length": w.n_ctx,
        }));
    }
    Json(json!({ "models": list }))
}
