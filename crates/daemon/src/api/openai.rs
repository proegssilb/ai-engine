//! OpenAI-compatible API: /v1/models, /v1/chat/completions, /v1/completions.

use super::{Body, State, content_text, default_sampling, stop_list};
use crate::{ApiError, GenOptions};
use aie_proto::worker::done::Reason;
use aie_proto::worker::generate_event::Event;
use aie_proto::worker::generate_request::Input;
use aie_proto::worker::{Chat, ChatMessage};
use axum::Json;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};

/// OpenAI-style error body.
fn err(e: ApiError) -> Response {
    let code = StatusCode::from_u16(e.0).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let kind = if e.0 == 404 {
        "not_found_error"
    } else if e.0 < 500 {
        "invalid_request_error"
    } else {
        "server_error"
    };
    (
        code,
        Json(json!({ "error": { "message": e.1, "type": kind } })),
    )
        .into_response()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn finish_reason(r: i32) -> &'static str {
    match Reason::try_from(r).unwrap_or(Reason::Unspecified) {
        Reason::Length | Reason::ContextFull => "length",
        _ => "stop",
    }
}

pub async fn models(st: State) -> Response {
    match st.0.store.list() {
        Ok(ms) => Json(json!({
            "object": "list",
            "data": ms.iter().map(|m| json!({ "id": m.manifest.name, "object": "model", "created": m.manifest.pulled_at, "owned_by": "ai-engine" })).collect::<Vec<_>>(),
        }))
        .into_response(),
        Err(e) => err(ApiError::internal(e.to_string())),
    }
}

#[derive(Deserialize)]
pub struct Req {
    model: String,
    #[serde(default)]
    messages: Vec<Value>,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<i64>,
    stop: Option<Value>,
}

fn options(d: &crate::Daemon, r: &Req) -> GenOptions {
    let mut s = default_sampling();
    if let Some(t) = r.temperature {
        s.temperature = t;
    }
    if let Some(p) = r.top_p {
        s.top_p = p;
    }
    if let Some(v) = r.seed
        && v >= 0
    {
        s.seed = v as u32;
    }
    GenOptions {
        sampling: s,
        max_tokens: r.max_completion_tokens.or(r.max_tokens).unwrap_or(0),
        stop: stop_list(r.stop.as_ref()),
        n_ctx: None,
        keep_alive: Some(d.cfg.keep_alive),
    }
}

pub async fn chat(st: State, Body(r): Body<Req>) -> Response {
    let mut messages = Vec::new();
    for m in &r.messages {
        let Some(role) = m.get("role").and_then(|x| x.as_str()) else {
            return err(ApiError::bad_request("message without a role"));
        };
        let role = if role == "developer" { "system" } else { role };
        match content_text(m.get("content").unwrap_or(&Value::Null)) {
            Ok(c) => messages.push(ChatMessage {
                role: role.into(),
                content: c,
            }),
            Err(e) => return err(e),
        }
    }
    if messages.is_empty() {
        return err(ApiError::bad_request("messages is empty"));
    }
    let o = options(&st.0, &r);
    respond(
        st,
        r.model,
        Input::Chat(Chat { messages }),
        o,
        r.stream,
        true,
    )
    .await
}

pub async fn completions(st: State, Body(r): Body<Req>) -> Response {
    let prompt = match &r.prompt {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => {
            a[0].as_str().unwrap().to_string()
        }
        _ => return err(ApiError::bad_request("prompt must be a string")),
    };
    let o = options(&st.0, &r);
    respond(st, r.model, Input::Prompt(prompt), o, r.stream, false).await
}

async fn respond(
    st: State,
    model: String,
    input: Input,
    o: GenOptions,
    stream: bool,
    chat: bool,
) -> Response {
    let (resolved, events) = match st.0.generate(&model, input, o).await {
        Ok(x) => x,
        Err(e) => return err(e),
    };
    let name = resolved.name;
    let id = format!(
        "{}-{:x}",
        if chat { "chatcmpl" } else { "cmpl" },
        now() ^ (std::process::id() as u64) << 32 ^ rand_suffix()
    );
    let created = now();
    let object = if chat {
        "chat.completion"
    } else {
        "text_completion"
    };

    if stream {
        let (id2, name2) = (id.clone(), name.clone());
        let mut first = chat;
        let chunks = events.flat_map(move |ev| {
            let mut out: Vec<Value> = Vec::new();
            let chunk = |choice: Value| {
                json!({ "id": id2, "object": if chat { "chat.completion.chunk" } else { "text_completion" }, "created": created, "model": name2, "choices": [choice] })
            };
            match ev {
                Ok(e) => match e.event {
                    Some(Event::Text(t)) => {
                        if chat {
                            let delta = if first { json!({ "role": "assistant", "content": t }) } else { json!({ "content": t }) };
                            first = false;
                            out.push(chunk(json!({ "index": 0, "delta": delta, "finish_reason": Value::Null })));
                        } else {
                            out.push(chunk(json!({ "index": 0, "text": t, "finish_reason": Value::Null })));
                        }
                    }
                    Some(Event::Done(d)) => {
                        let fin = finish_reason(d.reason);
                        let mut c = if chat {
                            chunk(json!({ "index": 0, "delta": {}, "finish_reason": fin }))
                        } else {
                            chunk(json!({ "index": 0, "text": "", "finish_reason": fin }))
                        };
                        c["usage"] = json!({ "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens, "total_tokens": d.prompt_tokens + d.completion_tokens });
                        out.push(c);
                    }
                    None => {}
                },
                Err(e) => out.push(json!({ "error": { "message": e.1, "type": "server_error" } })),
            }
            futures_util::stream::iter(out)
        });
        let sse = chunks
            .map(|v| Ok::<_, std::convert::Infallible>(SseEvent::default().data(v.to_string())))
            .chain(futures_util::stream::once(async {
                Ok(SseEvent::default().data("[DONE]"))
            }));
        return Sse::new(sse).into_response();
    }

    let mut text = String::new();
    let mut events = Box::pin(events);
    while let Some(ev) = events.next().await {
        match ev {
            Err(e) => return err(e),
            Ok(e) => match e.event {
                Some(Event::Text(t)) => text.push_str(&t),
                Some(Event::Done(d)) => {
                    let usage = json!({ "prompt_tokens": d.prompt_tokens, "completion_tokens": d.completion_tokens, "total_tokens": d.prompt_tokens + d.completion_tokens });
                    let choice = if chat {
                        json!({ "index": 0, "message": { "role": "assistant", "content": text }, "finish_reason": finish_reason(d.reason) })
                    } else {
                        json!({ "index": 0, "text": text, "finish_reason": finish_reason(d.reason) })
                    };
                    return Json(json!({ "id": id, "object": object, "created": created, "model": name, "choices": [choice], "usage": usage })).into_response();
                }
                None => {}
            },
        }
    }
    err(ApiError::internal("generation ended without a result"))
}

fn rand_suffix() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(now());
    h.finish()
}
