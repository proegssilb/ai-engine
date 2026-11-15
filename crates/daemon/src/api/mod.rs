//! HTTP routes.

pub mod aie;
pub mod ollama;
pub mod openai;

use crate::{ApiError, Daemon};
use aie_proto::worker::Sampling;
use axum::Router;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use std::sync::Arc;

pub type State = axum::extract::State<Arc<Daemon>>;

pub fn router(d: Arc<Daemon>) -> Router {
    Router::new()
        // ollama-compatible
        .route("/", get(|| async { "ai-engine is running" }))
        .route("/api/version", get(ollama::version))
        .route("/api/tags", get(ollama::tags))
        .route("/api/ps", get(ollama::ps))
        .route("/api/show", post(ollama::show))
        .route("/api/pull", post(ollama::pull))
        .route("/api/delete", delete(ollama::delete))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        // OpenAI-compatible
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat))
        .route("/v1/completions", post(openai::completions))
        // placement
        .route(
            "/api/aie/placement",
            get(aie::get_placement).put(aie::put_placement),
        )
        .route("/api/aie/machine", get(aie::machine))
        .with_state(d)
}

/// JSON body regardless of `Content-Type`: ollama accepts requests without
/// one (`curl -d`), so clients rely on it.
pub struct Body<T>(pub T);

impl<T: serde::de::DeserializeOwned, S: Send + Sync> axum::extract::FromRequest<S> for Body<T> {
    type Rejection = ApiError;
    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, ApiError> {
        let bytes = axum::body::Bytes::from_request(req, state)
            .await
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        let bytes = if bytes.is_empty() {
            axum::body::Bytes::from_static(b"{}")
        } else {
            bytes
        };
        serde_json::from_slice(&bytes)
            .map(Body)
            .map_err(|e| ApiError::bad_request(format!("invalid JSON: {e}")))
    }
}

/// ollama-style error body: `{"error": "..."}`.
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let code = StatusCode::from_u16(self.0).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (code, axum::Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

/// ollama's sampling defaults.
pub fn default_sampling() -> Sampling {
    Sampling {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.0,
        seed: u32::MAX,
    }
}

/// Text of a message's `content`: a string, or OpenAI's array of parts
/// (text parts joined; other kinds rejected).
pub fn content_text(v: &serde_json::Value) -> Result<String, ApiError> {
    match v {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match p.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        out.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or(""))
                    }
                    Some(other) => {
                        return Err(ApiError::bad_request(format!(
                            "content part type {other:?} isn't supported"
                        )));
                    }
                    None => return Err(ApiError::bad_request("content part without a type")),
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad_request(
            "message content must be a string or an array of parts",
        )),
    }
}

/// `stop` as a string or a list of strings.
pub fn stop_list(v: Option<&serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        _ => vec![],
    }
}
