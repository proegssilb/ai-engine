//! The embedded web UI, served at /ui/.

use axum::extract::Path;
use axum::http::header;
use axum::response::{Html, IntoResponse, Redirect, Response};

include!(concat!(env!("OUT_DIR"), "/ui_assets.rs"));

fn mime(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("json") => "application/json",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn asset(path: &str) -> Option<Response> {
    let (_, bytes) = ASSETS.iter().find(|(p, _)| *p == path)?;
    let cache = if path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    Some(
        (
            [
                (header::CONTENT_TYPE, mime(path)),
                (header::CACHE_CONTROL, cache),
            ],
            *bytes,
        )
            .into_response(),
    )
}

const NOT_BUILT: &str = "<!doctype html><title>ai-engine</title><body style=\"font-family:system-ui;background:#15171c;color:#e7e9ee;padding:40px\">\
<h1>The web UI isn't built into this binary</h1><p>Build it, then rebuild ai-engine:</p>\
<pre>cd ui &amp;&amp; npm install &amp;&amp; npm run build\ncargo build --release -p ai-engine</pre></body>";

pub async fn index() -> Response {
    asset("index.html").unwrap_or_else(|| Html(NOT_BUILT).into_response())
}

pub async fn file(Path(path): Path<String>) -> Response {
    // Unknown paths get the app (client-side routes); missing assets 404.
    asset(&path).unwrap_or_else(|| {
        if path.starts_with("assets/") {
            axum::http::StatusCode::NOT_FOUND.into_response()
        } else {
            app_index()
        }
    })
}

fn app_index() -> Response {
    asset("index.html").unwrap_or_else(|| Html(NOT_BUILT).into_response())
}

pub async fn redirect() -> Redirect {
    Redirect::permanent("/ui/")
}
