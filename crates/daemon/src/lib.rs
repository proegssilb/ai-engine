//! The daemon: model store, worker lifecycle, and the HTTP APIs
//! (OpenAI-compatible under `/v1`, ollama-compatible under `/api`, and
//! placement under `/api/aie`). It never initialises CUDA itself: planning
//! against devices and llama.cpp's fit run in the workers.

#![forbid(unsafe_code)]

pub mod api;
pub mod apply;
pub mod config;
pub mod memory;
pub mod stores;
pub mod workers;

pub use config::Config;

use aie_plan::Intent;
use aie_proto::worker::{GenerateEvent, GenerateRequest, Sampling, generate_request::Input};
use aie_store::ModelRef;
use aie_worker::Placement;
use futures_util::{Stream, StreamExt};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use workers::{Lease, LoadSpec, Manager};

/// Shared daemon state.
pub struct Daemon {
    pub cfg: Config,
    pub stores: stores::Stores,
    pub hub: aie_store::hf::Hub,
    pub workers: Arc<Manager>,
    /// Host memory over the last minute, for the machine view's RAM bar.
    pub memory: memory::Sampler,
    /// One pull at a time (the store serialises per blob).
    pub pulling: tokio::sync::Mutex<()>,
}

/// An API error: HTTP status and message.
#[derive(Debug)]
pub struct ApiError(pub u16, pub String);

impl ApiError {
    pub fn not_found(m: impl Into<String>) -> Self {
        ApiError(404, m.into())
    }
    pub fn bad_request(m: impl Into<String>) -> Self {
        ApiError(400, m.into())
    }
    pub fn internal(m: impl Into<String>) -> Self {
        ApiError(500, m.into())
    }
}

/// A model resolved against the store.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// Canonical name, e.g. `hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0`, or the file path.
    pub name: String,
    pub path: PathBuf,
    pub manifest: Option<aie_store::Manifest>,
}

/// How a request wants to generate.
pub struct GenOptions {
    pub sampling: Sampling,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    pub n_ctx: Option<u32>,
    /// `None` = stay loaded; `Some(0)` = unload right after.
    pub keep_alive: Option<Duration>,
}

impl Daemon {
    pub fn new(cfg: Config) -> Result<Arc<Daemon>, String> {
        let stores = stores::Stores::open(&cfg.stores)?;
        let hub = aie_store::hf::Hub::public().map_err(|e| e.to_string())?;
        std::fs::create_dir_all(cfg.home.join("placements")).map_err(|e| e.to_string())?;
        Ok(Arc::new(Daemon {
            workers: Manager::new(cfg.clone()),
            memory: memory::Sampler::default(),
            cfg,
            stores,
            hub,
            pulling: tokio::sync::Mutex::new(()),
        }))
    }

    /// Find a pulled model by the name a client used. A tagless `hf.co/o/r`
    /// picks the only pulled tag, or Q4_K_M among several.
    pub fn resolve(&self, name: &str) -> Result<Resolved, ApiError> {
        let r = ModelRef::parse(name).map_err(|e| ApiError::bad_request(e.to_string()))?;
        match r {
            ModelRef::Local(p) => {
                if !p.exists() {
                    return Err(ApiError::not_found(format!(
                        "{} doesn't exist",
                        p.display()
                    )));
                }
                Ok(Resolved {
                    name: p.display().to_string(),
                    path: p,
                    manifest: None,
                })
            }
            ModelRef::Oci(o) => {
                let m = self
                    .stores
                    .get(&ModelRef::Oci(o))
                    .map_err(|e| ApiError::internal(e.to_string()))?
                    .ok_or_else(|| {
                        ApiError::not_found(format!("model {name} not found; pull it first"))
                    })?;
                Ok(Resolved {
                    name: m.manifest.name.clone(),
                    path: m.path,
                    manifest: Some(m.manifest),
                })
            }
            ModelRef::Hf(h) => {
                let found = if h.tag.is_some() {
                    self.stores
                        .get(&ModelRef::Hf(h.clone()))
                        .map_err(|e| ApiError::internal(e.to_string()))?
                } else {
                    let all = self
                        .stores
                        .list()
                        .map_err(|e| ApiError::internal(e.to_string()))?;
                    let mut same: Vec<_> = all
                        .into_iter()
                        .filter(|m| matches!(&m.manifest.source, aie_store::Source::Hf { repo, .. } if repo.eq_ignore_ascii_case(&h.repo)))
                        .collect();
                    if same.len() > 1 {
                        let def = same.iter().position(|m| {
                            m.manifest.name.to_ascii_uppercase().ends_with(":Q4_K_M")
                        });
                        match def {
                            Some(i) => Some(same.swap_remove(i)),
                            None => {
                                let names: Vec<String> =
                                    same.iter().map(|m| m.manifest.name.clone()).collect();
                                return Err(ApiError::bad_request(format!(
                                    "{name} is ambiguous; pulled: {}",
                                    names.join(", ")
                                )));
                            }
                        }
                    } else {
                        same.pop()
                    }
                };
                let m = found.ok_or_else(|| {
                    ApiError::not_found(format!("model {name} not found; pull it first"))
                })?;
                Ok(Resolved {
                    name: m.manifest.name.clone(),
                    path: m.path,
                    manifest: Some(m.manifest),
                })
            }
        }
    }

    fn placement_path(&self, name: &str) -> PathBuf {
        let safe: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.cfg
            .home
            .join("placements")
            .join(format!("{safe}.json"))
    }

    /// The saved placement intent for a model, if any.
    pub fn saved_intent(&self, name: &str) -> Result<Option<Intent>, ApiError> {
        let p = self.placement_path(name);
        if !p.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&p).map_err(|e| ApiError::internal(e.to_string()))?;
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| ApiError::internal(format!("{}: {e}", p.display())))
    }

    /// Save (or with `None`, forget) a model's placement intent. Applies at next load.
    pub fn save_intent(&self, name: &str, intent: Option<&Intent>) -> Result<(), ApiError> {
        let p = self.placement_path(name);
        match intent {
            Some(i) => std::fs::write(&p, serde_json::to_vec_pretty(i).unwrap())
                .map_err(|e| ApiError::internal(e.to_string())),
            None => match std::fs::remove_file(&p) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                    Err(ApiError::internal(e.to_string()))
                }
                _ => Ok(()),
            },
        }
    }

    /// The load spec for a model: saved intent, else llama.cpp's own fit.
    pub fn load_spec(&self, r: &Resolved, n_ctx: Option<u32>) -> Result<LoadSpec, ApiError> {
        // The request's context wins; else the one saved with the placement
        // (set in the UI); else the config's default.
        let saved = self.saved_intent(&r.name)?;
        let n_ctx = n_ctx
            .or_else(|| {
                saved
                    .as_ref()
                    .and_then(|i| u32::try_from(i.n_ctx).ok())
                    .filter(|n| *n > 0)
            })
            .unwrap_or(self.cfg.n_ctx);
        let placement = match saved {
            Some(mut intent) => {
                intent.n_ctx = u64::from(n_ctx);
                Placement::Intent { intent }
            }
            None => Placement::Auto {
                n_ctx_min: 512.min(n_ctx),
                margin: 512 << 20,
            },
        };
        Ok(LoadSpec {
            name: r.name.clone(),
            path: r.path.clone(),
            placement,
            n_ctx,
        })
    }

    /// Load (if needed) and stream a generation. The worker stays leased
    /// until the stream is dropped.
    pub async fn generate(
        &self,
        model: &str,
        input: Input,
        o: GenOptions,
    ) -> Result<
        (
            Resolved,
            impl Stream<Item = Result<GenerateEvent, ApiError>> + Send + 'static,
        ),
        ApiError,
    > {
        let r = self.resolve(model)?;
        let spec = self.load_spec(&r, o.n_ctx)?;
        let lease = self
            .workers
            .lease(&spec, o.keep_alive)
            .await
            .map_err(ApiError::internal)?;
        let mut client = lease.client();
        let req = GenerateRequest {
            input: Some(input),
            sampling: Some(o.sampling),
            max_tokens: o.max_tokens,
            stop: o.stop,
        };
        let stream = client
            .generate(req)
            .await
            .map_err(|s| ApiError::internal(s.message().to_string()))?
            .into_inner();
        let unload_after = o.keep_alive == Some(Duration::ZERO);
        let workers = self.workers.clone();
        let name = r.name.clone();
        let guarded = LeasedStream {
            inner: stream,
            _lease: Some(lease),
            on_end: unload_after.then_some((workers, name)),
        };
        Ok((
            r,
            guarded.map(|item| item.map_err(|s| ApiError::internal(s.message().to_string()))),
        ))
    }
}

/// A worker's event stream that keeps the worker leased until it ends.
struct LeasedStream {
    inner: tonic::Streaming<GenerateEvent>,
    _lease: Option<Lease>,
    on_end: Option<(Arc<Manager>, String)>,
}

impl LeasedStream {
    /// Release the worker, and unload it if the request asked (keep_alive 0).
    fn finish(&mut self) {
        self._lease = None;
        if let Some((m, n)) = self.on_end.take()
            && let Ok(rt) = tokio::runtime::Handle::try_current()
        {
            rt.spawn(async move {
                m.unload(&n).await;
            });
        }
    }
}

impl Stream for LeasedStream {
    type Item = Result<GenerateEvent, tonic::Status>;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let r = std::pin::Pin::new(&mut self.inner).poll_next(cx);
        if let std::task::Poll::Ready(None) = r {
            self.finish();
        }
        r
    }
}

/// Non-streaming handlers stop reading at the final event and drop the
/// stream rather than draining it; finish then too.
impl Drop for LeasedStream {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Serve until Ctrl-C / SIGTERM, then stop every worker.
pub async fn serve(cfg: Config) -> Result<(), String> {
    let listen = cfg.listen;
    let d = Daemon::new(cfg)?;
    let reaper = d.workers.clone();
    let sampler = d.clone();
    tokio::spawn(async move {
        loop {
            sampler.memory.sample(&sampler.workers).await;
            tokio::time::sleep(memory::INTERVAL).await;
        }
    });
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            reaper.reap().await;
        }
    });
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .map_err(|e| format!("{listen}: {e}"))?;
    eprintln!(
        "workers start {}",
        if d.workers.swap_free_scope() {
            "with swap forbidden (systemd user scope)"
        } else {
            "normally (RAM tier relies on mlock or no swap)"
        }
    );
    eprintln!(
        "ai-engine listening on http://{listen} (store: {})",
        d.cfg
            .stores
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let workers = d.workers.clone();
    axum::serve(listener, api::router(d))
        .with_graceful_shutdown(async {
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("signal");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
        })
        .await
        .map_err(|e| e.to_string())?;
    workers.shutdown().await;
    Ok(())
}

/// RFC 3339 UTC timestamp with microseconds, as ollama writes them.
pub fn rfc3339(t: std::time::SystemTime) -> String {
    let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let secs = d.as_secs() as i64;
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60,
        d.subsec_micros()
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn timestamps() {
        let t = std::time::UNIX_EPOCH + std::time::Duration::from_micros(1_700_000_000_123_456);
        assert_eq!(super::rfc3339(t), "2023-11-14T22:13:20.123456Z");
        assert_eq!(
            super::rfc3339(std::time::UNIX_EPOCH),
            "1970-01-01T00:00:00.000000Z"
        );
    }
}
