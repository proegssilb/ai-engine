//! The worker: one process per loaded model (decision D3, architecture.md).
//!
//! It loads the model with a compiled placement, checks where every weight
//! really is against the plan, then serves the [`aie_proto::worker`] gRPC
//! service on a Unix socket. Requests run one at a time (an MVP-only limit;
//! the protocol already allows more in flight).

pub mod stop;

use aie_llama::context::{Stop, Utf8Stream};
use aie_llama::pinned::PinOptions;
use aie_llama::verify;
use aie_llama::{Context, ContextParams, LoadOptions, Model, Sampling};
use aie_plan::Plan;
use aie_plan::llama::Settings;
use aie_proto::worker::generate_event::Event;
use aie_proto::worker::generate_request::Input;
use aie_proto::worker::worker_server::{Worker, WorkerServer};
use aie_proto::worker::{self as pb, done::Reason};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::{Request, Response, Status};

/// How the worker decides where weights go.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Placement {
    /// llama.cpp's own fit to the memory free right now (the MVP's auto-place),
    /// resolved into a plan so it's shown and checked like any other. Runs in
    /// the worker so the daemon never initialises CUDA.
    Auto {
        /// Smallest context the fit may shrink to when the requested one doesn't fit.
        n_ctx_min: u32,
        /// Bytes to leave free on each GPU.
        margin: u64,
    },
    /// A plan the daemon made and compiled.
    Planned { plan: Plan, settings: Settings },
}

/// Everything a worker needs, passed as JSON by the daemon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub model: PathBuf,
    pub placement: Placement,
    /// How the RAM tier's pinned buffer behaves.
    pub pin: PinOptions,
    pub context: ContextParams,
}

/// Resolve the placement mode into a plan and load settings.
fn decide(cfg: &Config, anatomy: &aie_anatomy::Anatomy) -> Result<(Plan, Settings), String> {
    match &cfg.placement {
        Placement::Planned { plan, settings } => Ok((plan.clone(), settings.clone())),
        Placement::Auto { n_ctx_min, margin } => {
            let fitted = aie_llama::fit::fit(&cfg.model, cfg.context.n_ctx, *n_ctx_min, *margin)
                .map_err(|e| e.to_string())?;
            let s = fitted.settings;
            let tensors = aie_plan::llama::resolve(anatomy, &s).map_err(|e| e.to_string())?;
            let plan = Plan {
                layer_home: aie_plan::llama::resolve_layer_homes(anatomy, &s),
                tensors,
                n_ctx: s.n_ctx,
                expert_cache_bytes: 0,
                allow_repack: false,
            };
            Ok((plan, s))
        }
    }
}

struct Loaded {
    ctx: Arc<Mutex<Context>>,
    model: Arc<Model>,
    status: pb::StatusReply,
}

/// Load the model and check its placement. Blocking; run before serving.
fn load(cfg: &Config) -> Result<Loaded, String> {
    let gguf = aie_gguf::Gguf::open(&cfg.model).map_err(|e| e.to_string())?;
    let anatomy = aie_anatomy::Anatomy::from_gguf(&gguf).map_err(|e| e.to_string())?;
    let (plan, settings) = decide(cfg, &anatomy)?;

    let t0 = Instant::now();
    let opts = LoadOptions {
        settings,
        pin: cfg.pin.clone(),
    };
    let model = Arc::new(Model::load(&cfg.model, &opts).map_err(|e| e.to_string())?);
    let load_ms = t0.elapsed().as_millis() as u64;

    let prompt = model
        .tokenize(
            &"The placement check reads every weight once. ".repeat(4),
            true,
        )
        .map_err(|e| e.to_string())?;
    let report = verify::check(
        &model,
        &cfg.context,
        &prompt,
        anatomy.parts.iter().map(|p| p.name.clone()),
    )
    .map_err(|e| e.to_string())?;
    let misplaced = report.compare(&plan).len() as u32;
    let placement_json = serde_json::to_string(&report).map_err(|e| e.to_string())?;
    let plan_json = serde_json::to_string(&plan).map_err(|e| e.to_string())?;

    let ctx = Context::new(model.clone(), cfg.context.clone()).map_err(|e| e.to_string())?;
    let status = pb::StatusReply {
        model_path: cfg.model.display().to_string(),
        description: model.description(),
        n_ctx: ctx.n_ctx(),
        weight_bytes: model.size(),
        placement_json,
        misplaced,
        load_ms,
        plan_json,
    };
    Ok(Loaded {
        ctx: Arc::new(Mutex::new(ctx)),
        model,
        status,
    })
}

#[derive(Clone)]
struct Service(Arc<Loaded>);

fn sampling(s: Option<pb::Sampling>) -> Sampling {
    match s {
        None => Sampling::default(),
        Some(s) => Sampling {
            temperature: s.temperature,
            top_k: s.top_k,
            top_p: s.top_p,
            min_p: s.min_p,
            seed: s.seed,
        },
    }
}

#[tonic::async_trait]
impl Worker for Service {
    async fn status(
        &self,
        _: Request<pb::StatusRequest>,
    ) -> Result<Response<pb::StatusReply>, Status> {
        Ok(Response::new(self.0.status.clone()))
    }

    async fn tokenize(
        &self,
        r: Request<pb::TokenizeRequest>,
    ) -> Result<Response<pb::TokenizeReply>, Status> {
        let tokens = self
            .0
            .model
            .tokenize(&r.into_inner().text, true)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        Ok(Response::new(pb::TokenizeReply { tokens }))
    }

    type GenerateStream = ReceiverStream<Result<pb::GenerateEvent, Status>>;

    async fn generate(
        &self,
        r: Request<pb::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let req = r.into_inner();
        let model = self.0.model.clone();
        let text = match req.input {
            Some(Input::Prompt(p)) => p,
            Some(Input::Chat(c)) => {
                let msgs: Vec<(String, String)> = c
                    .messages
                    .into_iter()
                    .map(|m| (m.role, m.content))
                    .collect();
                model
                    .apply_chat_template(&msgs, true)
                    .map_err(|e| Status::invalid_argument(e.to_string()))?
            }
            None => return Err(Status::invalid_argument("no prompt or chat given")),
        };
        let prompt = model
            .tokenize(&text, true)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;
        let max = if req.max_tokens == 0 {
            u32::MAX as usize
        } else {
            req.max_tokens as usize
        };
        let sampling = sampling(req.sampling);
        let stops = req.stop;

        let (tx, rx) = mpsc::channel(64);
        let loaded = self.0.clone();
        tokio::spawn(async move {
            // One request at a time: wait for the context.
            let mut ctx = loaded.ctx.clone().lock_owned().await;
            let tx2 = tx.clone();
            let result = tokio::task::spawn_blocking(move || {
                let mut utf8 = Utf8Stream::default();
                let mut scan = stop::StopScanner::new(stops);
                let mut hit_stop = false;
                let mut cancelled = false;
                let g = ctx.generate(&prompt, max, &sampling, |t| {
                    let (out, hit) = scan.push(&utf8.push(&model.piece(t)));
                    if !out.is_empty()
                        && tx2
                            .blocking_send(Ok(pb::GenerateEvent {
                                event: Some(Event::Text(out)),
                            }))
                            .is_err()
                    {
                        cancelled = true;
                        return false;
                    }
                    hit_stop = hit;
                    !hit
                })?;
                if !hit_stop && !cancelled {
                    let (rest, _) = scan.push(&utf8.finish());
                    let rest = rest + &scan.finish();
                    if !rest.is_empty() {
                        let _ = tx2.blocking_send(Ok(pb::GenerateEvent {
                            event: Some(Event::Text(rest)),
                        }));
                    }
                }
                let reason = match (g.stop, hit_stop, cancelled) {
                    (_, _, true) => Reason::Cancelled,
                    (_, true, _) => Reason::StopString,
                    (Stop::EndOfText, ..) => Reason::EndOfText,
                    (Stop::Length, ..) => Reason::Length,
                    (Stop::ContextFull, ..) => Reason::ContextFull,
                    (Stop::Caller, ..) => Reason::Cancelled,
                };
                Ok::<_, aie_llama::Error>(pb::Done {
                    reason: reason as i32,
                    prompt_tokens: g.prompt_tokens as u32,
                    completion_tokens: g.tokens.len() as u32,
                    prefill_ns: g.prefill.as_nanos() as u64,
                    decode_ns: g.decode.as_nanos() as u64,
                })
            })
            .await;
            let last = match result {
                Ok(Ok(done)) => Ok(pb::GenerateEvent {
                    event: Some(Event::Done(done)),
                }),
                Ok(Err(e)) => Err(Status::internal(e.to_string())),
                Err(e) => Err(Status::internal(format!("generation panicked: {e}"))),
            };
            let _ = tx.send(last).await;
        });
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

/// Load, then serve on `socket` until the process is killed.
pub async fn run(cfg: Config, socket: &Path) -> Result<(), String> {
    let loaded = tokio::task::spawn_blocking(move || load(&cfg))
        .await
        .map_err(|e| e.to_string())??;
    let _ = std::fs::remove_file(socket);
    let listener =
        tokio::net::UnixListener::bind(socket).map_err(|e| format!("{}: {e}", socket.display()))?;
    tonic::transport::Server::builder()
        .add_service(WorkerServer::new(Service(Arc::new(loaded))))
        .serve_with_incoming(UnixListenerStream::new(listener))
        .await
        .map_err(|e| e.to_string())
}

/// Connect to a worker's socket.
pub async fn connect(
    socket: &Path,
) -> Result<pb::worker_client::WorkerClient<tonic::transport::Channel>, tonic::transport::Error> {
    let path = socket.to_path_buf();
    // The URI is required by tonic but unused: the connector dials the socket.
    let channel = tonic::transport::Endpoint::from_static("http://worker.local")
        .connect_with_connector(tower::service_fn(move |_| {
            let path = path.clone();
            async move {
                Ok::<_, std::io::Error>(hyper_util::rt::TokioIo::new(
                    tokio::net::UnixStream::connect(path).await?,
                ))
            }
        }))
        .await?;
    Ok(pb::worker_client::WorkerClient::new(channel))
}
