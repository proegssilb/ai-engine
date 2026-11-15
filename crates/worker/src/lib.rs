//! The worker: one process per loaded model (decision D3, architecture.md).
//!
//! It loads the model with a compiled placement, checks where every weight
//! really is against the plan, then serves the [`aie_proto::worker`] gRPC
//! service on a Unix socket. Requests run one at a time (an MVP-only limit;
//! the protocol already allows more in flight).

pub mod numa;
pub mod ram_tier;
pub mod stop;

pub use aie_llama::ContextParams;
pub use aie_llama::context::{Stop, Utf8Stream};
pub use aie_llama::pinned::PinOptions;
use aie_llama::verify;
use aie_llama::{Context, LoadOptions, Model, Sampling};
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
    /// A saved intent ("experts: 10% GPU 1, 80% RAM, rest on disk"), planned
    /// here because planning needs the devices. Refused, with the reason, if
    /// it doesn't fit.
    Intent { intent: aie_plan::Intent },
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

/// What `ai-engine probe` prints: the devices as llama.cpp sees them, host RAM
/// and its NUMA nodes. Runs in a short-lived process so the daemon never
/// initialises CUDA.
pub fn probe() -> serde_json::Value {
    let devices: Vec<serde_json::Value> = aie_llama::devices()
        .into_iter()
        .map(|d| {
            let node = d.pci_bus_id.as_deref().map(|p| numa::pci_node(Path::new("/"), p));
            serde_json::json!({ "name": d.name, "description": d.description, "total": d.total, "free": d.free, "gpu": d.is_gpu, "pci_bus_id": d.pci_bus_id, "numa_node": node })
        })
        .collect();
    let m = machine();
    let root = Path::new("/");
    let topo = numa::Topology::read(root);
    let numa: Vec<serde_json::Value> = topo
        .nodes
        .iter()
        .map(|n| serde_json::json!({ "node": n.id, "total": n.total, "free": n.free, "cpus": n.cpulist }))
        .collect();
    serde_json::json!({
        "devices": devices,
        "ram": { "total": m.ram.total, "available": m.ram.free },
        "numa": numa,
        "cuda_build": aie_llama::CUDA,
    })
}

/// GPUs as llama.cpp sees them, and host RAM from /proc/meminfo.
pub fn machine() -> aie_plan::fit::Machine {
    let gpus = aie_llama::devices()
        .into_iter()
        .filter(|d| d.is_gpu)
        .map(|d| aie_plan::fit::Capacity {
            name: d.name,
            total: d.total,
            free: d.free,
        })
        .collect();
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let kib = |key: &str| {
        meminfo
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024
    };
    aie_plan::fit::Machine {
        gpus,
        ram: aie_plan::fit::Capacity {
            name: "RAM".into(),
            total: kib("MemTotal:"),
            free: kib("MemAvailable:"),
        },
    }
}

/// Resolve the placement mode into a plan and load settings.
fn decide(cfg: &Config, anatomy: &aie_anatomy::Anatomy) -> Result<(Plan, Settings), String> {
    match &cfg.placement {
        Placement::Planned { plan, settings } => Ok((plan.clone(), settings.clone())),
        Placement::Intent { intent } => {
            let machine = machine();
            let gpus = machine.gpus.len() as u32;
            let plan = aie_plan::plan(anatomy, intent, gpus).map_err(|e| e.to_string())?;
            let report = aie_plan::fit::check(
                anatomy,
                &plan,
                &machine,
                &aie_plan::fit::Overheads::default(),
            );
            if !report.fits() {
                return Err(format!("this placement doesn't fit:\n{report}"));
            }
            let settings =
                aie_plan::llama::compile(anatomy, &plan, gpus).map_err(|e| e.to_string())?;
            Ok((plan, settings))
        }
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
                host_node: None,
            };
            Ok((plan, s))
        }
    }
}

struct Loaded {
    ctx: Arc<Mutex<Context>>,
    model: Arc<Model>,
    status: pb::StatusReply,
    /// File ranges (offset, length) of weights on the disk tier.
    disk_ranges: Vec<(u64, u64)>,
    stats: std::sync::Mutex<Stats>,
    /// Where the host side was pinned (see [`numa`]), and the machine's sockets.
    host: numa::Applied,
    topo: numa::Topology,
    threads: i32,
}

#[derive(Default)]
struct Stats {
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    prefill_ns: u64,
    decode_ns: u64,
    last_prefill_tps: f32,
    last_decode_tps: f32,
}

/// Problems with the RAM tier's buffers that don't stop the load.
fn load_warnings() -> Vec<String> {
    let mut out = Vec::new();
    for b in aie_llama::pinned::REGISTRY
        .lock()
        .unwrap()
        .iter()
        .filter(|b| !b.freed)
    {
        if let Some(Err(e)) = b.hugetlb_result {
            out.push(format!(
                "the RAM tier asked for reserved huge pages but got normal ones (mmap errno {e}); reserve more (vm.nr_hugepages / /sys/kernel/mm/hugepages)"
            ));
        }
        if let Some(Err(e)) = b.mbind_result {
            out.push(format!(
                "the RAM tier couldn't be bound to its NUMA node (mbind errno {e})"
            ));
        }
    }
    out.dedup();
    out
}

/// Load the model and check its placement. Blocking; run before serving.
fn load(cfg: &Config) -> Result<Loaded, String> {
    let gguf = aie_gguf::Gguf::open(&cfg.model).map_err(|e| e.to_string())?;
    let anatomy = aie_anatomy::Anatomy::from_gguf(&gguf).map_err(|e| e.to_string())?;
    // Refuse a socket that doesn't exist before doing anything expensive.
    let root = Path::new("/");
    let topo = numa::Topology::read(root);
    let requested = match &cfg.placement {
        Placement::Intent { intent } => intent.host_node,
        Placement::Planned { plan, .. } => plan.host_node,
        Placement::Auto { .. } => None,
    };
    numa::choose(&topo, requested, &[], &[])?;
    let (plan, settings) = decide(cfg, &anatomy)?;

    // Pin the host side to one socket before llama.cpp allocates or starts
    // threads: next to the GPU holding the layers, unless chosen.
    let gpu_nodes: Vec<u32> = aie_llama::devices()
        .iter()
        .filter(|d| d.is_gpu)
        .map(|d| {
            d.pci_bus_id
                .as_deref()
                .map_or(0, |p| numa::pci_node(root, p))
        })
        .collect();
    let choice = numa::choose(&topo, plan.host_node, &gpu_nodes, &plan.layer_home)?;
    let host = numa::apply(&topo, &choice)?;
    let mut context = cfg.context.clone();
    if context.threads <= 0 {
        context.threads = numa::default_threads(&host);
    }
    let threads = context.threads;

    // Driver features are used when they're there (spikes/memory): CUDA
    // registration only with a CUDA build and a GPU to copy to.
    let mut pin = cfg.pin.clone();
    pin.cuda_register &= aie_llama::CUDA && aie_llama::devices().iter().any(|d| d.is_gpu);
    // The RAM tier binds strictly to the node. On one socket there's nothing
    // to bind to, so it stays as it always was.
    if topo.multi() {
        pin.numa_node = Some(host.node);
    }
    let opts = LoadOptions { settings, pin };
    // llama.cpp's own dry run: the authoritative check before allocating.
    let measured =
        aie_llama::fit::measure(&cfg.model, &opts, &context).map_err(|e| e.to_string())?;
    if matches!(cfg.placement, Placement::Intent { .. }) {
        let short: Vec<String> = measured
            .iter()
            .filter(|m| m.kind == "device" && m.total > 0 && m.need() > m.free)
            .map(|m| {
                format!(
                    "{} needs {} MiB, has {} MiB free",
                    m.name,
                    m.need() >> 20,
                    m.free >> 20
                )
            })
            .collect();
        if !short.is_empty() {
            return Err(format!(
                "llama.cpp's dry run says this placement doesn't fit: {}",
                short.join("; ")
            ));
        }
    }
    let memory_json = serde_json::to_string(&measured).map_err(|e| e.to_string())?;

    let t0 = Instant::now();
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
        &context,
        &prompt,
        anatomy.parts.iter().map(|p| p.name.clone()),
    )
    .map_err(|e| e.to_string())?;
    let misplaced = report.compare(&plan).len() as u32;
    let placement_json = serde_json::to_string(&report).map_err(|e| e.to_string())?;
    let plan_json = serde_json::to_string(&plan).map_err(|e| e.to_string())?;

    let disk_ranges: Vec<(u64, u64)> = gguf
        .tensors
        .iter()
        .filter(|t| plan.tensors.get(&t.name) == Some(&aie_plan::Tier::Disk))
        .map(|t| (gguf.file_offset(t), t.size))
        .collect();
    let disk_tier_bytes = disk_ranges.iter().map(|r| r.1).sum();
    let mut warnings = load_warnings();

    let ctx = Context::new(model.clone(), context.clone()).map_err(|e| e.to_string())?;
    // Measure where things landed; on several sockets, say what isn't where promised.
    let hr = numa::report(&topo, &host, threads);
    for p in &hr.problems {
        if !warnings.contains(p) {
            warnings.push(p.clone());
        }
    }
    let status = pb::StatusReply {
        model_path: cfg.model.display().to_string(),
        description: model.description(),
        n_ctx: ctx.n_ctx(),
        weight_bytes: model.size(),
        placement_json,
        misplaced,
        load_ms,
        plan_json,
        disk_tier_bytes,
        warnings,
        memory_json,
        ..Default::default()
    };
    Ok(Loaded {
        ctx: Arc::new(Mutex::new(ctx)),
        model,
        status,
        disk_ranges,
        stats: Default::default(),
        host,
        topo,
        threads,
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
        let mut st = self.0.status.clone();
        {
            let s = self.0.stats.lock().unwrap();
            st.requests = s.requests;
            st.prompt_tokens = s.prompt_tokens;
            st.completion_tokens = s.completion_tokens;
            st.prefill_ns = s.prefill_ns;
            st.decode_ns = s.decode_ns;
            st.last_prefill_tps = s.last_prefill_tps;
            st.last_decode_tps = s.last_decode_tps;
        }
        st.ram_tier_json = serde_json::to_string(&ram_tier::status()).unwrap_or_default();
        st.host_json = numa::report_json(&self.0.topo, &self.0.host, self.0.threads);
        st.disk_tier_resident =
            aie_llama::residency::resident_file_bytes(self.0.model.path(), &self.0.disk_ranges);
        Ok(Response::new(st))
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
            let loaded2 = loaded.clone();
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
                {
                    let mut s = loaded2.stats.lock().unwrap();
                    s.requests += 1;
                    s.prompt_tokens += g.prompt_tokens as u64;
                    s.completion_tokens += g.tokens.len() as u64;
                    s.prefill_ns += g.prefill.as_nanos() as u64;
                    s.decode_ns += g.decode.as_nanos() as u64;
                    s.last_prefill_tps = g.prefill_tps() as f32;
                    s.last_decode_tps = g.decode_tps() as f32;
                }
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
