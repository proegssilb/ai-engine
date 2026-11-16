//! Worker lifecycle: one process per loaded model (D3). Load = spawn the
//! product binary as `worker`; unload = stop it, which returns all its memory.

use crate::config::Config;
use aie_proto::worker::worker_client::WorkerClient;
use aie_proto::worker::{StatusReply, StatusRequest};
use aie_worker::Placement;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant, SystemTime};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tonic::transport::Channel;

/// A running worker.
pub struct Worker {
    /// The model reference it serves, as users name it.
    pub name: String,
    pub model_path: PathBuf,
    pub status: StatusReply,
    pub loaded_at: SystemTime,
    pub n_ctx: u32,
    socket: PathBuf,
    child: Mutex<Child>,
    client: WorkerClient<Channel>,
    last_used: StdMutex<Instant>,
    /// `None` = stay loaded.
    keep_alive: StdMutex<Option<Duration>>,
    active: AtomicUsize,
}

impl Worker {
    pub fn client(&self) -> WorkerClient<Channel> {
        self.client.clone()
    }

    /// The worker process's id, while it runs.
    pub async fn pid(&self) -> Option<u32> {
        self.child.lock().await.id()
    }

    /// When an idle worker will be unloaded, if ever.
    pub fn expires_at(&self) -> Option<SystemTime> {
        let ka = (*self.keep_alive.lock().unwrap())?;
        let idle = self.last_used.lock().unwrap().elapsed();
        Some(SystemTime::now() + ka.saturating_sub(idle))
    }

    pub fn is_busy(&self) -> bool {
        self.active.load(Ordering::SeqCst) > 0
    }
}

/// Holds a worker busy for the length of a request; on drop it's marked used.
pub struct Lease(pub Arc<Worker>);

impl Drop for Lease {
    fn drop(&mut self) {
        *self.0.last_used.lock().unwrap() = Instant::now();
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl std::ops::Deref for Lease {
    type Target = Worker;
    fn deref(&self) -> &Worker {
        &self.0
    }
}

/// What to load.
#[derive(Clone, Debug)]
pub struct LoadSpec {
    pub name: String,
    pub path: PathBuf,
    pub placement: Placement,
    pub n_ctx: u32,
}

pub struct Manager {
    cfg: Config,
    workers: Mutex<HashMap<String, Arc<Worker>>>,
    /// Loads happen one at a time: each needs to see the memory the previous one took.
    loading: Mutex<()>,
    seq: AtomicU64,
    /// Start workers in a systemd user scope with swap forbidden: holds the
    /// RAM tier without root, mlock or a driver (spikes/memory, phase C).
    swap_free_scope: bool,
}

/// Can workers be started with swap forbidden from here? Only outside a
/// system service (the unit handles that) and with a systemd user manager
/// that controls memory; one trial run decides.
fn swap_free_scope_available() -> bool {
    let cg = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    if cg.contains("/system.slice/") {
        return false;
    }
    std::process::Command::new("systemd-run")
        .args([
            "--user",
            "--scope",
            "-q",
            "--collect",
            "-p",
            "MemorySwapMax=0",
            "true",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl Manager {
    pub fn new(cfg: Config) -> Arc<Manager> {
        let swap_free_scope = cfg.forbid_worker_swap && swap_free_scope_available();
        Arc::new(Manager {
            cfg,
            workers: Mutex::new(HashMap::new()),
            loading: Mutex::new(()),
            seq: AtomicU64::new(0),
            swap_free_scope,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Whether workers start with swap forbidden (systemd user scope).
    pub fn swap_free_scope(&self) -> bool {
        self.swap_free_scope
    }

    /// A worker for the model, loading it if needed. `keep_alive` (`None` =
    /// forever) applies from the end of the request.
    pub async fn lease(
        &self,
        spec: &LoadSpec,
        keep_alive: Option<Duration>,
    ) -> Result<Lease, String> {
        let w = self.get_or_load(spec).await?;
        w.active.fetch_add(1, Ordering::SeqCst);
        *w.keep_alive.lock().unwrap() = keep_alive;
        *w.last_used.lock().unwrap() = Instant::now();
        Ok(Lease(w))
    }

    async fn alive(&self, name: &str) -> Option<Arc<Worker>> {
        let w = self.workers.lock().await.get(name).cloned()?;
        let exited = w.child.lock().await.try_wait().ok().flatten().is_some();
        if exited {
            self.workers.lock().await.remove(name);
            return None;
        }
        Some(w)
    }

    async fn get_or_load(&self, spec: &LoadSpec) -> Result<Arc<Worker>, String> {
        if let Some(w) = self.alive(&spec.name).await
            && w.n_ctx >= spec.n_ctx
        {
            return Ok(w);
        }
        let _one = self.loading.lock().await;
        if let Some(w) = self.alive(&spec.name).await {
            if w.n_ctx >= spec.n_ctx {
                return Ok(w);
            }
            // a bigger context was asked for: reload
            self.unload(&spec.name).await;
        }
        match self.spawn(spec).await {
            Ok(w) => Ok(w),
            Err(first) => {
                // Out of memory is the likeliest cause: free idle models and retry once.
                if self.unload_idle().await == 0 {
                    return Err(first);
                }
                self.spawn(spec).await.map_err(|e| {
                    format!("{e}\n(retried after unloading idle models; first attempt: {first})")
                })
            }
        }
    }

    fn worker_config(&self, spec: &LoadSpec) -> aie_worker::Config {
        aie_worker::Config {
            model: spec.path.clone(),
            placement: spec.placement.clone(),
            pin: aie_llama_pin_defaults(),
            context: aie_worker::ContextParams {
                n_ctx: spec.n_ctx,
                n_batch: self.cfg.n_batch,
                // 0 = the worker uses one thread per CPU of its socket.
                threads: self.cfg.threads.max(0),
                ..Default::default()
            },
        }
    }

    /// What loading `spec` would need, from llama.cpp's dry run in a short-lived
    /// process (the daemon doesn't link a GPU backend's state). Loads nothing.
    pub async fn dry_run(&self, spec: &LoadSpec) -> Result<aie_worker::DryRun, String> {
        let run = self.cfg.home.join("run");
        tokio::fs::create_dir_all(&run)
            .await
            .map_err(|e| format!("{}: {e}", run.display()))?;
        let id = self.seq.fetch_add(1, Ordering::SeqCst);
        let cfg_path = run.join(format!(
            "{}-{}-dry{id}.json",
            sanitize(&spec.name),
            std::process::id()
        ));
        let json =
            serde_json::to_vec_pretty(&self.worker_config(spec)).map_err(|e| e.to_string())?;
        tokio::fs::write(&cfg_path, json)
            .await
            .map_err(|e| e.to_string())?;
        let out = Command::new(&self.cfg.worker_exe)
            .arg("dry-run")
            .arg("--config")
            .arg(&cfg_path)
            .output()
            .await;
        let _ = tokio::fs::remove_file(&cfg_path).await;
        let out = out.map_err(|e| format!("starting {}: {e}", self.cfg.worker_exe.display()))?;
        if !out.status.success() {
            return Err(error_reason(&String::from_utf8_lossy(&out.stderr)));
        }
        serde_json::from_slice(&out.stdout).map_err(|e| format!("dry run output: {e}"))
    }

    async fn spawn(&self, spec: &LoadSpec) -> Result<Arc<Worker>, String> {
        let run = self.cfg.home.join("run");
        let logs = self.cfg.home.join("logs");
        let sockets = socket_dir();
        for d in [&run, &logs, &sockets] {
            tokio::fs::create_dir_all(d)
                .await
                .map_err(|e| format!("{}: {e}", d.display()))?;
        }
        let id = format!(
            "{}-{}",
            std::process::id(),
            self.seq.fetch_add(1, Ordering::SeqCst)
        );
        // Unix socket paths are limited to ~108 bytes, so sockets live in a
        // short runtime directory rather than under AIE_HOME.
        let socket = sockets.join(format!("w{id}.sock"));
        if socket.as_os_str().len() >= 104 {
            return Err(format!(
                "socket path {} is too long for a Unix socket; set XDG_RUNTIME_DIR to a short directory",
                socket.display()
            ));
        }
        let cfg_path = run.join(format!("{}-{id}.json", sanitize(&spec.name)));
        let log_path = logs.join(format!("{}.log", sanitize(&spec.name)));
        let wcfg = self.worker_config(spec);
        let json = serde_json::to_vec_pretty(&wcfg).map_err(|e| e.to_string())?;
        tokio::fs::write(&cfg_path, json)
            .await
            .map_err(|e| e.to_string())?;
        let log =
            std::fs::File::create(&log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;

        let mut cmd = if self.swap_free_scope {
            // `systemd-run --scope` execs the worker: same PID, own cgroup.
            let mut c = Command::new("systemd-run");
            c.args([
                "--user",
                "--scope",
                "-q",
                "--collect",
                "-p",
                "MemorySwapMax=0",
            ])
            .arg(&self.cfg.worker_exe);
            c
        } else {
            Command::new(&self.cfg.worker_exe)
        };
        let mut child = cmd
            .arg("worker")
            .arg("--config")
            .arg(&cfg_path)
            .arg("--socket")
            .arg(&socket)
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("starting {}: {e}", self.cfg.worker_exe.display()))?;

        let deadline = Instant::now() + self.cfg.load_timeout;
        let (client, status) = loop {
            if let Some(code) = child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!(
                    "loading {} failed ({code}):\n{}",
                    spec.name,
                    tail(&log_path, 12)
                ));
            }
            if Instant::now() > deadline {
                let _ = child.kill().await;
                return Err(format!(
                    "loading {} timed out after {:?}",
                    spec.name, self.cfg.load_timeout
                ));
            }
            if socket.exists()
                && let Ok(mut c) = aie_worker::connect(&socket).await
                && let Ok(st) = c.status(StatusRequest {}).await
            {
                break (c, st.into_inner());
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        };
        let w = Arc::new(Worker {
            name: spec.name.clone(),
            model_path: spec.path.clone(),
            n_ctx: status.n_ctx,
            status,
            loaded_at: SystemTime::now(),
            socket,
            child: Mutex::new(child),
            client,
            last_used: StdMutex::new(Instant::now()),
            keep_alive: StdMutex::new(Some(self.cfg.keep_alive)),
            active: AtomicUsize::new(0),
        });
        self.workers
            .lock()
            .await
            .insert(spec.name.clone(), w.clone());
        Ok(w)
    }

    /// Stop a model's worker. Returns whether one was running.
    pub async fn unload(&self, name: &str) -> bool {
        let Some(w) = self.workers.lock().await.remove(name) else {
            return false;
        };
        stop(&w).await;
        true
    }

    /// Unload every worker not serving a request. Returns how many.
    pub async fn unload_idle(&self) -> usize {
        let idle: Vec<String> = self
            .workers
            .lock()
            .await
            .values()
            .filter(|w| !w.is_busy())
            .map(|w| w.name.clone())
            .collect();
        for n in &idle {
            self.unload(n).await;
        }
        idle.len()
    }

    pub async fn list(&self) -> Vec<Arc<Worker>> {
        let mut v: Vec<Arc<Worker>> = self.workers.lock().await.values().cloned().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    pub async fn get(&self, name: &str) -> Option<Arc<Worker>> {
        self.alive(name).await
    }

    /// Unload idle workers whose keep-alive has run out; call periodically.
    pub async fn reap(&self) {
        let expired: Vec<String> = self
            .workers
            .lock()
            .await
            .values()
            .filter(|w| {
                !w.is_busy()
                    && match *w.keep_alive.lock().unwrap() {
                        Some(ka) => w.last_used.lock().unwrap().elapsed() >= ka,
                        None => false,
                    }
            })
            .map(|w| w.name.clone())
            .collect();
        for n in expired {
            self.unload(&n).await;
        }
    }

    pub async fn shutdown(&self) {
        let names: Vec<String> = self.workers.lock().await.keys().cloned().collect();
        for n in names {
            self.unload(&n).await;
        }
    }
}

fn aie_llama_pin_defaults() -> aie_worker::PinOptions {
    aie_worker::PinOptions {
        mlock: true,
        hugepages: true,
        // CUDA registration when the worker has a GPU (driver speed-up; the
        // worker drops it otherwise). Locking is the OS-level backstop.
        cuda_register: true,
        ..Default::default()
    }
}

async fn stop(w: &Worker) {
    let mut child = w.child.lock().await;
    if let Some(pid) = child.id() {
        // SIGTERM first, then SIGKILL if it lingers.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .is_err()
        {
            let _ = child.kill().await;
        }
    }
    let _ = std::fs::remove_file(&w.socket);
}

/// Where worker sockets go: `$XDG_RUNTIME_DIR/ai-engine` or `/tmp/aie-<uid>`.
fn socket_dir() -> PathBuf {
    match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(d) if !d.is_empty() && PathBuf::from(&d).is_dir() => {
            PathBuf::from(d).join("ai-engine")
        }
        _ => PathBuf::from(format!("/tmp/aie-{}", unsafe { libc::getuid() })),
    }
}

fn tail(path: &Path, n: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// The reason from a CLI's stderr, on one line: what follows its last
/// `error: `, which may span lines (the fit report). For a fit report, only
/// the devices that don't fit are kept. Callers add the model's name.
fn error_reason(stderr: &str) -> String {
    let start = stderr.rfind("error: ").map_or(0, |i| i + "error: ".len());
    let lines: Vec<String> = stderr[start..]
        .lines()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .collect();
    let short: Vec<String> = lines
        .iter()
        .filter(|l| l.ends_with("DOES NOT FIT"))
        .map(|l| l.trim_end_matches("DOES NOT FIT").trim_end().to_string())
        .collect();
    match lines.first() {
        Some(head) if !short.is_empty() => format!("{head} {}", short.join("; ")),
        _ => lines.join("; "),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn error_reasons_are_one_line_and_name_what_does_not_fit() {
        let fit = "llama log noise\nerror: this placement doesn't fit:\nRAM          15000 MiB needed of      7000 free (weights 600, KV 14000, expert cache 0, overhead 256)  DOES NOT FIT\ndisk             0 MiB mapped (may be dropped from RAM)\n";
        assert_eq!(
            super::error_reason(fit),
            "this placement doesn't fit: RAM 15000 MiB needed of 7000 free (weights 600, KV 14000, expert cache 0, overhead 256)"
        );
        assert_eq!(
            super::error_reason("error: GPU 3 doesn't exist (this machine has 0)\n"),
            "GPU 3 doesn't exist (this machine has 0)"
        );
    }
}
