//! `ai-engine`: daemon, worker and CLI in one binary.

use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "ai-engine",
    version,
    about = "Run models on your own hardware, with control over where each part lives"
)]
struct Cli {
    /// Daemon address for client commands.
    #[arg(
        long,
        global = true,
        env = "AIE_HOST",
        default_value = "127.0.0.1:11435"
    )]
    host: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the daemon (API on AIE_HOST; data in AIE_HOME, default ~/.ai-engine).
    Serve,
    /// Download a model: hf.co/OWNER/REPO[:TAG] or REGISTRY/REPO[:TAG].
    Pull {
        model: String,
        /// Which store (disk) to put it in, by root path; default the first.
        #[arg(long)]
        store: Option<String>,
    },
    /// List downloaded models.
    #[command(alias = "ls")]
    List,
    /// Delete a downloaded model.
    Rm { model: String },
    /// Show loaded models.
    Ps,
    /// Show a model's anatomy and, if loaded, where its weights really are.
    Place {
        model: String,
        /// Save a placement intent from a JSON file (see docs), applied at next load.
        #[arg(long)]
        set: Option<PathBuf>,
        /// Forget the saved placement (back to automatic).
        #[arg(long, conflicts_with = "set")]
        auto: bool,
        /// Unload the model now so the new placement applies at the next request.
        #[arg(long)]
        reload: bool,
    },
    /// Internal: print the devices llama.cpp sees, as JSON (used by the daemon).
    #[command(hide = true)]
    Probe,
    /// Internal: what a placement would need, without loading (used by the daemon).
    #[command(hide = true)]
    DryRun {
        #[arg(long)]
        config: PathBuf,
    },
    /// Internal: one model's worker process, started by the daemon.
    #[command(hide = true)]
    Worker {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        socket: PathBuf,
    },
}

fn main() {
    // Exit quietly when a pipe closes (`ai-engine list | head`) instead of
    // panicking on the failed write, as command-line tools do.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
    let cli = Cli::parse();
    // Threads the runtime starts later (the blocking pool runs the model) get
    // the worker's socket pinning; before a worker applies one this does nothing.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .on_thread_start(aie_worker::numa::init_thread)
        .build()
        .expect("tokio runtime");
    let result = rt.block_on(async move {
        match cli.cmd {
            Cmd::Serve => {
                let mut cfg = aie_daemon::Config::load()?;
                if std::env::var_os("AIE_HOST").is_some() || cli.host != "127.0.0.1:11435" {
                    cfg.listen = aie_daemon::config::parse_host(&cli.host)
                        .ok_or(format!("bad --host {}", cli.host))?;
                }
                aie_daemon::serve(cfg).await
            }
            Cmd::Probe => {
                println!("{}", aie_worker::probe());
                Ok(())
            }
            Cmd::DryRun { config } => {
                let text = std::fs::read_to_string(&config)
                    .map_err(|e| format!("{}: {e}", config.display()))?;
                let cfg: aie_worker::Config = serde_json::from_str(&text)
                    .map_err(|e| format!("{}: {e}", config.display()))?;
                let d = aie_worker::dry_run(&cfg)?;
                println!("{}", serde_json::to_string(&d).map_err(|e| e.to_string())?);
                Ok(())
            }
            Cmd::Worker { config, socket } => {
                let text = std::fs::read_to_string(&config)
                    .map_err(|e| format!("{}: {e}", config.display()))?;
                let cfg: aie_worker::Config = serde_json::from_str(&text)
                    .map_err(|e| format!("{}: {e}", config.display()))?;
                aie_worker::run(cfg, &socket).await
            }
            Cmd::Pull { model, store } => pull(&cli.host, &model, store).await,
            Cmd::List => list(&cli.host).await,
            Cmd::Rm { model } => {
                call(
                    &cli.host,
                    reqwest::Method::DELETE,
                    "/api/delete",
                    Some(json!({ "model": model })),
                )
                .await?;
                println!("deleted {model}");
                Ok(())
            }
            Cmd::Ps => ps(&cli.host).await,
            Cmd::Place {
                model,
                set,
                auto,
                reload,
            } => place(&cli.host, &model, set, auto, reload).await,
        }
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn url(host: &str, path: &str) -> String {
    let h = host.trim_end_matches('/');
    if h.starts_with("http") {
        format!("{h}{path}")
    } else {
        format!("http://{h}{path}")
    }
}

async fn call(
    host: &str,
    method: reqwest::Method,
    path: &str,
    body: Option<Value>,
) -> Result<Value, String> {
    let mut req = reqwest::Client::new().request(method, url(host, path));
    if let Some(b) = body {
        req = req.json(&b);
    }
    let resp = req.send().await.map_err(|e| {
        format!("can't reach the daemon at {host} ({e}); is `ai-engine serve` running?")
    })?;
    let status = resp.status();
    let v: Value = resp.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(v
            .get("error")
            .and_then(|e| e.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("HTTP {status}")));
    }
    Ok(v)
}

fn human(b: u64) -> String {
    let g = b as f64 / (1u64 << 30) as f64;
    if g >= 1.0 {
        format!("{g:.1} GiB")
    } else {
        format!("{:.0} MiB", b as f64 / (1u64 << 20) as f64)
    }
}

async fn pull(host: &str, model: &str, store: Option<String>) -> Result<(), String> {
    let mut resp = reqwest::Client::new()
        .post(url(host, "/api/pull"))
        .json(&json!({ "model": model, "stream": true, "store": store }))
        .send()
        .await
        .map_err(|e| format!("can't reach the daemon at {host} ({e})"))?;
    let mut buf = Vec::new();
    let mut last = String::new();
    while let Some(chunk) = resp.chunk().await.map_err(|e| e.to_string())? {
        buf.extend_from_slice(&chunk);
        while let Some(i) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=i).collect();
            let Ok(v) = serde_json::from_slice::<Value>(&line) else {
                continue;
            };
            if let Some(e) = v.get("error") {
                eprintln!();
                return Err(e.as_str().unwrap_or("pull failed").to_string());
            }
            let status = v["status"].as_str().unwrap_or("");
            match (
                v.get("total").and_then(|x| x.as_u64()),
                v.get("completed").and_then(|x| x.as_u64()),
            ) {
                (Some(t), Some(c)) => {
                    eprint!(
                        "\r{status}: {:>5.1}% of {}   ",
                        100.0 * c as f64 / t.max(1) as f64,
                        human(t)
                    );
                    last = status.to_string();
                }
                _ => {
                    if !last.is_empty() {
                        eprintln!();
                        last.clear();
                    }
                    eprintln!("{status}");
                }
            }
        }
    }
    Ok(())
}

async fn list(host: &str) -> Result<(), String> {
    let v = call(host, reqwest::Method::GET, "/api/tags", None).await?;
    println!("{:<55} {:>10}  {:<10} QUANT", "NAME", "SIZE", "FAMILY");
    for m in v["models"].as_array().into_iter().flatten() {
        println!(
            "{:<55} {:>10}  {:<10} {}",
            m["name"].as_str().unwrap_or(""),
            human(m["size"].as_u64().unwrap_or(0)),
            m["details"]["family"].as_str().unwrap_or(""),
            m["details"]["quantization_level"].as_str().unwrap_or("")
        );
    }
    Ok(())
}

async fn ps(host: &str) -> Result<(), String> {
    let v = call(host, reqwest::Method::GET, "/api/ps", None).await?;
    println!(
        "{:<55} {:>10} {:>10} {:>7}  UNTIL",
        "NAME", "SIZE", "ON GPU", "CONTEXT"
    );
    for m in v["models"].as_array().into_iter().flatten() {
        println!(
            "{:<55} {:>10} {:>10} {:>7}  {}",
            m["name"].as_str().unwrap_or(""),
            human(m["size"].as_u64().unwrap_or(0)),
            human(m["size_vram"].as_u64().unwrap_or(0)),
            m["context_length"].as_u64().unwrap_or(0),
            m["expires_at"].as_str().unwrap_or("")
        );
    }
    Ok(())
}

async fn place(
    host: &str,
    model: &str,
    set: Option<PathBuf>,
    auto: bool,
    reload: bool,
) -> Result<(), String> {
    if set.is_some() || auto {
        let intent: Value = match &set {
            Some(p) => serde_json::from_str(
                &std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
            )
            .map_err(|e| format!("{}: {e}", p.display()))?,
            None => Value::Null,
        };
        let v = call(
            host,
            reqwest::Method::PUT,
            "/api/aie/placement",
            Some(json!({ "model": model, "intent": intent, "reload": reload })),
        )
        .await?;
        let applies = if v["unloaded"].as_bool() == Some(true) {
            "unloaded; applies at the next request"
        } else {
            "applies at the next load"
        };
        let name = v["model"].as_str().unwrap_or(model);
        if auto {
            println!("{name}: back to automatic placement; {applies}");
        } else {
            println!("{name}: placement saved; {applies}");
        }
        return Ok(());
    }
    let v = call(
        host,
        reqwest::Method::GET,
        &format!("/api/aie/placement?model={}", urlencode(model)),
        None,
    )
    .await?;
    let a = &v["anatomy"];
    println!(
        "{}  ({}, {} layers{})",
        v["model"].as_str().unwrap_or(""),
        a["architecture"].as_str().unwrap_or(""),
        a["layers"],
        if a["moe"].is_null() {
            String::new()
        } else {
            format!(
                ", {} experts, {} active",
                a["moe"]["experts"], a["moe"]["experts_used"]
            )
        }
    );
    println!(
        "placement: {}",
        if v["intent"].is_null() {
            "automatic (llama.cpp's fit)".to_string()
        } else {
            "saved intent".to_string()
        }
    );
    let loaded = &v["loaded"];
    if loaded.is_null() {
        println!("\nnot loaded; subsystem sizes:");
        for (s, b) in a["subsystems"].as_object().into_iter().flatten() {
            println!("  {s:<16} {:>10}", human(b.as_u64().unwrap_or(0)));
        }
        return Ok(());
    }
    println!(
        "loaded: {}, context {}, {}\n",
        loaded["description"].as_str().unwrap_or(""),
        loaded["n_ctx"],
        match loaded["misplaced"].as_u64() {
            Some(0) => "every weight verified where planned".to_string(),
            Some(n) => format!("{n} WEIGHTS NOT WHERE PLANNED"),
            None => String::new(),
        }
    );
    for w in loaded["warnings"].as_array().into_iter().flatten() {
        println!("warning: {}", w.as_str().unwrap_or(""));
    }
    let rt = &loaded["stats"]["ram_tier"];
    if rt["bytes"].as_u64().unwrap_or(0) > 0 {
        let held = rt["held"].as_bool() == Some(true);
        let by: Vec<String> = rt["mechanisms"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|m| m["ok"].as_bool() == Some(true))
            .map(|m| {
                format!(
                    "{}{}",
                    m["name"].as_str().unwrap_or(""),
                    if m["layer"] == "os" {
                        String::new()
                    } else {
                        format!(" ({})", m["layer"].as_str().unwrap_or(""))
                    }
                )
            })
            .collect();
        println!(
            "RAM tier: {} in RAM, {} swapped: {}",
            human(rt["resident"].as_u64().unwrap_or(0)),
            human(rt["swapped"].as_u64().unwrap_or(0)),
            if !held {
                "NOT HELD, may be swapped out".to_string()
            } else if rt["held_by_os"].as_bool() == Some(true) {
                format!("held ({})", by.join(", "))
            } else {
                format!(
                    "held only by the driver ({}); no OS-level backstop",
                    by.join(", ")
                )
            }
        );
        if rt["held_by_os"].as_bool() != Some(true) {
            for m in rt["mechanisms"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|m| m["ok"].as_bool() == Some(false))
            {
                println!(
                    "  {}: {}",
                    m["name"].as_str().unwrap_or(""),
                    m["detail"].as_str().unwrap_or("")
                );
            }
            for f in rt["fixes"].as_array().into_iter().flatten() {
                println!("  fix: {}", f.as_str().unwrap_or(""));
            }
        }
    }
    // The host side's socket, only where there's a choice of sockets.
    let host = &loaded["host"];
    if host["sockets"].as_u64().unwrap_or(1) > 1 {
        let gbs = |m: &Value| -> String {
            m.as_object()
                .into_iter()
                .flatten()
                .map(|(n, b)| format!("socket {n} {}", human(b.as_u64().unwrap_or(0))))
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "host: socket {} ({}), {} threads; RAM tier on {}",
            host["node"],
            if host["auto"].as_bool() == Some(true) {
                host["reason"].as_str().unwrap_or("automatic")
            } else {
                "chosen"
            },
            host["threads"],
            gbs(&host["ram_tier_by_node"])
        );
        for p in host["problems"].as_array().into_iter().flatten() {
            println!("  problem: {}", p.as_str().unwrap_or(""));
        }
    }
    println!("{:<16} where (planned, verified)", "subsystem");
    for (s, tiers) in loaded["planned"].as_object().into_iter().flatten() {
        let parts: Vec<String> = tiers
            .as_object()
            .into_iter()
            .flatten()
            .map(|(t, b)| format!("{t} {}", human(b.as_u64().unwrap_or(0))))
            .collect();
        println!("{s:<16} {}", parts.join(", "));
    }
    Ok(())
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}
