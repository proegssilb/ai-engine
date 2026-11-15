//! Placement API: what the placement UI and `ai-engine place` use.
//!
//! GET  /api/aie/placement?model=NAME  anatomy, saved intent, and (if loaded)
//!                                    the plan and where the weights really are
//! PUT  /api/aie/placement             {model, intent | null, reload?}: save or
//!                                    forget a placement; applies at next load

use super::{Body, State};
use crate::ApiError;
use crate::workers::Worker;
use aie_anatomy::Anatomy;
use aie_plan::{Intent, Plan, Tier};
use axum::Json;
use axum::extract::Query;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub struct ModelQuery {
    model: String,
}

fn anatomy(path: &std::path::Path) -> Result<Anatomy, ApiError> {
    let g = aie_gguf::Gguf::open_header(path).map_err(|e| ApiError::internal(e.to_string()))?;
    Anatomy::from_gguf(&g).map_err(|e| ApiError::internal(e.to_string()))
}

fn tier_key(t: &Tier) -> String {
    match t {
        Tier::Gpu(i) => format!("gpu{i}"),
        Tier::Ram => "ram".into(),
        Tier::Disk => "disk".into(),
    }
}

/// Live numbers from a worker: speeds, totals, disk-tier residency.
pub async fn live_stats(w: &Worker) -> Value {
    let Ok(st) = w.client().status(aie_proto::worker::StatusRequest {}).await else {
        return Value::Null;
    };
    let st = st.into_inner();
    let rate = |tokens: u64, ns: u64| {
        if ns == 0 {
            0.0
        } else {
            tokens as f64 / (ns as f64 / 1e9)
        }
    };
    json!({
        "requests": st.requests,
        "prompt_tokens": st.prompt_tokens,
        "completion_tokens": st.completion_tokens,
        "prefill_tps": rate(st.prompt_tokens, st.prefill_ns),
        "decode_tps": rate(st.completion_tokens.saturating_sub(st.requests), st.decode_ns),
        "last_prefill_tps": st.last_prefill_tps,
        "last_decode_tps": st.last_decode_tps,
        "disk_tier_bytes": st.disk_tier_bytes,
        "disk_tier_resident": st.disk_tier_resident,
    })
}

/// GET /api/aie/stats: every loaded model's live numbers.
pub async fn stats(st: State) -> Json<Value> {
    let mut out = serde_json::Map::new();
    for w in st.0.workers.list().await {
        out.insert(w.name.clone(), live_stats(&w).await);
    }
    Json(Value::Object(out))
}

/// Bytes of weights on GPUs, per the worker's plan.
pub fn gpu_bytes(w: &Worker) -> u64 {
    let Ok(plan) = serde_json::from_str::<Plan>(&w.status.plan_json) else {
        return 0;
    };
    let Ok(a) = anatomy(&w.model_path) else {
        return 0;
    };
    plan.bytes_by_tier(&a)
        .iter()
        .filter(|(t, _)| matches!(t, Tier::Gpu(_)))
        .map(|(_, b)| b)
        .sum()
}

pub async fn get_placement(
    st: State,
    Query(q): Query<ModelQuery>,
) -> Result<Json<Value>, ApiError> {
    let r = st.0.resolve(&q.model)?;
    let a = anatomy(&r.path)?;
    let subsystems: BTreeMap<String, u64> = a
        .subsystem_bytes()
        .into_iter()
        .map(|(s, b)| (s.label().to_string(), b))
        .collect();
    let mut out = json!({
        "model": r.name,
        "anatomy": {
            "architecture": a.architecture,
            "supported": a.is_supported(),
            "layers": a.layers,
            "moe": a.moe,
            "kv": a.kv,
            "bytes": a.total_bytes(),
            "subsystems": subsystems,
        },
        "intent": st.0.saved_intent(&r.name)?,
        "loaded": Value::Null,
    });
    if let Some(w) = st.0.workers.get(&r.name).await {
        let plan: Option<Plan> = serde_json::from_str(&w.status.plan_json).ok();
        let report: Option<Value> = serde_json::from_str(&w.status.placement_json).ok();
        // per-subsystem bytes per tier, as planned
        let mut by_sub: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
        if let Some(p) = &plan {
            for part in &a.parts {
                if let Some(t) = p.tensors.get(&part.name) {
                    *by_sub
                        .entry(part.subsystem.label().to_string())
                        .or_default()
                        .entry(tier_key(t))
                        .or_default() += part.bytes;
                }
            }
        }
        out["loaded"] = json!({
            "description": w.status.description,
            "n_ctx": w.n_ctx,
            "misplaced": w.status.misplaced,
            "load_ms": w.status.load_ms,
            "warnings": w.status.warnings,
            "planned": by_sub,
            "stats": live_stats(&w).await,
            "plan": plan,
            "verified": report,
        });
    }
    Ok(Json(out))
}

#[derive(Deserialize)]
pub struct PutReq {
    model: String,
    intent: Option<Intent>,
    #[serde(default)]
    reload: bool,
}

pub async fn put_placement(st: State, Body(r): Body<PutReq>) -> Result<Json<Value>, ApiError> {
    let res = st.0.resolve(&r.model)?;
    if let Some(i) = &r.intent {
        // Plan it against the model now so obvious mistakes fail here, not at
        // load. The GPU count isn't known in the daemon (no CUDA here), so
        // allow as many GPUs as the intent mentions.
        let a = anatomy(&res.path)?;
        let gpus = i
            .layers
            .iter()
            .chain(i.subsystems.values().flatten())
            .filter_map(|s| {
                if let Tier::Gpu(g) = s.tier {
                    Some(g + 1)
                } else {
                    None
                }
            })
            .chain(
                [i.embeddings, i.output]
                    .into_iter()
                    .flatten()
                    .filter_map(|t| {
                        if let Tier::Gpu(g) = t {
                            Some(g + 1)
                        } else {
                            None
                        }
                    }),
            )
            .max()
            .unwrap_or(0);
        aie_plan::plan(&a, i, gpus).map_err(|e| ApiError::bad_request(e.to_string()))?;
    }
    st.0.save_intent(&res.name, r.intent.as_ref())?;
    let unloaded = if r.reload {
        st.0.workers.unload(&res.name).await
    } else {
        false
    };
    Ok(Json(
        json!({ "model": res.name, "saved": r.intent.is_some(), "unloaded": unloaded }),
    ))
}

/// Free bytes and size of the filesystem holding `path`.
fn disk(path: &std::path::Path) -> Option<(u64, u64)> {
    let c = std::ffi::CString::new(path.to_str()?).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0).then(|| {
        (
            st.f_bavail as u64 * st.f_frsize as u64,
            st.f_blocks as u64 * st.f_frsize as u64,
        )
    })
}

async fn probe(exe: &std::path::Path) -> Result<Value, ApiError> {
    static CACHE: std::sync::Mutex<Option<(std::time::Instant, Value)>> =
        std::sync::Mutex::new(None);
    if let Some((t, v)) = CACHE.lock().unwrap().as_ref()
        && t.elapsed() < std::time::Duration::from_secs(5)
    {
        return Ok(v.clone());
    }
    let out = tokio::process::Command::new(exe)
        .arg("probe")
        .output()
        .await
        .map_err(|e| ApiError::internal(format!("probe: {e}")))?;
    if !out.status.success() {
        return Err(ApiError::internal(format!(
            "probe failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let v: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| ApiError::internal(format!("probe output: {e}")))?;
    *CACHE.lock().unwrap() = Some((std::time::Instant::now(), v.clone()));
    Ok(v)
}

/// The machine for the placement view: devices with free memory, RAM per
/// NUMA node, the store's disk, and what each loaded model planned where.
pub async fn machine(st: State) -> Result<Json<Value>, ApiError> {
    let mut v = probe(&st.0.cfg.worker_exe).await?;
    let disks: Vec<Value> =
        st.0.stores
            .roots()
            .map(|p| {
                let (free, total) = disk(p).unwrap_or((0, 0));
                json!({ "path": p, "total": total, "free": free })
            })
            .collect();
    v["disk"] = disks[0].clone();
    v["disks"] = json!(disks);
    let mut models = Vec::new();
    for w in st.0.workers.list().await {
        let Ok(plan) = serde_json::from_str::<Plan>(&w.status.plan_json) else {
            continue;
        };
        let Ok(a) = anatomy(&w.model_path) else {
            continue;
        };
        let by: BTreeMap<String, u64> = plan
            .bytes_by_tier(&a)
            .iter()
            .map(|(t, b)| (tier_key(t), *b))
            .collect();
        // the store (disk) the model's file lives on: its disk tier
        let store =
            st.0.stores
                .roots()
                .find(|r| w.model_path.starts_with(r))
                .map(|r| r.display().to_string());
        models.push(json!({ "model": w.name, "bytes_by_tier": by, "misplaced": w.status.misplaced, "store": store }));
    }
    v["models"] = json!(models);
    Ok(Json(v))
}
