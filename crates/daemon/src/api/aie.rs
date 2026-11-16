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
        "ram_tier": serde_json::from_str::<Value>(&st.ram_tier_json).unwrap_or(Value::Null),
        "host": serde_json::from_str::<Value>(&st.host_json).unwrap_or(Value::Null),
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

/// Bytes per subsystem per tier key, as a plan puts them.
fn planned_by_subsystem(a: &Anatomy, plan: &Plan) -> BTreeMap<String, BTreeMap<String, u64>> {
    let mut by_sub: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for part in &a.parts {
        if let Some(t) = plan.tensors.get(&part.name) {
            *by_sub
                .entry(part.subsystem.label().to_string())
                .or_default()
                .entry(tier_key(t))
                .or_default() += part.bytes;
        }
    }
    by_sub
}

/// How many GPUs an intent mentions (the daemon has no CUDA to ask).
fn intent_gpus(i: &Intent) -> u32 {
    let gpu = |t: &Tier| if let Tier::Gpu(g) = t { Some(g + 1) } else { None };
    i.layers
        .iter()
        .chain(i.subsystems.values().flatten())
        .filter_map(|s| gpu(&s.tier))
        .chain([i.embeddings, i.output].into_iter().flatten().filter_map(|t| gpu(&t)))
        .max()
        .unwrap_or(0)
}

/// What a tier or location looks like in the verified report (`Where` as JSON).
fn where_text(w: &Value) -> String {
    if let Some(t) = w.get("Tier") {
        return match serde_json::from_value::<Tier>(t.clone()) {
            Ok(t) => match t {
                Tier::Gpu(i) => format!("GPU {i}"),
                Tier::Ram => "RAM".into(),
                Tier::Disk => "the model file (disk tier)".into(),
            },
            Err(_) => t.to_string(),
        };
    }
    if let Some(b) = w.get("Buffer").and_then(Value::as_str) {
        return format!("a {b} buffer");
    }
    "host memory, seen only via copies".into()
}

/// The weights verify found somewhere other than planned, grouped by
/// subsystem and planned tier: `{subsystem, planned, actual, tensors, bytes}`.
fn misplaced_detail(a: &Anatomy, plan: &Plan, report: &Value) -> Vec<Value> {
    let Some(weights) = report.get("weights").and_then(Value::as_object) else {
        return vec![];
    };
    let mut groups: BTreeMap<(String, String, String), (u64, u64)> = BTreeMap::new();
    for part in &a.parts {
        let (Some(planned), Some(actual)) = (plan.tensors.get(&part.name), weights.get(&part.name))
        else {
            continue;
        };
        if actual.get("Tier").and_then(|t| serde_json::from_value::<Tier>(t.clone()).ok()) == Some(*planned) {
            continue;
        }
        let e = groups
            .entry((part.subsystem.label().into(), tier_key(planned), where_text(actual)))
            .or_default();
        e.0 += 1;
        e.1 += part.bytes;
    }
    groups
        .into_iter()
        .map(|((sub, planned, actual), (n, b))| {
            json!({ "subsystem": sub, "planned": planned, "actual": actual, "tensors": n, "bytes": b })
        })
        .collect()
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
    // A saved placement that isn't loaded: where it would put each subsystem.
    if let Some(i) = st.0.saved_intent(&r.name)?
        && st.0.workers.get(&r.name).await.is_none()
        && let Ok(plan) = aie_plan::plan(&a, &i, intent_gpus(&i))
    {
        out["intended"] = json!(planned_by_subsystem(&a, &plan));
    }
    if let Some(w) = st.0.workers.get(&r.name).await {
        let plan: Option<Plan> = serde_json::from_str(&w.status.plan_json).ok();
        let report: Option<Value> = serde_json::from_str(&w.status.placement_json).ok();
        // per-subsystem bytes per tier, as planned
        let by_sub = plan
            .as_ref()
            .map(|p| planned_by_subsystem(&a, p))
            .unwrap_or_default();
        // The expert cache follows the layer split; users can't place it.
        let expert_cache: Vec<Value> = plan
            .as_ref()
            .map(|p| {
                aie_plan::fit::expert_cache_split(&a, p)
                    .into_iter()
                    .map(|(gpu, bytes)| json!({ "gpu": gpu, "bytes": bytes }))
                    .collect()
            })
            .unwrap_or_default();
        let stats = live_stats(&w).await;
        out["loaded"] = json!({
            "description": w.status.description,
            "n_ctx": w.n_ctx,
            "misplaced": w.status.misplaced,
            "load_ms": w.status.load_ms,
            "warnings": w.status.warnings,
            "measured": serde_json::from_str::<Value>(&w.status.memory_json).unwrap_or(Value::Null),
            "planned": by_sub,
            // weights verify found elsewhere than planned, by subsystem
            "misplaced_detail": match (&plan, &report) {
                (Some(p), Some(r)) => misplaced_detail(&a, p, r),
                _ => vec![],
            },
            "expert_cache": expert_cache,
            // which socket the host side is pinned to, and where it really is
            "host": stats["host"].clone(),
            "stats": stats,
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
        let gpus = intent_gpus(i);
        aie_plan::plan(&a, i, gpus).map_err(|e| ApiError::bad_request(e.to_string()))?;
        // The socket must exist here (the daemon reads the same sysfs the workers do).
        let topo = aie_worker::numa::Topology::read(std::path::Path::new("/"));
        aie_worker::numa::choose(&topo, i.host_node, &[], &[]).map_err(ApiError::bad_request)?;
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
                // the socket its block device hangs off; null when unknown
                let node = aie_worker::numa::path_node(std::path::Path::new("/"), p);
                json!({ "path": p, "total": total, "free": free, "numa_node": node })
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
        // the live host report: socket, affinity, RAM tier per node
        let host = live_stats(&w).await["host"].clone();
        models.push(json!({
            "model": w.name,
            "bytes_by_tier": by,
            "planned": planned_by_subsystem(&a, &plan),
            // what the live context allocated per device and on the host
            "memory": serde_json::from_str::<Value>(&w.status.memory_json).unwrap_or(Value::Null),
            "n_ctx": w.n_ctx,
            "misplaced": w.status.misplaced,
            "store": store,
            "host": host,
        }));
    }
    v["models"] = json!(models);
    // RAM over the last minute (latest, min, max), for the fuzzy RAM bar.
    v["memory"] = st.0.memory.snapshot();
    Ok(Json(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aie_anatomy::{Part, Subsystem};

    fn anatomy() -> Anatomy {
        let part = |name: &str, subsystem, bytes| Part {
            name: name.into(),
            subsystem,
            layer: None,
            ty: aie_gguf::GgmlType::F32,
            bytes,
        };
        Anatomy {
            architecture: "t".into(),
            layers: 1,
            embedding_len: None,
            context_train: None,
            moe: None,
            kv: None,
            parts: vec![
                part("a", Subsystem::Attention, 10),
                part("b", Subsystem::Attention, 20),
                part("c", Subsystem::DenseFfn, 40),
            ],
        }
    }

    #[test]
    fn misplaced_weights_are_grouped_by_subsystem_and_where_they_are() {
        let a = anatomy();
        let plan = Plan {
            layer_home: vec![Tier::Gpu(0)],
            tensors: [
                ("a".to_string(), Tier::Gpu(0)),
                ("b".to_string(), Tier::Gpu(0)),
                ("c".to_string(), Tier::Ram),
            ]
            .into(),
            n_ctx: 1,
            expert_cache_bytes: 0,
            allow_repack: false,
            host_node: None,
        };
        let report = json!({ "weights": {
            "a": { "Tier": { "Gpu": 0 } },
            "b": { "Buffer": "CUDA_Host" },
            "c": "HostViaCopy",
        }});
        let d = misplaced_detail(&a, &plan, &report);
        assert_eq!(d.len(), 2, "{d:?}");
        assert_eq!(d[0]["subsystem"], "attention");
        assert_eq!(d[0]["planned"], "gpu0");
        assert_eq!(d[0]["actual"], "a CUDA_Host buffer");
        assert_eq!((d[0]["tensors"].as_u64(), d[0]["bytes"].as_u64()), (Some(1), Some(20)));
        assert_eq!(d[1]["subsystem"], "dense ffn");
        assert_eq!(d[1]["actual"], "host memory, seen only via copies");
        // all where planned: nothing to report
        let ok = json!({ "weights": { "a": { "Tier": { "Gpu": 0 } }, "b": { "Tier": { "Gpu": 0 } }, "c": { "Tier": "Ram" } }});
        assert!(misplaced_detail(&a, &plan, &ok).is_empty());
    }
}
