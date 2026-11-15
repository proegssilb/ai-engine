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
            "planned": by_sub,
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
