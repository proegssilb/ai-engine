//! Placement edits for several models, checked and applied together.
//!
//! The UI collects edits for any number of models and applies them at once.
//! They're checked as one: what all the edited models would need on each
//! device, against what will be free once the edited models that are loaded
//! now have been unloaded. If it doesn't fit, nothing changes. Needs come from
//! llama.cpp's own dry run (the same numbers each load checks against), so a
//! check that passes here shouldn't fail at load for lack of memory, unless
//! something else took memory in between.

use aie_anatomy::Anatomy;
use aie_plan::{Plan, Tier};
use aie_worker::Measured;
use serde::Serialize;
use std::collections::BTreeMap;

/// The RAM device's key in the maps below (GPUs use llama.cpp's names, `CUDA0`).
pub const RAM: &str = "RAM";

/// One device that placements compete for.
#[derive(Clone, Debug, PartialEq)]
pub struct Device {
    pub name: String,
    pub total: u64,
    /// Free right now (RAM: available, i.e. including cache that can be dropped).
    pub free: u64,
}

/// Bytes a model holds (or would hold) per device.
pub type Usage = BTreeMap<String, u64>;

/// What a model with this plan uses per device: on GPUs, llama.cpp's
/// breakdown (weights, KV cache and scratch); in RAM, the RAM tier's weights
/// plus the host's KV cache and scratch. The disk tier isn't counted: it's
/// the mapped file, which the kernel can always drop.
pub fn usage(a: &Anatomy, plan: &Plan, measured: &[Measured]) -> Usage {
    let mut u = Usage::new();
    for m in measured {
        match m.kind.as_str() {
            "device" if m.total > 0 => *u.entry(m.name.clone()).or_default() += m.need(),
            "host" => *u.entry(RAM.into()).or_default() += m.context + m.compute,
            _ => {}
        }
    }
    let ram_weights = plan.bytes_by_tier(a).get(&Tier::Ram).copied().unwrap_or(0);
    *u.entry(RAM.into()).or_default() += ram_weights;
    u.retain(|_, b| *b > 0);
    u
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DeviceCheck {
    pub name: String,
    pub total: u64,
    /// Free now.
    pub free: u64,
    /// Freed by unloading the edited models that are loaded now.
    pub reclaimed: u64,
    /// `free + reclaimed`, capped at the total.
    pub available: u64,
    /// What the edited models need here, in all.
    pub need: u64,
    pub by_model: BTreeMap<String, u64>,
    pub fits: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub fits: bool,
    pub devices: Vec<DeviceCheck>,
    /// Plain-language reasons it doesn't fit, or models that couldn't be planned.
    pub problems: Vec<String>,
}

fn gib(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
    } else {
        format!("{} MiB", b >> 20)
    }
}

/// `CUDA1` -> `GPU 1`, as the UI names devices.
fn label(name: &str) -> String {
    match name.strip_prefix("CUDA") {
        Some(n) if n.parse::<u32>().is_ok() => format!("GPU {n}"),
        _ => name.to_string(),
    }
}

/// Check the edited models' needs together.
///
/// - `devices`: the machine now (GPUs and RAM).
/// - `edits`: each edited model's need, or why it couldn't be planned.
/// - `loaded`: what the edited models that are loaded now use; it's freed
///   when they're unloaded to apply the edits.
pub fn check(
    devices: &[Device],
    edits: &[(String, Result<Usage, String>)],
    loaded: &[(String, Usage)],
) -> Check {
    let mut problems = Vec::new();
    for (model, r) in edits {
        if let Err(e) = r {
            problems.push(format!("{model}: {e}"));
        }
    }
    let mut out = Vec::new();
    for d in devices {
        let reclaimed: u64 = loaded.iter().filter_map(|(_, u)| u.get(&d.name)).sum();
        let available = (d.free + reclaimed).min(d.total.max(d.free));
        let by_model: BTreeMap<String, u64> = edits
            .iter()
            .filter_map(|(m, r)| Some((m.clone(), *r.as_ref().ok()?.get(&d.name)?)))
            .collect();
        let need: u64 = by_model.values().sum();
        let fits = need <= available;
        if !fits {
            let who: Vec<String> = by_model
                .iter()
                .map(|(m, b)| format!("{m} {}", gib(*b)))
                .collect();
            problems.push(format!(
                "{} would need {} but only {} will be free ({})",
                label(&d.name),
                gib(need),
                gib(available),
                who.join(", ")
            ));
        }
        out.push(DeviceCheck {
            name: d.name.clone(),
            total: d.total,
            free: d.free,
            reclaimed,
            available,
            need,
            by_model,
            fits,
        });
    }
    // A plan that uses a device this machine doesn't have.
    for (model, r) in edits {
        if let Ok(u) = r {
            for name in u.keys() {
                if !devices.iter().any(|d| &d.name == name) {
                    problems.push(format!(
                        "{model}: {} doesn't exist on this machine",
                        label(name)
                    ));
                }
            }
        }
    }
    Check {
        fits: problems.is_empty(),
        devices: out,
        problems,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1 << 30;

    fn dev(name: &str, total: u64, free: u64) -> Device {
        Device {
            name: name.into(),
            total,
            free,
        }
    }
    fn u(pairs: &[(&str, u64)]) -> Usage {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn edits_are_checked_together() {
        let devices = [dev("CUDA0", 32 * G, 20 * G), dev(RAM, 64 * G, 40 * G)];
        // Each fits alone; together they don't.
        let edits = [
            ("a".to_string(), Ok(u(&[("CUDA0", 12 * G)]))),
            ("b".to_string(), Ok(u(&[("CUDA0", 12 * G), (RAM, G)]))),
        ];
        let c = check(&devices, &edits, &[]);
        assert!(!c.fits);
        assert_eq!(c.devices[0].need, 24 * G);
        assert!(!c.devices[0].fits && c.devices[1].fits);
        assert_eq!(
            c.problems,
            ["GPU 0 would need 24.0 GiB but only 20.0 GiB will be free (a 12.0 GiB, b 12.0 GiB)"]
        );
    }

    #[test]
    fn memory_of_models_being_reloaded_counts_as_free() {
        let devices = [dev("CUDA0", 32 * G, 4 * G)];
        // a holds 20 GiB now and moves to a 22 GiB plan: 4 free + 20 reclaimed.
        let edits = [("a".to_string(), Ok(u(&[("CUDA0", 22 * G)])))];
        let loaded = [("a".to_string(), u(&[("CUDA0", 20 * G)]))];
        let c = check(&devices, &edits, &loaded);
        assert!(c.fits, "{:?}", c.problems);
        assert_eq!(
            (c.devices[0].reclaimed, c.devices[0].available),
            (20 * G, 24 * G)
        );
    }

    #[test]
    fn unplannable_models_and_missing_devices_are_problems() {
        let devices = [dev("CUDA0", 32 * G, 30 * G)];
        let edits = [
            (
                "a".to_string(),
                Err("core layers can't go on disk".to_string()),
            ),
            ("b".to_string(), Ok(u(&[("CUDA1", G)]))),
        ];
        let c = check(&devices, &edits, &[]);
        assert!(!c.fits);
        assert_eq!(
            c.problems,
            [
                "a: core layers can't go on disk",
                "b: GPU 1 doesn't exist on this machine"
            ]
        );
    }

    #[test]
    fn usage_counts_gpus_from_the_dry_run_and_ram_from_the_plan_and_host() {
        let part = |name: &str, layer: Option<u32>, bytes: u64| aie_anatomy::Part {
            name: name.into(),
            subsystem: aie_anatomy::Subsystem::Attention,
            layer,
            ty: aie_gguf::GgmlType::F32,
            bytes,
        };
        let a = Anatomy {
            architecture: "qwen3".into(),
            layers: 2,
            embedding_len: None,
            context_train: None,
            moe: None,
            kv: None,
            parts: vec![
                part("blk.0.attn_q.weight", Some(0), 100),
                part("blk.1.attn_q.weight", Some(1), 50),
            ],
        };
        let plan = Plan {
            layer_home: vec![Tier::Gpu(0), Tier::Ram],
            tensors: [
                ("blk.0.attn_q.weight".to_string(), Tier::Gpu(0)),
                ("blk.1.attn_q.weight".to_string(), Tier::Ram),
            ]
            .into(),
            n_ctx: 512,
            expert_cache_bytes: 0,
            allow_repack: false,
            host_node: None,
        };
        let m =
            |name: &str, kind: &str, total: u64, model: u64, context: u64, compute: u64| Measured {
                name: name.into(),
                kind: kind.into(),
                total,
                free: total,
                model,
                context,
                compute,
            };
        let measured = [
            m("CUDA0", "device", 1000, 100, 20, 30),
            m("Host", "host", 0, 999, 7, 3),
            m("CPU_Mapped", "buffer_type", 0, 50, 0, 0),
        ];
        assert_eq!(usage(&a, &plan, &measured), u(&[("CUDA0", 150), (RAM, 60)]));
    }
}
