use crate::{Amount, Intent, Plan, Share, Tier};
use aie_anatomy::{Anatomy, Subsystem};
use std::collections::BTreeMap;
use std::fmt;

/// Why an intent can't be planned.
#[derive(Clone, Debug, PartialEq)]
pub enum PlanError {
    NoLayerShares,
    /// Layers (attention + KV cache) must live on a GPU or in RAM.
    LayersOnDisk,
    NoSuchGpu {
        index: u32,
        gpus: u32,
    },
    BadFraction(f64),
    MoreThanOneRest(String),
    /// The shares claim more than the whole.
    OverCommitted {
        what: String,
    },
    /// The shares leave part of the subsystem unplaced and none is `Rest`.
    Uncovered {
        what: String,
        placed: u64,
        total: u64,
    },
    /// Embeddings and the output head have their own fields in [`Intent`].
    NotPerLayer(Subsystem),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::NoLayerShares => write!(f, "no placement given for the layers"),
            PlanError::LayersOnDisk => write!(
                f,
                "layers (attention and KV cache) must be on a GPU or in RAM"
            ),
            PlanError::NoSuchGpu { index, gpus } => {
                write!(f, "GPU {index} doesn't exist (this machine has {gpus})")
            }
            PlanError::BadFraction(x) => write!(f, "share {x} is not between 0 and 1"),
            PlanError::MoreThanOneRest(w) => write!(f, "{w}: only one share can take the rest"),
            PlanError::OverCommitted { what } => {
                write!(f, "{what}: shares add up to more than 100%")
            }
            PlanError::Uncovered {
                what,
                placed,
                total,
            } => {
                write!(
                    f,
                    "{what}: shares place {placed} of {total} and none takes the rest"
                )
            }
            PlanError::NotPerLayer(s) => {
                write!(f, "{s} isn't per-layer; use the embeddings/output fields")
            }
        }
    }
}

impl std::error::Error for PlanError {}

/// Turn an intent into a concrete plan for a model on a machine with `gpus` GPUs.
pub fn plan(a: &Anatomy, intent: &Intent, gpus: u32) -> Result<Plan, PlanError> {
    let check_tier = |t: Tier| match t {
        Tier::Gpu(i) if i >= gpus => Err(PlanError::NoSuchGpu { index: i, gpus }),
        _ => Ok(()),
    };
    for s in intent
        .layers
        .iter()
        .chain(intent.subsystems.values().flatten())
    {
        check_tier(s.tier)?;
        if let Amount::Fraction(x) = s.amount
            && !(0.0..=1.0).contains(&x)
        {
            return Err(PlanError::BadFraction(x));
        }
    }
    for t in [intent.embeddings, intent.output].into_iter().flatten() {
        check_tier(t)?;
    }

    let layer_home = homes(a, intent)?;

    // Per-layer subsystems with their own shares.
    let mut sub_tier: BTreeMap<Subsystem, Vec<Option<Tier>>> = BTreeMap::new();
    for (&s, shares) in &intent.subsystems {
        if matches!(s, Subsystem::Embeddings | Subsystem::OutputHead) {
            return Err(PlanError::NotPerLayer(s));
        }
        let sizes = a.layer_bytes(s);
        sub_tier.insert(s, assign(&sizes, shares, &layer_home, s.label())?);
    }

    // llama.cpp puts the output layer after the last offloaded layer: the last
    // GPU that has layers, or the CPU (the mapped file) if none does.
    let last_gpu = layer_home
        .iter()
        .filter_map(|t| matches!(t, Tier::Gpu(_)).then_some(*t))
        .max();
    let output_default = last_gpu.unwrap_or(Tier::Disk);

    let mut tensors = BTreeMap::new();
    for p in &a.parts {
        let tier = match (p.layer, p.subsystem) {
            (Some(l), s) => sub_tier
                .get(&s)
                .and_then(|v| v[l as usize])
                .unwrap_or(layer_home[l as usize]),
            (None, Subsystem::Embeddings) => intent.embeddings.unwrap_or(Tier::Disk),
            (None, Subsystem::OutputHead) => intent.output.unwrap_or(output_default),
            (None, _) => Tier::Disk,
        };
        tensors.insert(p.name.clone(), tier);
    }

    Ok(Plan {
        layer_home,
        tensors,
        n_ctx: intent.n_ctx,
        expert_cache_bytes: intent.expert_cache_bytes,
        allow_repack: intent.allow_repack,
    })
}

/// Layer homes from the layer shares. llama.cpp can only express "the first
/// k layers on the CPU, then a contiguous block per GPU in index order", so
/// shares become counts laid out in that order.
fn homes(a: &Anatomy, intent: &Intent) -> Result<Vec<Tier>, PlanError> {
    if intent.layers.is_empty() {
        return Err(PlanError::NoLayerShares);
    }
    if intent.layers.iter().any(|s| s.tier == Tier::Disk) {
        return Err(PlanError::LayersOnDisk);
    }
    let n = a.layers as usize;
    // Average bytes of what follows a layer home, to turn Bytes into a count.
    let own: Vec<Subsystem> = intent.subsystems.keys().copied().collect();
    let home_bytes: u64 = a
        .parts
        .iter()
        .filter(|p| p.layer.is_some() && !own.contains(&p.subsystem))
        .map(|p| p.bytes)
        .sum();
    let avg = (home_bytes / a.layers.max(1) as u64).max(1);

    let mut counts: Vec<(Tier, usize, bool)> = Vec::new(); // tier, count, is_rest
    let mut fixed = 0usize;
    for s in &intent.layers {
        let c = match s.amount {
            Amount::Fraction(x) => (x * n as f64).round() as usize,
            Amount::Bytes(b) => ((b as f64) / (avg as f64)).round() as usize,
            Amount::Rest => 0,
        };
        fixed += c;
        counts.push((s.tier, c, s.amount == Amount::Rest));
    }
    let rests = counts.iter().filter(|c| c.2).count();
    if rests > 1 {
        return Err(PlanError::MoreThanOneRest("layers".into()));
    }
    if fixed > n {
        // Rounding of fractions that sum to 1 can overshoot by a layer per share.
        if rests == 0 && fixed - n <= counts.len() {
            trim(&mut counts, fixed - n);
        } else {
            return Err(PlanError::OverCommitted {
                what: "layers".into(),
            });
        }
    } else if rests == 1 {
        let r = counts.iter_mut().find(|c| c.2).unwrap();
        r.1 = n - fixed;
    } else if fixed < n {
        if n - fixed <= counts.len() {
            // rounding slack goes to the biggest share
            counts.iter_mut().max_by_key(|c| c.1).unwrap().1 += n - fixed;
        } else {
            return Err(PlanError::Uncovered {
                what: "layers".into(),
                placed: fixed as u64,
                total: n as u64,
            });
        }
    }

    // Merge duplicate tiers, then lay out: RAM first, then GPUs by index.
    let mut by_tier: BTreeMap<Tier, usize> = BTreeMap::new();
    for (t, c, _) in counts {
        *by_tier.entry(t).or_insert(0) += c;
    }
    let mut order: Vec<(Tier, usize)> = by_tier.into_iter().collect();
    order.sort_by_key(|(t, _)| match t {
        Tier::Ram => (0, 0),
        Tier::Gpu(i) => (1, *i),
        Tier::Disk => (2, 0),
    });
    let mut out = Vec::with_capacity(n);
    for (t, c) in order {
        out.extend(std::iter::repeat_n(t, c));
    }
    debug_assert_eq!(out.len(), n);
    Ok(out)
}

fn trim(counts: &mut [(Tier, usize, bool)], mut excess: usize) {
    while excess > 0 {
        let c = counts.iter_mut().max_by_key(|c| c.1).unwrap();
        c.1 -= 1;
        excess -= 1;
    }
}

/// Assign each layer's part of a subsystem to a tier. `sizes[l]` is the
/// subsystem's bytes in layer l (0 = layer has none). GPUs are filled first,
/// preferring layers homed on that GPU so no cross-device hop is added, then
/// RAM, then disk; a `Rest` share takes what's left.
fn assign(
    sizes: &[u64],
    shares: &[Share],
    home: &[Tier],
    what: &str,
) -> Result<Vec<Option<Tier>>, PlanError> {
    let total: u64 = sizes.iter().sum();
    let mut out: Vec<Option<Tier>> = vec![None; sizes.len()];
    if total == 0 {
        return Ok(out);
    }
    let rests: Vec<&Share> = shares.iter().filter(|s| s.amount == Amount::Rest).collect();
    if rests.len() > 1 {
        return Err(PlanError::MoreThanOneRest(what.into()));
    }
    let target = |s: &Share| match s.amount {
        Amount::Fraction(x) => (x * total as f64) as u64,
        Amount::Bytes(b) => b,
        Amount::Rest => 0,
    };
    let claimed: u64 = shares.iter().map(target).sum();
    if claimed > total + total / 1000 {
        return Err(PlanError::OverCommitted { what: what.into() });
    }

    let mut fixed: Vec<&Share> = shares.iter().filter(|s| s.amount != Amount::Rest).collect();
    fixed.sort_by_key(|s| match s.tier {
        Tier::Gpu(i) => (0, i),
        Tier::Ram => (1, 0),
        Tier::Disk => (2, 0),
    });
    let mut got: Vec<u64> = vec![0; fixed.len()];
    for (k, s) in fixed.iter().enumerate() {
        let want = target(s);
        let mut cand: Vec<usize> = (0..sizes.len())
            .filter(|&l| sizes[l] > 0 && out[l].is_none())
            .collect();
        cand.sort_by_key(|&l| (home[l] != s.tier, l));
        for l in cand {
            // take a layer if that brings us closer to the target
            if got[k] + sizes[l] / 2 > want {
                continue;
            }
            out[l] = Some(s.tier);
            got[k] += sizes[l];
        }
    }

    let left: Vec<usize> = (0..sizes.len())
        .filter(|&l| sizes[l] > 0 && out[l].is_none())
        .collect();
    if let Some(r) = rests.first() {
        for l in left {
            out[l] = Some(r.tier);
        }
    } else if !left.is_empty() {
        let left_bytes: u64 = left.iter().map(|&l| sizes[l]).sum();
        // Fractions meant to cover everything leave rounding crumbs; anything
        // more than a layer's worth per share is a real gap.
        let slack = sizes.iter().max().copied().unwrap_or(0) * fixed.len() as u64;
        if left_bytes > slack || fixed.is_empty() {
            return Err(PlanError::Uncovered {
                what: what.into(),
                placed: total - left_bytes,
                total,
            });
        }
        for l in left {
            // most under-filled share takes it
            let k = (0..fixed.len())
                .max_by_key(|&k| target(fixed[k]) as i128 - got[k] as i128)
                .unwrap();
            out[l] = Some(fixed[k].tier);
            got[k] += sizes[l];
        }
    }
    Ok(out)
}
