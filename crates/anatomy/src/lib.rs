//! Model anatomy: a model's tensors grouped into the subsystems the placement
//! UI shows (attention, routed experts, embeddings, ...), with their sizes per
//! layer, plus the shape needed to size the KV cache.
//!
//! Grouping goes by llama.cpp's tensor naming, which is shared across
//! architectures (`token_embd`, `blk.N.attn_*`, `blk.N.ffn_*_exps`, ...), so it
//! works beyond the MVP's Qwen3 families. [`Anatomy::is_supported`] says
//! whether the architecture is one we've checked.

use aie_gguf::{GgmlType, Gguf};
use std::collections::BTreeMap;
use std::fmt;

/// Architectures whose anatomy has been checked against real files.
pub const SUPPORTED_ARCHITECTURES: &[&str] = &["qwen3", "qwen3moe"];

/// A placeable part of a model.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Subsystem {
    /// Token embeddings (`token_embd`).
    Embeddings,
    /// Attention projections (`attn_q/k/v/output`).
    Attention,
    /// Normalisation weights inside layers (`attn_norm`, `ffn_norm`, `attn_q_norm`, ...).
    Norms,
    /// The MoE router (`ffn_gate_inp`, `exp_probs_b`).
    Router,
    /// Routed experts (`ffn_*_exps`).
    RoutedExperts,
    /// Shared experts (`ffn_*_shexp`).
    SharedExperts,
    /// The feed-forward block of a dense layer (`ffn_gate/up/down`).
    DenseFfn,
    /// Final norm and output projection (`output_norm`, `output`).
    OutputHead,
    /// Anything else (rope tables, unknown tensors).
    Other,
}

impl Subsystem {
    pub const ALL: [Subsystem; 9] = [
        Subsystem::Embeddings,
        Subsystem::Attention,
        Subsystem::Norms,
        Subsystem::Router,
        Subsystem::RoutedExperts,
        Subsystem::SharedExperts,
        Subsystem::DenseFfn,
        Subsystem::OutputHead,
        Subsystem::Other,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Subsystem::Embeddings => "embeddings",
            Subsystem::Attention => "attention",
            Subsystem::Norms => "norms",
            Subsystem::Router => "router",
            Subsystem::RoutedExperts => "routed experts",
            Subsystem::SharedExperts => "shared experts",
            Subsystem::DenseFfn => "dense ffn",
            Subsystem::OutputHead => "output head",
            Subsystem::Other => "other",
        }
    }
}

impl fmt::Display for Subsystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Which subsystem a tensor belongs to, and its layer if it's inside one.
pub fn classify(name: &str) -> (Subsystem, Option<u32>) {
    if let Some(rest) = name.strip_prefix("blk.") {
        let mut parts = rest.splitn(2, '.');
        let layer = parts.next().and_then(|l| l.parse().ok());
        let base = parts.next().unwrap_or("");
        let s = if base.contains("_exps") {
            Subsystem::RoutedExperts
        } else if base.contains("_shexp") {
            Subsystem::SharedExperts
        } else if base.starts_with("ffn_gate_inp") || base.starts_with("exp_probs") {
            Subsystem::Router
        } else if base.contains("norm") {
            Subsystem::Norms
        } else if base.starts_with("attn_") {
            Subsystem::Attention
        } else if base.starts_with("ffn_") {
            Subsystem::DenseFfn
        } else {
            Subsystem::Other
        };
        return (s, layer);
    }
    let s = if name.starts_with("token_embd") {
        Subsystem::Embeddings
    } else if name.starts_with("output") {
        Subsystem::OutputHead
    } else {
        Subsystem::Other
    };
    (s, None)
}

/// One tensor, placed in the anatomy.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    pub name: String,
    pub subsystem: Subsystem,
    pub layer: Option<u32>,
    pub ty: GgmlType,
    pub bytes: u64,
}

/// What the KV cache's size depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KvShape {
    pub layers: u32,
    pub kv_heads: u32,
    pub key_len: u32,
    pub value_len: u32,
}

impl KvShape {
    /// Bytes of KV cache per token of context, for a cache element type of
    /// `elem_bytes` (2 for f16, the default).
    pub fn bytes_per_token(&self, elem_bytes: u64) -> u64 {
        u64::from(self.layers)
            * u64::from(self.kv_heads)
            * u64::from(self.key_len + self.value_len)
            * elem_bytes
    }

    /// Bytes of KV cache for a context of `n_ctx` tokens.
    pub fn bytes(&self, n_ctx: u64, elem_bytes: u64) -> u64 {
        self.bytes_per_token(elem_bytes) * n_ctx
    }
}

/// Mixture-of-experts shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MoeShape {
    pub experts: u32,
    pub experts_used: u32,
}

/// Errors building an anatomy.
#[derive(Debug, PartialEq)]
pub enum Error {
    NoArchitecture,
    MissingKey(String),
    /// A tensor names a layer at or beyond `block_count`.
    LayerOutOfRange {
        tensor: String,
        layer: u32,
        layers: u32,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoArchitecture => write!(f, "general.architecture is missing"),
            Error::MissingKey(k) => write!(f, "metadata key {k} is missing"),
            Error::LayerOutOfRange {
                tensor,
                layer,
                layers,
            } => {
                write!(
                    f,
                    "tensor {tensor} is in layer {layer}, but the model has {layers} layers"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// A model broken into placeable parts.
#[derive(Clone, Debug)]
pub struct Anatomy {
    pub architecture: String,
    pub layers: u32,
    pub embedding_len: Option<u64>,
    pub context_train: Option<u64>,
    pub moe: Option<MoeShape>,
    /// `None` if the metadata doesn't describe attention heads.
    pub kv: Option<KvShape>,
    /// Every tensor, in file order.
    pub parts: Vec<Part>,
}

impl Anatomy {
    pub fn from_gguf(g: &Gguf) -> Result<Anatomy, Error> {
        let arch = g.architecture().ok_or(Error::NoArchitecture)?.to_string();
        let key = |s: &str| g.arch_u64(s);
        let layers = key("block_count")
            .ok_or_else(|| Error::MissingKey(format!("{arch}.block_count")))?
            as u32;

        let moe = match (key("expert_count"), key("expert_used_count")) {
            (Some(e), Some(u)) if e > 0 => Some(MoeShape {
                experts: e as u32,
                experts_used: u as u32,
            }),
            _ => None,
        };

        let kv = key("attention.head_count").map(|heads| {
            let kv_heads = key("attention.head_count_kv").unwrap_or(heads);
            let head_dim = key("embedding_length").map(|e| e / heads.max(1));
            KvShape {
                layers,
                kv_heads: kv_heads as u32,
                key_len: key("attention.key_length").or(head_dim).unwrap_or(0) as u32,
                value_len: key("attention.value_length").or(head_dim).unwrap_or(0) as u32,
            }
        });

        let mut parts = Vec::with_capacity(g.tensors.len());
        for t in &g.tensors {
            let (subsystem, layer) = classify(&t.name);
            if let Some(l) = layer
                && l >= layers
            {
                return Err(Error::LayerOutOfRange {
                    tensor: t.name.clone(),
                    layer: l,
                    layers,
                });
            }
            parts.push(Part {
                name: t.name.clone(),
                subsystem,
                layer,
                ty: t.ty,
                bytes: t.size,
            });
        }

        Ok(Anatomy {
            architecture: arch,
            layers,
            embedding_len: key("embedding_length"),
            context_train: key("context_length"),
            moe,
            kv,
            parts,
        })
    }

    pub fn is_supported(&self) -> bool {
        SUPPORTED_ARCHITECTURES.contains(&self.architecture.as_str())
    }

    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Total weight bytes.
    pub fn total_bytes(&self) -> u64 {
        self.parts.iter().map(|p| p.bytes).sum()
    }

    /// Bytes per subsystem (subsystems with no tensors are left out).
    pub fn subsystem_bytes(&self) -> BTreeMap<Subsystem, u64> {
        let mut m = BTreeMap::new();
        for p in &self.parts {
            *m.entry(p.subsystem).or_insert(0) += p.bytes;
        }
        m
    }

    /// Bytes of one subsystem in each layer (index = layer). Zero where a layer
    /// has none of it.
    pub fn layer_bytes(&self, s: Subsystem) -> Vec<u64> {
        let mut v = vec![0; self.layers as usize];
        for p in self.parts.iter().filter(|p| p.subsystem == s) {
            if let Some(l) = p.layer {
                v[l as usize] += p.bytes;
            }
        }
        v
    }

    /// Bytes of one expert in one layer: the layer's routed-expert bytes
    /// divided by the expert count. `None` for dense models.
    pub fn bytes_per_expert(&self, layer: u32) -> Option<u64> {
        let moe = self.moe?;
        let b = *self
            .layer_bytes(Subsystem::RoutedExperts)
            .get(layer as usize)?;
        Some(b / u64::from(moe.experts))
    }

    /// The parts of one subsystem.
    pub fn parts_of(&self, s: Subsystem) -> impl Iterator<Item = &Part> {
        self.parts.iter().filter(move |p| p.subsystem == s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification() {
        use Subsystem::*;
        let cases = [
            ("token_embd.weight", Embeddings, None),
            ("blk.3.attn_q.weight", Attention, Some(3)),
            ("blk.3.attn_output.weight", Attention, Some(3)),
            ("blk.3.attn_q_norm.weight", Norms, Some(3)),
            ("blk.3.attn_norm.weight", Norms, Some(3)),
            ("blk.12.ffn_norm.weight", Norms, Some(12)),
            ("blk.0.ffn_gate_inp.weight", Router, Some(0)),
            ("blk.0.exp_probs_b.bias", Router, Some(0)),
            ("blk.0.ffn_up_exps.weight", RoutedExperts, Some(0)),
            ("blk.0.ffn_gate_inp_shexp.weight", SharedExperts, Some(0)),
            ("blk.0.ffn_down_shexp.weight", SharedExperts, Some(0)),
            ("blk.0.ffn_gate.weight", DenseFfn, Some(0)),
            ("output.weight", OutputHead, None),
            ("output_norm.weight", OutputHead, None),
            ("rope_freqs.weight", Other, None),
        ];
        for (name, s, l) in cases {
            assert_eq!(classify(name), (s, l), "{name}");
        }
    }

    #[test]
    fn kv_bytes() {
        let kv = KvShape {
            layers: 48,
            kv_heads: 4,
            key_len: 128,
            value_len: 128,
        };
        assert_eq!(kv.bytes_per_token(2), 48 * 4 * 256 * 2);
        assert_eq!(kv.bytes(4096, 2), 384 << 20);
    }
}
