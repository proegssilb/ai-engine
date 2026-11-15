//! Checks against real model headers in `fixtures/`. The expected sizes for
//! Qwen3-30B-A3B come from the GPU placement spike, where llama.cpp reported
//! the same totals per buffer (spikes/placement/README.md).

use aie_anatomy::{Anatomy, Error, KvShape, MoeShape, Subsystem};
use aie_gguf::write::Writer;
use aie_gguf::{GgmlType, Gguf};

const MIB: f64 = 1048576.0;

fn load(name: &str) -> Anatomy {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name);
    let g = Gguf::open_header(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Anatomy::from_gguf(&g).unwrap()
}

fn mib(b: u64) -> f64 {
    (b as f64 / MIB * 10.0).round() / 10.0
}

#[test]
fn qwen3_30b_a3b() {
    let a = load("qwen3-30b-a3b-q4_k_m.header.gguf");
    assert_eq!(a.architecture, "qwen3moe");
    assert!(a.is_supported() && a.is_moe());
    assert_eq!(a.layers, 48);
    assert_eq!(
        a.moe,
        Some(MoeShape {
            experts: 128,
            experts_used: 8
        })
    );
    assert_eq!(
        a.kv,
        Some(KvShape {
            layers: 48,
            kv_heads: 4,
            key_len: 128,
            value_len: 128
        })
    );
    assert_eq!(a.context_train, Some(40960));

    let s = a.subsystem_bytes();
    // Totals llama.cpp reported on the GPU box for the same file.
    assert_eq!(mib(s[&Subsystem::RoutedExperts]), 16740.0);
    assert_eq!(mib(s[&Subsystem::Embeddings]), 166.9);
    assert_eq!(mib(s[&Subsystem::OutputHead]), 243.4);
    assert_eq!(mib(s[&Subsystem::Attention]), 492.2);
    assert_eq!(mib(s[&Subsystem::Router]), 48.0);
    assert!(!s.contains_key(&Subsystem::DenseFfn));
    assert!(!s.contains_key(&Subsystem::Other));
    assert_eq!(a.parts.len(), 579);
    assert_eq!(a.total_bytes(), s.values().sum::<u64>());

    // Experts aren't uniform: Q4_K_M keeps half the layers' down projections at q6_K.
    // Each [2048, 768, 128] expert tensor is 108 MiB in q4_K (8 blocks x 144 B x 768 x 128);
    // a q6_K down projection [768, 2048, 128] is 157.5 MiB (3 x 210 B x 2048 x 128).
    // So a layer is 3 x 108 = 324 MiB or 108 + 108 + 157.5 = 373.5 MiB, and
    // 24 x 324 + 24 x 373.5 = 16740 MiB, the total above.
    let per_layer = a.layer_bytes(Subsystem::RoutedExperts);
    assert_eq!(per_layer.len(), 48);
    let (min, max) = (
        *per_layer.iter().min().unwrap(),
        *per_layer.iter().max().unwrap(),
    );
    assert_eq!((mib(min), mib(max)), (324.0, 373.5));
    assert_eq!(per_layer.iter().filter(|&&b| b == max).count(), 24);
    assert_eq!(a.bytes_per_expert(0), Some(per_layer[0] / 128));
}

#[test]
fn qwen3_8b() {
    let a = load("qwen3-8b-q4_k_m.header.gguf");
    assert_eq!(a.architecture, "qwen3");
    assert!(a.is_supported() && !a.is_moe());
    assert_eq!(a.layers, 36);
    assert_eq!(
        a.kv,
        Some(KvShape {
            layers: 36,
            kv_heads: 8,
            key_len: 128,
            value_len: 128
        })
    );
    let s = a.subsystem_bytes();
    assert!(s.contains_key(&Subsystem::DenseFfn));
    assert!(!s.contains_key(&Subsystem::RoutedExperts));
    assert_eq!(a.bytes_per_expert(0), None);
    assert_eq!(
        a.layer_bytes(Subsystem::Attention)
            .iter()
            .filter(|&&b| b == 0)
            .count(),
        0
    );
}

#[test]
fn errors() {
    let dir = std::env::temp_dir().join(format!("aie-anatomy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("bad.gguf");

    Writer::new().write_header_file(&p).unwrap();
    assert_eq!(
        Anatomy::from_gguf(&Gguf::open_header(&p).unwrap()).unwrap_err(),
        Error::NoArchitecture
    );

    Writer::new()
        .str("general.architecture", "x")
        .write_header_file(&p)
        .unwrap();
    assert!(matches!(
        Anatomy::from_gguf(&Gguf::open_header(&p).unwrap()),
        Err(Error::MissingKey(_))
    ));

    Writer::new()
        .str("general.architecture", "x")
        .u32("x.block_count", 2)
        .tensor("blk.2.attn_q.weight", GgmlType::F32, &[4])
        .write_header_file(&p)
        .unwrap();
    let a = Anatomy::from_gguf(&Gguf::open_header(&p).unwrap());
    assert!(matches!(
        a,
        Err(Error::LayerOutOfRange {
            layer: 2,
            layers: 2,
            ..
        })
    ));

    // Unknown architectures still get an anatomy, flagged as unsupported.
    Writer::new()
        .str("general.architecture", "x")
        .u32("x.block_count", 1)
        .write_header_file(&p)
        .unwrap();
    let a = Anatomy::from_gguf(&Gguf::open_header(&p).unwrap()).unwrap();
    assert!(!a.is_supported());
    assert_eq!(a.kv, None);
}
