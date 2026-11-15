use aie_gguf::write::Writer;
use aie_gguf::{Error, GgmlType, Gguf, Value};

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("aie-gguf-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn sample() -> Writer {
    Writer::new()
        .str("general.architecture", "qwen3moe")
        .u32("qwen3moe.block_count", 2)
        .kv("tokenizer.ggml.tokens", Value::Array(8, vec![Value::String("a".into()), Value::String("▁b".into())]))
        .kv("some.float", Value::F32(0.5))
        .kv("some.bool", Value::Bool(true))
        .tensor("token_embd.weight", GgmlType::F16, &[64, 10])
        .tensor("blk.0.ffn_up_exps.weight", GgmlType::Q4_K, &[256, 32, 4])
        .tensor("output_norm.weight", GgmlType::F32, &[64])
}

#[test]
fn roundtrip_full_file() {
    let p = tmp("full.gguf");
    sample().write_file(&p).unwrap();
    let g = Gguf::open(&p).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.architecture(), Some("qwen3moe"));
    assert_eq!(g.arch_u64("block_count"), Some(2));
    assert_eq!(g.get("some.float"), Some(&Value::F32(0.5)));
    assert_eq!(g.get("some.bool"), Some(&Value::Bool(true)));
    assert_eq!(g.get("tokenizer.ggml.tokens").unwrap().as_array().unwrap()[1].as_str(), Some("▁b"));
    assert_eq!(g.tensors.len(), 3);

    let e = g.tensor("blk.0.ffn_up_exps.weight").unwrap();
    assert_eq!(e.size, 144 * 32 * 4);
    assert_eq!(e.offset, 64 * 10 * 2); // 1280, already 32-aligned
    assert_eq!(g.data_offset % 32, 0);
    // data section = padded sizes; the file is exactly that long
    let len = std::fs::metadata(&p).unwrap().len();
    assert_eq!(len, g.data_offset + 1280 + 144 * 32 * 4 + 256);
}

#[test]
fn header_only_needs_open_header() {
    let p = tmp("header.gguf");
    sample().write_header_file(&p).unwrap();
    assert!(matches!(Gguf::open(&p), Err(Error::Truncated { .. })));
    let g = Gguf::open_header(&p).unwrap();
    assert_eq!(g.tensors.len(), 3);
}

#[test]
fn custom_alignment_and_data() {
    let p = tmp("align.gguf");
    Writer::new()
        .u32("general.alignment", 64)
        .tensor_data("a", GgmlType::F32, &[3], vec![1; 12])
        .tensor("b", GgmlType::F32, &[5])
        .write_file(&p)
        .unwrap();
    let g = Gguf::open(&p).unwrap();
    assert_eq!(g.alignment, 64);
    assert_eq!(g.data_offset % 64, 0);
    assert_eq!(g.tensor("b").unwrap().offset, 64);
    let bytes = std::fs::read(&p).unwrap();
    let a = g.tensor("a").unwrap();
    assert_eq!(&bytes[g.file_offset(a) as usize..][..12], &[1u8; 12]);
}

#[test]
fn rejects_what_llama_cpp_rejects() {
    let mut h = sample().header().unwrap();
    // bad magic
    let mut bad = h.clone();
    bad[0] = b'X';
    assert!(matches!(Gguf::from_reader(&bad[..]), Err(Error::BadMagic(_))));
    // v1 and too-new versions
    for v in [1u32, 4, 0x0300_0000] {
        h[4..8].copy_from_slice(&v.to_le_bytes());
        assert!(matches!(Gguf::from_reader(&h[..]), Err(Error::UnsupportedVersion(_))), "version {v}");
    }
    // name too long, duplicate name, too many dims, block-size mismatch
    let long = "x".repeat(64);
    for w in [
        Writer::new().tensor(long, GgmlType::F32, &[1]),
        Writer::new().tensor("a", GgmlType::F32, &[1]).tensor("a", GgmlType::F32, &[1]),
        Writer::new().tensor("a", GgmlType::F32, &[1, 1, 1, 1, 1]),
        Writer::new().u32("general.alignment", 24),
    ] {
        let h = w.header().unwrap();
        assert!(matches!(Gguf::from_reader(&h[..]), Err(Error::Malformed(_))));
    }
    assert!(Writer::new().tensor("q", GgmlType::Q4_K, &[100]).header().is_err());
    // truncated header
    let h = sample().header().unwrap();
    assert!(matches!(Gguf::from_reader(&h[..40]), Err(Error::Malformed(_))));
}

#[test]
fn rejects_non_contiguous_offsets() {
    let mut h = Writer::new().tensor("a", GgmlType::F32, &[8]).tensor("b", GgmlType::F32, &[8]).header().unwrap();
    // the last u64 before padding is b's offset (32); make it 64
    let g = Gguf::from_reader(&h[..]).unwrap();
    let pos = (0..h.len() - 8).rev().find(|&i| u64::from_le_bytes(h[i..i + 8].try_into().unwrap()) == 32).unwrap();
    assert_eq!(g.tensor("b").unwrap().offset, 32);
    h[pos..pos + 8].copy_from_slice(&64u64.to_le_bytes());
    assert!(matches!(Gguf::from_reader(&h[..]), Err(Error::Malformed(_))));
}
