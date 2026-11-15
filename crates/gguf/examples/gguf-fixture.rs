//! Make a small test fixture from a real GGUF: the same metadata (minus the
//! tokenizer, which is most of the header) and the same tensor table, written
//! as a header-only file. Works on a partial download of just the header:
//!
//!     curl -r 0-33554431 -o head.gguf https://huggingface.co/<repo>/resolve/main/<file>.gguf
//!     cargo run -p aie-gguf --example gguf-fixture -- head.gguf fixture.gguf

use aie_gguf::Gguf;
use aie_gguf::write::Writer;

fn main() {
    let mut args = std::env::args().skip(1);
    let (src, dst) = match (args.next(), args.next()) {
        (Some(s), Some(d)) => (s, d),
        _ => {
            eprintln!("usage: gguf-fixture SOURCE.gguf FIXTURE.gguf");
            std::process::exit(2)
        }
    };
    let g = Gguf::open_header(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let mut w = Writer::new();
    for (k, v) in &g.metadata {
        if !k.starts_with("tokenizer.") {
            w = w.kv(k.clone(), v.clone());
        }
    }
    for t in &g.tensors {
        w = w.tensor(t.name.clone(), t.ty, &t.dims);
    }
    w.write_header_file(&dst)
        .unwrap_or_else(|e| panic!("{dst}: {e}"));
    let out = Gguf::open_header(&dst).expect("fixture must parse");
    assert_eq!(
        out.tensors, g.tensors,
        "tensor table must be preserved exactly"
    );
    eprintln!(
        "{dst}: {} tensors, {} metadata keys, {} bytes",
        out.tensors.len(),
        out.metadata.len(),
        std::fs::metadata(&dst).unwrap().len()
    );
}
