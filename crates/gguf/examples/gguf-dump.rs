//! Print a GGUF's header in the same format as the C cross-check against
//! llama.cpp's reader (`crates/gguf/xcheck/run.sh`):
//!
//!     cargo run -p aie-gguf --example gguf-dump -- model.gguf

fn main() {
    let path = std::env::args().nth(1).expect("usage: gguf-dump FILE.gguf");
    let g = aie_gguf::Gguf::open_header(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    });
    println!("version {} alignment {} data_offset {} n_kv {}", g.version, g.alignment, g.data_offset, g.metadata.len());
    for t in &g.tensors {
        let mut ne = [1u64; 4];
        ne[..t.dims.len()].copy_from_slice(&t.dims);
        println!(
            "{} {} [{},{},{},{}] off {} size {}",
            t.name, t.ty, ne[0], ne[1], ne[2], ne[3], t.offset, t.size
        );
    }
}
