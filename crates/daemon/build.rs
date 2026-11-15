//! Embed the web UI (ui/dist, built with `npm run build` in ui/) so the
//! product stays one binary. Without a build, /ui/ explains how to make one;
//! Rust builds never need Node.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    let dist = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../ui/dist");
    println!("cargo:rerun-if-changed={}", dist.display());
    let mut files = Vec::new();
    if dist.is_dir() {
        walk(&dist, &mut files);
    }
    files.sort();
    let mut out = String::from("pub static ASSETS: &[(&str, &[u8])] = &[\n");
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
        let rel = f.strip_prefix(&dist).unwrap().to_string_lossy().replace('\\', "/");
        out.push_str(&format!("    ({rel:?}, include_bytes!({:?})),\n", f.canonicalize().unwrap()));
    }
    out.push_str("];\n");
    fs::write(PathBuf::from(env::var("OUT_DIR").unwrap()).join("ui_assets.rs"), out).unwrap();
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}
