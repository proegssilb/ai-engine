//! Build llama.cpp (pinned at vendor/llama.cpp) as static libraries, link
//! them into this crate, compile the fit shim, and generate bindings.
//!
//! Static so the product ships as one binary. With the `cuda` feature the CUDA
//! runtime and cuBLAS stay dynamic: the driver is per-machine anyway, and the
//! pinned buffer's cudaHostRegister must find the same runtime llama.cpp uses.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest
        .join("../../vendor/llama.cpp")
        .canonicalize()
        .expect("vendor/llama.cpp missing: git submodule update --init");
    for f in ["wrapper.h", "shim/fit.cpp"] {
        println!("cargo:rerun-if-changed={f}");
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/llama.h").display()
    );
    println!("cargo:rerun-if-env-changed=CUDA_ARCHITECTURES");
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();

    let mut cfg = cmake::Config::new(&root);
    cfg.define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_COMMON", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("LLAMA_SUBPROCESS", "OFF")
        .define("LLAMA_LLGUIDANCE", "OFF")
        .define("GGML_OPENMP", "OFF")
        // Tuned to the build machine. Release builds for other CPUs need
        // GGML_NATIVE=OFF plus per-variant CPU backends; see crates/llama/README.md.
        .define("GGML_NATIVE", "ON")
        .profile("Release")
        .build_target("llama-common");
    if cuda {
        let arch = env::var("CUDA_ARCHITECTURES").unwrap_or_else(|_| "70".into());
        cfg.define("GGML_CUDA", "ON")
            .define("CMAKE_CUDA_ARCHITECTURES", arch);
    }
    let dst = cfg.build();
    let build = dst.join("build");

    // Link every static library the build produced, as one group so their
    // mutual references resolve regardless of order.
    let mut libs = Vec::new();
    find_static_libs(&build, &mut libs);
    libs.sort();
    assert!(
        libs.iter().any(|l| l.file_name().unwrap() == "libllama.a"),
        "libllama.a not built"
    );
    let mut dirs: Vec<PathBuf> = libs
        .iter()
        .map(|l| l.parent().unwrap().to_path_buf())
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    for l in &libs {
        println!("cargo:rustc-link-arg={}", l.display());
    }
    println!("cargo:rustc-link-arg=-Wl,--end-group");
    for l in ["stdc++", "m", "pthread", "dl"] {
        println!("cargo:rustc-link-arg=-l{l}");
    }
    if cuda {
        for d in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
            println!("cargo:rustc-link-search=native={d}");
        }
        for l in ["cudart", "cublas", "cublasLt", "cuda"] {
            println!("cargo:rustc-link-arg=-l{l}");
        }
    }

    // The fit shim (C++, the D3 exception).
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("shim/fit.cpp")
        .include(root.join("include"))
        .include(root.join("common"))
        .include(root.join("ggml/include"))
        .include(root.join("vendor"))
        .warnings(false)
        .compile("aie_fit_shim");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", root.join("include").display()))
        .clang_arg(format!("-I{}", root.join("ggml/include").display()))
        .clang_arg(format!("-I{}", root.join("ggml/src").display()))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .derive_default(true)
        .generate()
        .expect("bindgen failed");
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .unwrap();
}

fn find_static_libs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            find_static_libs(&p, out);
        } else if p.extension().is_some_and(|x| x == "a") {
            out.push(p);
        }
    }
}
