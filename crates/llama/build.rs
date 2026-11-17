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
    // Rebuild when the vendored sources change at all (a submodule bump), not
    // only llama.h: the bindings and the callback signatures in pinned.rs
    // depend on ggml's internal headers too. Directories are scanned recursively.
    for d in ["include", "src", "common", "ggml/include", "ggml/src"] {
        println!("cargo:rerun-if-changed={}", root.join(d).display());
    }
    println!("cargo:rerun-if-env-changed=CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=AIE_CPU_TARGET");
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
        .profile("Release")
        .build_target("llama-common");
    for (k, v) in cpu_target_flags(&env::var("AIE_CPU_TARGET").unwrap_or_default()) {
        cfg.define(k, v);
    }
    if cuda {
        let arch = env::var("CUDA_ARCHITECTURES").unwrap_or_else(|_| "70".into());
        cfg.define("GGML_CUDA", "ON")
            .define("CMAKE_CUDA_ARCHITECTURES", arch);
    }
    let dst = cfg.build();
    let build = dst.join("build");

    // Link the static libraries in dependency order (dependents first).
    // rustc-link-lib, unlike rustc-link-arg, carries through to every crate
    // that depends on this one.
    let mut libs = Vec::new();
    find_static_libs(&build, &mut libs);
    let mut dirs: Vec<PathBuf> = libs
        .iter()
        .map(|l| l.parent().unwrap().to_path_buf())
        .collect();
    dirs.sort();
    dirs.dedup();
    for d in &dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    let have = |n: &str| {
        libs.iter()
            .any(|l| l.file_name().unwrap() == format!("lib{n}.a").as_str())
    };
    let order = [
        "llama-common",
        "llama-common-base",
        "cpp-httplib",
        "llama",
        "ggml",
        "ggml-cpu",
        "ggml-cuda",
        "ggml-base",
    ];
    for n in order {
        if have(n) {
            println!("cargo:rustc-link-lib=static={n}");
        }
    }
    for l in &libs {
        let n = l
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_start_matches("lib");
        assert!(
            order.contains(&n),
            "llama.cpp built an unexpected library {n}; add it to the link order in build.rs"
        );
    }
    for l in ["stdc++", "m", "pthread", "dl"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
    if cuda {
        for d in ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"] {
            println!("cargo:rustc-link-search=native={d}");
        }
        for l in ["cudart", "cublas", "cublasLt", "cuda"] {
            println!("cargo:rustc-link-lib=dylib={l}");
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
        // llama.cpp is C++ behind a C API, and several of its functions can
        // throw (std::bad_alloc, std::out_of_range). Unwinding out of an
        // `extern "C"` import is undefined behaviour in Rust; with "C-unwind"
        // it is defined: the exception propagates and ends the process.
        .override_abi(bindgen::Abi::CUnwind, "(llama|ggml|gguf)_.*")
        .generate()
        .expect("bindgen failed");
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .unwrap();
}

/// ggml's CPU options for a build target, chosen with `AIE_CPU_TARGET`.
///
/// `native` (the default) tunes to the build machine: fastest, but the binary
/// only runs on CPUs with the same features. Releases use the x86-64
/// microarchitecture levels instead, one binary each, because llama.cpp's
/// alternative (GGML_CPU_ALL_VARIANTS) needs dynamically loaded backends,
/// which conflicts with shipping one static binary. Every option is set
/// explicitly so a level never depends on ggml's defaults.
fn cpu_target_flags(target: &str) -> Vec<(&'static str, &'static str)> {
    let onoff = |on: bool| if on { "ON" } else { "OFF" };
    // (sse4.2, avx, avx2 + bmi2 + fma + f16c, avx512 f/cd/vl/dq/bw)
    let (sse42, avx, avx2, avx512) = match target {
        "" | "native" => return vec![("GGML_NATIVE", "ON")],
        "x86-64-v2" => (true, false, false, false),
        "x86-64-v3" => (true, true, true, false),
        "x86-64-v4" => (true, true, true, true),
        other => {
            panic!("AIE_CPU_TARGET={other}: expected native, x86-64-v2, x86-64-v3 or x86-64-v4")
        }
    };
    assert!(
        env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|a| a == "x86_64"),
        "AIE_CPU_TARGET={target} only applies to x86-64 builds"
    );
    vec![
        ("GGML_NATIVE", "OFF"),
        ("GGML_SSE42", onoff(sse42)),
        ("GGML_AVX", onoff(avx)),
        ("GGML_AVX2", onoff(avx2)),
        ("GGML_BMI2", onoff(avx2)),
        ("GGML_FMA", onoff(avx2)),
        ("GGML_F16C", onoff(avx2)),
        ("GGML_AVX512", onoff(avx512)),
    ]
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
