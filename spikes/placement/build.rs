use std::{env, path::PathBuf};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../vendor/llama.cpp");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", root.join("include/llama.h").display());
    println!("cargo:rerun-if-env-changed=CUDA_ARCHITECTURES");

    let mut cfg = cmake::Config::new(&root);
    cfg.define("BUILD_SHARED_LIBS", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_NATIVE", "ON")
        .profile("Release")
        .build_target("llama");
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        // V100 = 70. Override with e.g. CUDA_ARCHITECTURES="70;86".
        let arch = env::var("CUDA_ARCHITECTURES").unwrap_or_else(|_| "70".into());
        cfg.define("GGML_CUDA", "ON").define("CMAKE_CUDA_ARCHITECTURES", arch);
    }
    let dst = cfg.build();

    // Shared libs land in build/bin; embed that path so the binary runs in place.
    let lib = dst.join("build/bin");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    for l in ["llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }

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
