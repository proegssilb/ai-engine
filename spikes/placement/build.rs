use std::{env, path::PathBuf};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../vendor/llama.cpp");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed={}", root.join("include/llama.h").display());

    let dst = cmake::Config::new(&root)
        .define("BUILD_SHARED_LIBS", "OFF")
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
        .build_target("llama")
        .build();

    let b = dst.join("build");
    for dir in ["src", "ggml/src"] {
        println!("cargo:rustc-link-search=native={}", b.join(dir).display());
    }
    for lib in ["llama", "ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", root.join("include").display()))
        .clang_arg(format!("-I{}", root.join("ggml/include").display()))
        .clang_arg(format!("-I{}", root.join("ggml/src").display()))
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .derive_default(true)
        .generate()
        .expect("bindgen failed");
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .unwrap();
}
