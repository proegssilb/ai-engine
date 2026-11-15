fn main() {
    println!("cargo:rerun-if-changed=proto");
    tonic_prost_build::configure()
        .compile_protos(&["proto/aie/worker/v1/worker.proto"], &["proto"])
        .expect("protoc failed; is protobuf-compiler installed?");
}
