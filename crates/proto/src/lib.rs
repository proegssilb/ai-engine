//! The daemon <-> worker gRPC protocol, generated from `proto/`.

#[allow(clippy::all)]
pub mod worker {
    tonic::include_proto!("aie.worker.v1");
}
