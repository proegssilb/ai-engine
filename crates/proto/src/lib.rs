//! The daemon <-> worker gRPC protocol, generated from `proto/`.

#![forbid(unsafe_code)]

#[allow(clippy::all)]
pub mod worker {
    tonic::include_proto!("aie.worker.v1");
}
