//! Safe wrappers for the raw Linux calls the workspace needs.
//!
//! This crate and `aie-llama` (the llama.cpp boundary) are the only places
//! allowed to use `unsafe`; every other crate has `#![forbid(unsafe_code)]`.
//! Every `unsafe` block here carries a `SAFETY:` comment, enforced by the
//! lints below.
//!
//! Where a raw call is only sound for some arguments, the wrapper either
//! takes a type that makes the bad arguments unrepresentable (an owned
//! [`AnonMap`], a non-zero pid) or checks them and returns an error. The one
//! `unsafe fn` is [`AnonMap::from_raw`], whose contract cannot be checked.
//!
//! Errors are `std::io::Error`; `raw_os_error()` gives the errno.

#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]

mod map;
mod mem;
mod process;
mod sched;

pub use map::{Advice, AnonMap};
pub use mem::{mincore, move_pages_status, page_size};
pub use process::{
    DiskSpace, MemlockLimit, UNLIMITED, disk_space, memlock_limit, sigpipe_default, terminate, uid,
};
pub use sched::{
    MAX_CPUS, MAX_NODES, get_affinity, get_mempolicy, set_affinity, set_mempolicy_preferred,
};

use std::io;

/// EINVAL, for arguments rejected before any system call is made.
pub(crate) fn einval() -> io::Error {
    io::Error::from_raw_os_error(libc::EINVAL)
}

/// The errno of the libc call that just failed. Call it immediately after
/// the failing call, with nothing in between that could set errno.
pub(crate) fn last_error() -> io::Error {
    io::Error::last_os_error()
}

/// Tests that create mappings and then check that one is gone take this
/// lock, so another test's mmap can't reuse the address in between.
#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}
