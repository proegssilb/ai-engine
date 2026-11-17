//! CPU affinity and NUMA memory policy.

use crate::{einval, last_error};
use rustix::process::Pid;
use rustix::thread::{CpuSet, sched_getaffinity, sched_setaffinity};
use std::io;

/// CPUs a affinity mask can name (glibc's `CPU_SETSIZE`).
pub const MAX_CPUS: usize = CpuSet::MAX_CPU;
/// NUMA nodes a node mask can name: 16 words of 64 bits.
pub const MAX_NODES: usize = MASK_WORDS * 64;

const MASK_WORDS: usize = 16;

fn pid_of(tid: u32) -> io::Result<Option<Pid>> {
    // Thread 0 is the calling thread. Anything that isn't a positive i32 is
    // rejected instead of being reinterpreted (a negative pid would address a
    // process group in some calls).
    match i32::try_from(tid) {
        Ok(0) => Ok(None),
        Ok(t) => Ok(Pid::from_raw(t)),
        Err(_) => Err(einval()),
    }
}

/// CPUs that thread `tid` (0 = the calling thread) may run on, ascending.
pub fn get_affinity(tid: u32) -> io::Result<Vec<u32>> {
    let set = sched_getaffinity(pid_of(tid)?)?;
    Ok((0..MAX_CPUS)
        .filter(|c| set.is_set(*c))
        .map(|c| c as u32)
        .collect())
}

/// Restrict thread `tid` (0 = the calling thread) to `cpus`. CPU numbers
/// of [`MAX_CPUS`] or more are ignored; an empty set (or none the thread's
/// cpuset allows) fails with EINVAL.
pub fn set_affinity(tid: u32, cpus: &[u32]) -> io::Result<()> {
    let mut set = CpuSet::new();
    for c in cpus {
        if (*c as usize) < MAX_CPUS {
            set.set(*c as usize);
        }
    }
    sched_setaffinity(pid_of(tid)?, &set)?;
    Ok(())
}

pub(crate) fn node_mask(node: u32) -> io::Result<[libc::c_ulong; MASK_WORDS]> {
    if node as usize >= MAX_NODES {
        return Err(einval());
    }
    let mut m = [0; MASK_WORDS];
    m[(node / 64) as usize] = 1 << (node % 64);
    Ok(m)
}

/// MPOL_PREFERRED toward `node` for the calling thread; threads it creates
/// afterwards inherit the policy.
pub fn set_mempolicy_preferred(node: u32) -> io::Result<()> {
    const MPOL_PREFERRED: libc::c_long = 1;
    let mask = node_mask(node)?;
    // SAFETY: set_mempolicy(mode, nodemask, maxnode): `mask` is a local array
    // of MASK_WORDS words, `maxnode` is its bit count, and it outlives the
    // call (the kernel copies it). The call changes the calling thread's
    // allocation policy only; it touches no existing memory.
    let r = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            MPOL_PREFERRED,
            mask.as_ptr(),
            (mask.len() * 64) as libc::c_ulong,
        )
    };
    if r == 0 { Ok(()) } else { Err(last_error()) }
}

/// The calling thread's memory policy: (mode, lowest node in its mask).
/// Modes are the kernel's MPOL_* values (0 default, 1 preferred, 2 bind, ...).
pub fn get_mempolicy() -> io::Result<(i32, Option<u32>)> {
    let mut mode: libc::c_int = -1;
    let mut mask = [0 as libc::c_ulong; MASK_WORDS];
    // SAFETY: get_mempolicy(mode, nodemask, maxnode, addr, flags) with
    // addr = 0 and flags = 0 asks for the calling thread's policy. `mode` is
    // a valid, exclusively borrowed c_int and `mask` an array of MASK_WORDS
    // words with `maxnode` = its bit count; the kernel writes at most that
    // many bits into them before returning.
    let r = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut mode as *mut libc::c_int,
            mask.as_mut_ptr(),
            (mask.len() * 64) as libc::c_ulong,
            0usize,
            0usize,
        )
    };
    if r != 0 {
        return Err(last_error());
    }
    let node = (0..MAX_NODES as u32).find(|n| mask[(*n / 64) as usize] >> (n % 64) & 1 == 1);
    Ok((mode, node))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Own thread, so changed affinity or policy can't leak into other tests.
    fn isolated(f: impl FnOnce() + Send + 'static) {
        std::thread::spawn(f).join().unwrap();
    }

    #[test]
    fn affinity_round_trip() {
        isolated(|| {
            let all = get_affinity(0).unwrap();
            assert!(!all.is_empty());
            let one = vec![all[0]];
            set_affinity(0, &one).unwrap();
            assert_eq!(get_affinity(0).unwrap(), one);
            // Out-of-range CPUs are ignored; with nothing left the kernel refuses.
            assert!(set_affinity(0, &[MAX_CPUS as u32 + 5]).is_err());
            assert_eq!(get_affinity(0).unwrap(), one);
            set_affinity(0, &all).unwrap();
            assert_eq!(get_affinity(0).unwrap(), all);
        });
    }

    #[test]
    fn affinity_by_thread_id() {
        isolated(|| {
            let tid = std::fs::read_link("/proc/thread-self")
                .unwrap()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .parse::<u32>()
                .unwrap();
            assert_eq!(get_affinity(tid).unwrap(), get_affinity(0).unwrap());
        });
        assert!(get_affinity(u32::MAX).is_err());
        // No such thread.
        assert_eq!(
            get_affinity(i32::MAX as u32).unwrap_err().raw_os_error(),
            Some(libc::ESRCH)
        );
    }

    #[test]
    fn mempolicy_round_trip() {
        isolated(|| {
            let (mode, _) = get_mempolicy().unwrap();
            assert_eq!(mode, 0, "fresh thread starts with the default policy");
            match set_mempolicy_preferred(0) {
                Ok(()) => assert_eq!(get_mempolicy().unwrap(), (1, Some(0))),
                // Kernel without NUMA support, or a sandbox filtering it.
                Err(e) => assert!(
                    matches!(
                        e.raw_os_error(),
                        Some(libc::ENOSYS | libc::EPERM | libc::EINVAL)
                    ),
                    "{e}"
                ),
            }
            assert!(set_mempolicy_preferred(MAX_NODES as u32).is_err());
        });
    }

    #[test]
    fn node_masks() {
        assert_eq!(node_mask(0).unwrap()[0], 1);
        assert_eq!(node_mask(65).unwrap()[1], 2);
        assert!(node_mask(MAX_NODES as u32).is_err());
    }
}
