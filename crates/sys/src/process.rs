//! Process-level calls: signals, ids, limits, filesystem space.

use crate::{einval, last_error};
use rustix::process::{Pid, Resource, Signal};
use std::io;
use std::path::Path;

/// SIGTERM to the process `pid`. Only a real, positive process id is
/// accepted: 0 and ids above `i32::MAX` fail with EINVAL, because the
/// raw call would treat them as process groups or "everyone".
pub fn terminate(pid: u32) -> io::Result<()> {
    let pid = i32::try_from(pid)
        .ok()
        .and_then(Pid::from_raw)
        .ok_or_else(einval)?;
    rustix::process::kill_process(pid, Signal::TERM)?;
    Ok(())
}

/// The real user id of this process.
pub fn uid() -> u32 {
    rustix::process::getuid().as_raw()
}

/// "No limit" in a [`MemlockLimit`].
pub const UNLIMITED: u64 = u64::MAX;

/// RLIMIT_MEMLOCK in bytes; [`UNLIMITED`] for no limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemlockLimit {
    pub soft: u64,
    pub hard: u64,
}

pub fn memlock_limit() -> MemlockLimit {
    let r = rustix::process::getrlimit(Resource::Memlock);
    MemlockLimit {
        soft: r.current.unwrap_or(UNLIMITED),
        hard: r.maximum.unwrap_or(UNLIMITED),
    }
}

/// Space on a filesystem, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskSpace {
    /// Available to unprivileged users.
    pub free: u64,
    pub total: u64,
}

/// Free and total bytes of the filesystem holding `path` (`statvfs`).
pub fn disk_space(path: &Path) -> io::Result<DiskSpace> {
    let st = rustix::fs::statvfs(path)?;
    Ok(DiskSpace {
        free: st.f_bavail.saturating_mul(st.f_frsize),
        total: st.f_blocks.saturating_mul(st.f_frsize),
    })
}

/// Make SIGPIPE terminate the process (the default action), so writing to
/// a closed pipe (`cmd | head`) ends the program quietly instead of
/// returning EPIPE. Process-wide; call it early in `main`.
pub fn sigpipe_default() -> io::Result<()> {
    // SAFETY: signal(SIGPIPE, SIG_DFL) installs no handler (SIG_DFL is not a
    // function pointer), so no Rust code can run in signal context. SIGPIPE
    // is a valid, catchable signal number. The call is thread-safe; it just
    // replaces the process's disposition for that one signal.
    let prev = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };
    if prev == libc::SIG_ERR {
        Err(last_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminate_refuses_group_like_pids() {
        for pid in [0, u32::MAX, i32::MAX as u32 + 1] {
            assert_eq!(
                terminate(pid).unwrap_err().raw_os_error(),
                Some(libc::EINVAL)
            );
        }
    }

    #[test]
    fn terminates_a_child() {
        use std::os::unix::process::ExitStatusExt;
        let mut c = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        terminate(c.id()).unwrap();
        let st = c.wait().unwrap();
        assert_eq!(st.signal(), Some(libc::SIGTERM));
        // Reaped: the pid is gone (or reused); a nonexistent one gives ESRCH.
    }

    #[test]
    fn terminate_unknown_pid_is_an_error() {
        // pid_max is at most 2^22; this is above it.
        assert_eq!(
            terminate(i32::MAX as u32).unwrap_err().raw_os_error(),
            Some(libc::ESRCH)
        );
    }

    #[test]
    fn uid_matches_proc() {
        let s = std::fs::read_to_string("/proc/self/status").unwrap();
        let line = s.lines().find(|l| l.starts_with("Uid:")).unwrap();
        let real: u32 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
        assert_eq!(uid(), real);
    }

    #[test]
    fn memlock_limit_matches_proc() {
        let l = memlock_limit();
        assert!(l.soft <= l.hard);
        let s = std::fs::read_to_string("/proc/self/limits").unwrap();
        let line = s
            .lines()
            .find(|l| l.starts_with("Max locked memory"))
            .unwrap();
        let soft = line.split_whitespace().nth(3).unwrap();
        match soft {
            "unlimited" => assert_eq!(l.soft, UNLIMITED),
            n => assert_eq!(l.soft, n.parse::<u64>().unwrap()),
        }
    }

    #[test]
    fn disk_space_of_tmp_and_missing() {
        let d = disk_space(Path::new("/")).unwrap();
        assert!(d.total > 0 && d.free <= d.total);
        assert_eq!(
            disk_space(Path::new("/no/such/dir"))
                .unwrap_err()
                .raw_os_error(),
            Some(libc::ENOENT)
        );
    }

    #[test]
    fn sigpipe_default_is_installed() {
        sigpipe_default().unwrap();
        // The disposition is now SIG_DFL: read it back through /proc (SigCgt/SigIgn).
        let s = std::fs::read_to_string("/proc/self/status").unwrap();
        let mask = |k: &str| {
            let l = s.lines().find(|l| l.starts_with(k)).unwrap();
            u64::from_str_radix(l.split_whitespace().nth(1).unwrap(), 16).unwrap()
        };
        let bit = 1u64 << (libc::SIGPIPE - 1);
        assert_eq!(mask("SigIgn:") & bit, 0);
        assert_eq!(mask("SigCgt:") & bit, 0);
    }
}
