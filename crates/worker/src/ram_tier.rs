//! Is the RAM tier staying in RAM? (spikes/memory)
//!
//! Two layers. OS-level holds are the guarantee and the backstop: no swap on
//! the machine, the buffer locked (`mlock`), or swap forbidden for this
//! process's cgroup (`memory.swap.max = 0`), or reserved huge pages, which are
//! never swapped. Driver features are used when available, for speed, and
//! are reported per driver (CUDA registration for the MVP): registered pages
//! are also kept in RAM, but that's the driver's doing, so it's listed
//! separately from the OS-level backstop.
//!
//! When nothing holds it, the status says so and how to fix it for the way
//! the engine is running (systemd service, user session, container, shell).

use aie_llama::residency;
use serde::Serialize;

/// One thing that keeps (or tried to keep) the RAM tier in RAM.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Mechanism {
    /// "os" (the guaranteed backstop) or a driver name ("cuda").
    pub layer: &'static str,
    pub name: String,
    pub ok: bool,
    /// What it means here, or why it failed.
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RamTier {
    /// Bytes of the RAM tier (0 = this model has nothing in it).
    pub bytes: u64,
    pub resident: u64,
    pub swapped: u64,
    /// Whether something holds it in RAM.
    pub held: bool,
    /// Whether an OS-level mechanism holds it (the guarantee), as opposed
    /// to only a driver.
    pub held_by_os: bool,
    pub mechanisms: Vec<Mechanism>,
    /// How the engine is running: "systemd service", "user session",
    /// "container" or "shell".
    pub environment: String,
    /// How to make the OS-level hold work, if it doesn't.
    pub fixes: Vec<String>,
}

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// Bytes of swap configured (swap files, partitions, zram).
fn swap_total() -> u64 {
    read("/proc/meminfo")
        .lines()
        .find(|l| l.starts_with("SwapTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
        * 1024
}

/// This process's cgroup v2 path, e.g. `/user.slice/.../run-x.scope`.
fn cgroup() -> Option<String> {
    read("/proc/self/cgroup")
        .lines()
        .find_map(|l| l.strip_prefix("0::").map(String::from))
}

/// Whether swap is forbidden for this process: memory.swap.max is 0 here or
/// on any ancestor (the limit is hierarchical).
fn swap_forbidden(cg: &str) -> bool {
    let mut path = std::path::PathBuf::from(format!("/sys/fs/cgroup{cg}"));
    loop {
        if read(&path.join("memory.swap.max").to_string_lossy()).trim() == "0" {
            return true;
        }
        if !path.pop() || path == std::path::Path::new("/sys/fs") {
            return false;
        }
    }
}

/// How the engine is running, from its cgroup and container markers.
pub fn environment(cg: &str) -> &'static str {
    let container = std::path::Path::new("/.dockerenv").exists()
        || std::path::Path::new("/run/.containerenv").exists()
        || ["docker", "kubepods", "containerd", "libpod", "lxc"]
            .iter()
            .any(|m| cg.contains(m));
    if container {
        "container"
    } else if cg.starts_with("/system.slice/") {
        "systemd service"
    } else if cg.contains("/user@") {
        "user session"
    } else {
        "shell"
    }
}

fn memlock_limit() -> (u64, u64) {
    let l = aie_sys::memlock_limit();
    (l.soft, l.hard)
}

fn mib(b: u64) -> String {
    if b == aie_sys::UNLIMITED {
        "unlimited".into()
    } else {
        format!("{} MiB", b >> 20)
    }
}

/// The RAM tier's status right now.
pub fn status() -> RamTier {
    let bufs: Vec<_> = aie_llama::pinned::buffers()
        .into_iter()
        .filter(|b| !b.freed && b.len > 0)
        .collect();
    let bytes: u64 = bufs.iter().map(|b| b.len as u64).sum();
    let mut resident = 0;
    let mut swapped = 0;
    for b in &bufs {
        let s = residency::smaps(b.base, b.map_len);
        resident += s.resident.min(b.map_len as u64);
        swapped += s.swapped;
    }
    let cg = cgroup().unwrap_or_default();
    let env = environment(&cg);
    let swap = swap_total();
    let (lim_cur, _) = memlock_limit();

    let mut m = Vec::new();
    // OS level: the guarantee.
    m.push(Mechanism {
        layer: "os",
        name: "no swap".into(),
        ok: swap == 0,
        detail: if swap == 0 {
            "no swap is configured, so memory can't be swapped out".into()
        } else {
            format!("{} of swap is configured", mib(swap))
        },
    });
    let forbidden = swap_forbidden(&cg);
    m.push(Mechanism {
        layer: "os",
        name: "swap forbidden".into(),
        ok: forbidden,
        detail: if forbidden {
            "swap is forbidden for this process (cgroup memory.swap.max = 0)".into()
        } else {
            "this process may be swapped".into()
        },
    });
    let asked_lock: Vec<_> = bufs.iter().filter(|b| b.mlock_result.is_some()).collect();
    if !asked_lock.is_empty() {
        let failed = asked_lock
            .iter()
            .find_map(|b| b.mlock_result.and_then(|r| r.err()));
        m.push(Mechanism {
            layer: "os",
            name: "locked".into(),
            ok: failed.is_none(),
            detail: match failed {
                None => "locked in memory (mlock)".into(),
                Some(e) => format!(
                    "mlock failed (errno {e}); the lock limit is {}, the RAM tier needs {}",
                    mib(lim_cur),
                    mib(bytes)
                ),
            },
        });
    }
    let hugetlb: Vec<_> = bufs.iter().filter(|b| b.hugetlb_result.is_some()).collect();
    if !hugetlb.is_empty() {
        let ok = hugetlb
            .iter()
            .all(|b| matches!(b.hugetlb_result, Some(Ok(_))));
        m.push(Mechanism {
            layer: "os",
            name: "reserved huge pages".into(),
            ok,
            detail: if ok {
                "on reserved huge pages, which are never swapped".into()
            } else {
                "reserved huge pages asked for but not available; normal pages used".into()
            },
        });
    }
    // Driver level: used when available, for speed; also keeps pages in RAM.
    let registered: Vec<_> = bufs
        .iter()
        .filter(|b| b.cuda_register_result.is_some())
        .collect();
    if !registered.is_empty() {
        let failed = registered
            .iter()
            .find_map(|b| b.cuda_register_result.and_then(|r| r.err()));
        m.push(Mechanism {
            layer: "cuda",
            name: "CUDA registration".into(),
            ok: failed.is_none(),
            detail: match failed {
                None => "registered with CUDA: GPUs copy from it at full speed, and the driver keeps it in RAM".into(),
                Some(-1) => "the CUDA runtime couldn't be loaded".into(),
                Some(e) => format!("cudaHostRegister failed (cudaError {e})"),
            },
        });
    }

    let held_by_os = bytes == 0 || m.iter().any(|x| x.layer == "os" && x.ok);
    let held = held_by_os || m.iter().any(|x| x.ok);
    let mut fixes = Vec::new();
    if !held_by_os {
        match env {
            "systemd service" => fixes.push("add `LimitMEMLOCK=infinity` (or `MemorySwapMax=0`) to the service's [Service] section, then `systemctl daemon-reload` and restart".into()),
            "user session" => fixes.push("the daemon forbids swap for its workers when the systemd user manager can control memory; it couldn't here (is `systemd-run --user` available, with the memory controller delegated?)".into()),
            "container" => fixes.push("run the container with `--ulimit memlock=-1` (docker/podman), or without swap (`--memory-swap` equal to `--memory`)".into()),
            _ => fixes.push("raise the lock limit before starting: `ulimit -l unlimited` (needs a `memlock unlimited` line for your user in /etc/security/limits.conf), or run it as the systemd service in contrib/systemd".into()),
        }
        fixes.push("or give the binary the lock capability once: `sudo setcap cap_ipc_lock=ep $(which ai-engine)`".into());
    }

    RamTier {
        bytes,
        resident,
        swapped,
        held,
        held_by_os,
        mechanisms: m,
        environment: env.into(),
        fixes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environments() {
        assert_eq!(
            environment("/system.slice/ai-engine.service"),
            "systemd service"
        );
        assert_eq!(
            environment("/user.slice/user-1000.slice/user@1000.service/app.slice/run-1.scope"),
            "user session"
        );
        assert_eq!(
            environment("/user.slice/user-1000.slice/session-3.scope"),
            "shell"
        );
        assert_eq!(environment("/kubepods/burstable/pod1/abc"), "container");
    }

    #[test]
    fn status_without_a_ram_tier_is_held() {
        let s = status();
        if s.bytes == 0 {
            assert!(s.held && s.held_by_os && s.fixes.is_empty());
        }
    }
}
