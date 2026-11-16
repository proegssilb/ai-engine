//! Which socket (NUMA node) does a model's host side run on, and is it there?
//!
//! One node per model for everything on the host: RAM-tier memory, CPU
//! threads, host-side buffers (KV cache, compute buffers) and the page cache
//! pages the model reads. llama.cpp doesn't partition CPU work by node, so a
//! model is never split across nodes.
//!
//! The worker does it itself, unprivileged: CPU affinity for every thread
//! (`sched_setaffinity`) and a process memory policy preferring the node
//! (`set_mempolicy`, tolerant: a full node doesn't kill the process). The RAM
//! tier's buffer additionally `mbind`s strictly (aie_llama::pinned). Then it
//! measures, instead of assuming: where the RAM tier's pages are, what the
//! threads may run on, how much of the process's memory is on each node.
//!
//! Everything that reads the machine takes a root directory (normally `/`),
//! so tests can use fake sysfs trees.
//!
//! See docs/reference/placement-guarantees.md.

use aie_plan::Tier;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

/// A NUMA node as sysfs describes it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Node {
    pub id: u32,
    pub cpus: Vec<u32>,
    /// The kernel's own notation, e.g. `0-15,32-47`.
    pub cpulist: String,
    pub total: u64,
    pub free: u64,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Topology {
    pub nodes: Vec<Node>,
}

/// `0-3,8,10-11` → `[0,1,2,3,8,10,11]`.
pub fn parse_cpulist(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                    out.extend(a..=b);
                }
            }
            None => out.extend(part.trim().parse::<u32>()),
        }
    }
    out
}

impl Topology {
    /// Read `<root>/sys/devices/system/node/node*`. A kernel without NUMA
    /// (no such directory) is one node, 0, whose CPUs are left unrestricted.
    pub fn read(root: &Path) -> Topology {
        let dir = root.join("sys/devices/system/node");
        let mut nodes: Vec<Node> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let id: u32 = e
                    .file_name()
                    .to_string_lossy()
                    .strip_prefix("node")?
                    .parse()
                    .ok()?;
                let path = e.path();
                let info = std::fs::read_to_string(path.join("meminfo")).unwrap_or_default();
                let kib = |key: &str| {
                    info.lines()
                        .find(|l| l.contains(key))
                        .and_then(|l| l.split_whitespace().rev().nth(1))
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(0)
                        * 1024
                };
                let cpulist = std::fs::read_to_string(path.join("cpulist"))
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                Some(Node {
                    id,
                    cpus: parse_cpulist(&cpulist),
                    cpulist,
                    total: kib("MemTotal:"),
                    free: kib("MemFree:"),
                })
            })
            .collect();
        nodes.sort_by_key(|n| n.id);
        if nodes.is_empty() {
            nodes.push(Node {
                id: 0,
                cpus: vec![],
                cpulist: String::new(),
                total: 0,
                free: 0,
            });
        }
        Topology { nodes }
    }

    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Nodes a model can run on: those with CPUs (memory-only nodes, e.g.
    /// CXL or HBM, can't host threads). A kernel without NUMA counts as one.
    pub fn sockets(&self) -> Vec<&Node> {
        let s: Vec<&Node> = self.nodes.iter().filter(|n| !n.cpus.is_empty()).collect();
        if s.is_empty() {
            self.nodes.iter().take(1).collect()
        } else {
            s
        }
    }

    /// More than one socket: where the node choice matters.
    pub fn multi(&self) -> bool {
        self.sockets().len() > 1
    }

    /// Node whose CPUs include `cpu`.
    pub fn node_of_cpu(&self, cpu: u32) -> Option<u32> {
        self.nodes
            .iter()
            .find(|n| n.cpus.contains(&cpu))
            .map(|n| n.id)
    }
}

/// The NUMA node of a PCI device (`0000:c1:00.0`) from sysfs. `-1` or
/// missing (no NUMA info, or a single-node machine) means node 0.
pub fn pci_node(root: &Path, pci_bus_id: &str) -> u32 {
    std::fs::read_to_string(
        root.join("sys/bus/pci/devices")
            .join(pci_bus_id)
            .join("numa_node"),
    )
    .ok()
    .and_then(|t| t.trim().parse::<i64>().ok())
    .filter(|n| *n >= 0)
    .map_or(0, |n| n as u32)
}

/// Why a node was picked, and which.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub node: u32,
    pub auto: bool,
    pub reason: String,
}

/// The socket for a model. The user's choice, if it exists; else automatic:
/// the node of the GPU holding the most of the model's layers (the lowest
/// index on a tie), so layers' host-side work sits next to the GPU's PCIe
/// link. With no layers on GPUs, GPU 0's node. With no GPUs at all, the
/// socket with the most free memory. `gpu_nodes[i]` is GPU i's node.
pub fn choose(
    topo: &Topology,
    requested: Option<u32>,
    gpu_nodes: &[u32],
    layer_home: &[Tier],
) -> Result<Choice, String> {
    let sockets = topo.sockets();
    if let Some(n) = requested {
        let Some(node) = topo.node(n) else {
            let have: Vec<String> = sockets.iter().map(|s| s.id.to_string()).collect();
            return Err(format!(
                "socket {n} doesn't exist: this machine has {} (socket {})",
                if sockets.len() == 1 {
                    "one socket".to_string()
                } else {
                    format!("{} sockets", sockets.len())
                },
                have.join(", ")
            ));
        };
        if node.cpus.is_empty() && !(sockets.len() == 1 && sockets[0].id == n) {
            return Err(format!(
                "socket {n} has no CPUs (it's memory only), so a model can't run there"
            ));
        }
        return Ok(Choice {
            node: n,
            auto: false,
            reason: "chosen".into(),
        });
    }
    let usable = |n: u32| sockets.iter().any(|s| s.id == n);
    let mut layers = vec![0u32; gpu_nodes.len()];
    for t in layer_home {
        if let Tier::Gpu(g) = t
            && let Some(c) = layers.get_mut(*g as usize)
        {
            *c += 1;
        }
    }
    let best = layers
        .iter()
        .enumerate()
        .filter(|(_, c)| **c > 0)
        .max_by_key(|(i, c)| (**c, std::cmp::Reverse(*i)))
        .map(|(i, _)| i)
        .or(if gpu_nodes.is_empty() { None } else { Some(0) });
    if let Some(g) = best
        && usable(gpu_nodes[g])
    {
        return Ok(Choice {
            node: gpu_nodes[g],
            auto: true,
            reason: format!("next to GPU {g}"),
        });
    }
    let freest = sockets
        .iter()
        .max_by_key(|s| (s.free, std::cmp::Reverse(s.id)))
        .map_or(0, |s| s.id);
    Ok(Choice {
        node: freest,
        auto: true,
        reason: if sockets.len() == 1 {
            "the only socket".into()
        } else {
            "the socket with the most free memory".into()
        },
    })
}

// ---------------------------------------------------------------------
// Applying: affinity and memory policy
// ---------------------------------------------------------------------

/// What was applied to this process, decided once at load.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Applied {
    pub node: u32,
    pub auto: bool,
    pub reason: String,
    /// CPUs threads may run on: the node's CPUs that the process was allowed.
    pub cpus: Vec<u32>,
    pub affinity_error: Option<String>,
    /// Memory policy: "preferred" (MPOL_PREFERRED) or "default" if not set.
    pub policy: String,
    pub policy_verified: bool,
    pub policy_error: Option<String>,
}

static APPLIED: Mutex<Option<Applied>> = Mutex::new(None);

fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
}

fn set_of(cpus: &[u32]) -> libc::cpu_set_t {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    for c in cpus {
        if (*c as usize) < libc::CPU_SETSIZE as usize {
            unsafe { libc::CPU_SET(*c as usize, &mut set) };
        }
    }
    set
}

fn affinity_of(tid: libc::pid_t) -> Vec<u32> {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::sched_getaffinity(tid, std::mem::size_of_val(&set), &mut set) };
    if r != 0 {
        return vec![];
    }
    (0..libc::CPU_SETSIZE as usize)
        .filter(|c| unsafe { libc::CPU_ISSET(*c, &set) })
        .map(|c| c as u32)
        .collect()
}

fn set_affinity(tid: libc::pid_t, cpus: &[u32]) -> Result<(), i32> {
    let set = set_of(cpus);
    let r = unsafe { libc::sched_setaffinity(tid, std::mem::size_of_val(&set), &set) };
    if r == 0 { Ok(()) } else { Err(errno()) }
}

fn thread_ids() -> Vec<libc::pid_t> {
    std::fs::read_dir("/proc/self/task")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().parse().ok())
        .collect()
}

const MPOL_PREFERRED: libc::c_long = 1;

fn node_mask(node: u32) -> [libc::c_ulong; 16] {
    let mut m = [0; 16];
    m[(node / 64) as usize % 16] = 1 << (node % 64);
    m
}

/// MPOL_PREFERRED for the calling thread (new threads inherit it).
fn set_policy(node: u32) -> Result<(), i32> {
    let mask = node_mask(node);
    let r = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            MPOL_PREFERRED,
            mask.as_ptr(),
            (mask.len() * 64) as libc::c_ulong,
        )
    };
    if r == 0 { Ok(()) } else { Err(errno()) }
}

/// The calling thread's policy as (mode, first node of the mask).
fn get_policy() -> Option<(i32, Option<u32>)> {
    let mut mode: libc::c_int = -1;
    let mut mask = [0 as libc::c_ulong; 16];
    let r = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut mode,
            mask.as_mut_ptr(),
            (mask.len() * 64) as libc::c_ulong,
            0usize,
            0usize,
        )
    };
    if r != 0 {
        return None;
    }
    let node = (0..1024u32).find(|n| mask[(*n / 64) as usize] >> (n % 64) & 1 == 1);
    Some((mode, node))
}

/// Pin this process to the choice: affinity for every existing thread, a
/// preferred-node memory policy for the calling thread. Threads created later
/// inherit from their creator; [`init_thread`] covers runtime-spawned ones.
/// Call before llama.cpp creates its threads and buffers.
///
/// Fails only if the node's CPUs aren't among those this process may use
/// (e.g. started under `taskset` on another socket). Anything else is
/// recorded in the result and reported, not fatal.
pub fn apply(topo: &Topology, choice: &Choice) -> Result<Applied, String> {
    let node = topo
        .node(choice.node)
        .ok_or_else(|| format!("socket {} doesn't exist", choice.node))?;
    let allowed = affinity_of(0);
    let cpus: Vec<u32> = if node.cpus.is_empty() {
        allowed.clone() // a kernel without NUMA info: nothing to restrict to
    } else {
        node.cpus
            .iter()
            .copied()
            .filter(|c| allowed.is_empty() || allowed.contains(c))
            .collect()
    };
    if cpus.is_empty() {
        return Err(format!(
            "none of socket {}'s CPUs ({}) may be used by this process; it is restricted to CPUs {}",
            node.id,
            node.cpulist,
            list(&allowed)
        ));
    }
    let mut affinity_error = None;
    for tid in thread_ids() {
        if let Err(e) = set_affinity(tid, &cpus) {
            affinity_error = Some(format!("sched_setaffinity failed (errno {e})"));
            break;
        }
    }
    let (policy, policy_verified, policy_error) = match set_policy(node.id) {
        Ok(()) => (
            "preferred".to_string(),
            get_policy() == Some((MPOL_PREFERRED as i32, Some(node.id))),
            None,
        ),
        Err(e) => (
            "default".to_string(),
            false,
            Some(format!("set_mempolicy failed (errno {e})")),
        ),
    };
    let applied = Applied {
        node: node.id,
        auto: choice.auto,
        reason: choice.reason.clone(),
        cpus,
        affinity_error,
        policy,
        policy_verified,
        policy_error,
    };
    *APPLIED.lock().unwrap() = Some(applied.clone());
    Ok(applied)
}

/// For a runtime's `on_thread_start`: give a new thread the process's affinity
/// and memory policy, whoever created it. No-op until [`apply`] has run.
pub fn init_thread() {
    if let Some(a) = APPLIED.lock().unwrap().as_ref() {
        let _ = set_affinity(0, &a.cpus);
        if a.policy == "preferred" {
            let _ = set_policy(a.node);
        }
    }
}

/// llama.cpp's thread count when the config leaves it open: one per CPU of
/// the node. Without a pin, all CPUs this process may use.
pub fn default_threads(applied: &Applied) -> i32 {
    applied.cpus.len().max(1) as i32
}

fn list(cpus: &[u32]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let mut j = i;
        while j + 1 < cpus.len() && cpus[j + 1] == cpus[j] + 1 {
            j += 1;
        }
        out.push(if j > i {
            format!("{}-{}", cpus[i], cpus[j])
        } else {
            cpus[i].to_string()
        });
        i = j + 1;
    }
    out.join(",")
}

// ---------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------

/// Bytes mapped per node, from `/proc/<pid>/numa_maps` text: every
/// `N<node>=<pages>` times that mapping's page size.
pub fn parse_numa_maps(text: &str) -> BTreeMap<u32, u64> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let page_kb = line
            .split_whitespace()
            .find_map(|t| t.strip_prefix("kernelpagesize_kB="))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(4);
        for t in line.split_whitespace() {
            if let Some(rest) = t.strip_prefix('N')
                && let Some((n, pages)) = rest.split_once('=')
                && let (Ok(n), Ok(pages)) = (n.parse::<u32>(), pages.parse::<u64>())
            {
                *out.entry(n).or_insert(0) += pages * page_kb * 1024;
            }
        }
    }
    out
}

/// What `/proc/self/numa_maps` says about the pages mapped by this process,
/// split by what backs them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MapsSplit {
    /// Pages of the given file (the model) mapped by this process, per node:
    /// page cache, which the kernel manages. Reported, never placed.
    pub file: BTreeMap<u32, u64>,
    /// Anonymous memory (allocations, stacks), per node.
    pub anon: BTreeMap<u32, u64>,
}

/// [`parse_numa_maps`], keeping apart the pages of `file` (a canonical path)
/// and anonymous pages. Other file mappings (libraries) are ignored.
pub fn split_numa_maps(text: &str, file: &str) -> MapsSplit {
    let mut out = MapsSplit::default();
    let want = format!("file={file}");
    for line in text.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        let is_file = toks.iter().any(|t| *t == want);
        let is_anon = !toks.iter().any(|t| t.starts_with("file="));
        if !is_file && !is_anon {
            continue;
        }
        let page_kb = toks
            .iter()
            .find_map(|t| t.strip_prefix("kernelpagesize_kB="))
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(4);
        for t in &toks {
            if let Some(rest) = t.strip_prefix('N')
                && let Some((n, pages)) = rest.split_once('=')
                && let (Ok(n), Ok(pages)) = (n.parse::<u32>(), pages.parse::<u64>())
            {
                let b = pages * page_kb * 1024;
                if is_file {
                    *out.file.entry(n).or_insert(0) += b;
                } else {
                    *out.anon.entry(n).or_insert(0) += b;
                }
            }
        }
    }
    out
}

/// Estimated bytes per node of `len` bytes from `base`, by asking the kernel
/// (`move_pages` without a destination) about up to `samples` pages spread
/// evenly. Pages not yet in memory aren't counted.
pub fn sample_pages(
    base: usize,
    len: usize,
    page: usize,
    samples: usize,
) -> (BTreeMap<u32, u64>, bool) {
    let total = len / page.max(1);
    if total == 0 {
        return (BTreeMap::new(), true);
    }
    let n = total.min(samples.max(1));
    let exact = n == total;
    let mut pages: Vec<*mut libc::c_void> = (0..n)
        .map(|i| (base + (i * total / n) * page) as *mut libc::c_void)
        .collect();
    let mut status = vec![0 as libc::c_int; n];
    let r = unsafe {
        libc::syscall(
            libc::SYS_move_pages,
            0,
            n as libc::c_ulong,
            pages.as_mut_ptr(),
            std::ptr::null::<libc::c_int>(),
            status.as_mut_ptr(),
            0,
        )
    };
    let mut out = BTreeMap::new();
    if r != 0 {
        return (out, false);
    }
    let per_sample = (total as u64 * page as u64) / n as u64;
    for s in status.into_iter().filter(|s| *s >= 0) {
        *out.entry(s as u32).or_insert(0) += per_sample;
    }
    (out, exact)
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

/// What the measurements add up to, as plain-language problems. Pure, so
/// tests can feed it any situation. Nothing is a problem on one socket.
pub struct Facts<'a> {
    pub node: u32,
    pub multi: bool,
    /// Sockets (by node) any thread may run on that aren't `node`.
    pub other_sockets_allowed: &'a [u32],
    pub ram_tier_by_node: &'a BTreeMap<u32, u64>,
    /// Anonymous memory that isn't the RAM tier (KV cache, compute buffers,
    /// host-side allocations), per node.
    pub buffers_by_node: &'a BTreeMap<u32, u64>,
    pub errors: &'a [String],
}

pub fn problems(f: &Facts) -> Vec<String> {
    if !f.multi {
        return vec![];
    }
    let mut out: Vec<String> = f.errors.to_vec();
    let sock = f.node;
    if !f.other_sockets_allowed.is_empty() {
        let o: Vec<String> = f
            .other_sockets_allowed
            .iter()
            .map(|n| n.to_string())
            .collect();
        out.push(format!(
            "some threads of this model may run on socket {}, not only socket {sock}",
            o.join(", ")
        ));
    }
    let tier_total: u64 = f.ram_tier_by_node.values().sum();
    for (n, b) in f.ram_tier_by_node {
        // 1% slack: sampling and a few stray pages aren't a placement problem.
        if *n != sock && *b * 100 > tier_total.max(1) {
            out.push(format!(
                "{} of the RAM tier is on socket {n}, not socket {sock}",
                gb(*b)
            ));
        }
    }
    let total: u64 = f.buffers_by_node.values().sum();
    let off: u64 = f
        .buffers_by_node
        .iter()
        .filter(|(n, _)| **n != sock)
        .map(|(_, b)| b)
        .sum();
    if off > (256 << 20) && off * 10 > total {
        out.push(format!(
            "{} of the host-side buffers (KV cache, compute buffers) are on another socket than socket {sock}",
            gb(off)
        ));
    }
    out
}

/// The live report for the status: decided at load plus measured now.
#[derive(Clone, Debug, Serialize)]
pub struct HostReport {
    pub node: u32,
    pub auto: bool,
    /// "next to GPU 0", "chosen", ...
    pub reason: String,
    /// Sockets on this machine that can host a model.
    pub sockets: u32,
    pub threads: i32,
    pub affinity: Affinity,
    pub policy: Policy,
    /// The RAM tier's bytes per node (estimated from sampled pages).
    pub ram_tier_by_node: BTreeMap<u32, u64>,
    pub ram_tier_exact: bool,
    /// All of this process's mapped memory per node (numa_maps).
    pub memory_by_node: BTreeMap<u32, u64>,
    /// The model file's pages in the page cache that this process maps, per
    /// node. Reported only: the kernel manages the page cache, so where these
    /// sit isn't part of the socket's promise.
    pub file_cache_by_node: BTreeMap<u32, u64>,
    /// KV cache, compute buffers and other host allocations per node.
    pub host_buffers_by_node: BTreeMap<u32, u64>,
    pub problems: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Affinity {
    /// CPUs any thread of the process may run on right now.
    pub cpus: u32,
    /// Sockets those CPUs belong to.
    pub nodes: Vec<u32>,
    /// Whether they're all on the chosen socket.
    pub on_node: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Policy {
    pub mode: String,
    pub node: u32,
    pub verified: bool,
    pub error: Option<String>,
}

/// Measure now. `threads` is what llama.cpp was told to use.
pub fn report(topo: &Topology, a: &Applied, threads: i32, model: &Path) -> HostReport {
    // Every thread's affinity, not just this one's.
    let mut cpus: Vec<u32> = thread_ids().into_iter().flat_map(affinity_of).collect();
    cpus.sort();
    cpus.dedup();
    let mut nodes: Vec<u32> = cpus.iter().filter_map(|c| topo.node_of_cpu(*c)).collect();
    nodes.sort();
    nodes.dedup();
    let on_node = !nodes.iter().any(|n| *n != a.node);

    let mut tier: BTreeMap<u32, u64> = BTreeMap::new();
    let mut exact = true;
    let bufs: Vec<_> = aie_llama::pinned::REGISTRY
        .lock()
        .unwrap()
        .iter()
        .filter(|b| !b.freed && b.len > 0)
        .cloned()
        .collect();
    let small = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    for b in &bufs {
        let page = match b.hugetlb_result {
            Some(Ok(hp)) => hp as usize,
            _ => small,
        };
        let (m, ex) = sample_pages(b.base, b.map_len, page, 8192);
        exact &= ex;
        for (n, v) in m {
            *tier.entry(n).or_insert(0) += v;
        }
    }
    let canon = std::fs::canonicalize(model)
        .map(|c| c.display().to_string())
        .unwrap_or_default();
    let read_maps = || std::fs::read_to_string("/proc/self/numa_maps").unwrap_or_default();
    let text = read_maps();
    let split = split_numa_maps(&text, &canon);
    let mut errors = Vec::new();
    let memory = parse_numa_maps(&text);
    let mut buffers = split.anon.clone();
    for (n, b) in &tier {
        if let Some(v) = buffers.get_mut(n) {
            *v = v.saturating_sub(*b);
        }
    }

    errors.extend(a.affinity_error.iter().cloned());
    errors.extend(a.policy_error.iter().cloned());
    let other: Vec<u32> = nodes.iter().copied().filter(|n| *n != a.node).collect();
    let problems = problems(&Facts {
        node: a.node,
        multi: topo.multi(),
        other_sockets_allowed: &other,
        ram_tier_by_node: &tier,
        buffers_by_node: &buffers,
        errors: &errors,
    });
    HostReport {
        node: a.node,
        auto: a.auto,
        reason: a.reason.clone(),
        sockets: topo.sockets().len() as u32,
        threads,
        affinity: Affinity {
            cpus: cpus.len() as u32,
            nodes,
            on_node,
            error: a.affinity_error.clone(),
        },
        policy: Policy {
            mode: a.policy.clone(),
            node: a.node,
            verified: a.policy_verified,
            error: a.policy_error.clone(),
        },
        ram_tier_by_node: tier,
        ram_tier_exact: exact,
        memory_by_node: memory,
        file_cache_by_node: split.file,
        host_buffers_by_node: buffers,
        problems,
    }
}

/// [`report`] as JSON, at most every two seconds: it walks the process's page
/// tables, and the UI polls.
pub fn report_json(topo: &Topology, a: &Applied, threads: i32, model: &Path) -> String {
    static CACHE: Mutex<Option<(std::time::Instant, String)>> = Mutex::new(None);
    let mut c = CACHE.lock().unwrap();
    if let Some((t, s)) = c.as_ref()
        && t.elapsed() < std::time::Duration::from_secs(2)
    {
        return s.clone();
    }
    let s = serde_json::to_string(&report(topo, a, threads, model)).unwrap_or_default();
    *c = Some((std::time::Instant::now(), s.clone()));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A fake machine: two sockets, GPUs on PCI addresses with numa_node files.
    fn fake(root: &Path, nodes: &[(u32, &str, u64, u64)], pci: &[(&str, &str)]) {
        for (id, cpus, total_kib, free_kib) in nodes {
            let d = root.join(format!("sys/devices/system/node/node{id}"));
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("cpulist"), format!("{cpus}\n")).unwrap();
            fs::write(
                d.join("meminfo"),
                format!(
                    "Node {id} MemTotal:       {total_kib} kB\nNode {id} MemFree:        {free_kib} kB\n"
                ),
            )
            .unwrap();
        }
        for (addr, node) in pci {
            let d = root.join("sys/bus/pci/devices").join(addr);
            fs::create_dir_all(&d).unwrap();
            fs::write(d.join("numa_node"), format!("{node}\n")).unwrap();
        }
    }

    fn two_sockets() -> (tempdir::Dir, Topology) {
        let dir = tempdir::Dir::new();
        fake(
            dir.path(),
            &[
                (0, "0-15,32-47", 256 << 20, 100 << 20),
                (1, "16-31,48-63", 256 << 20, 200 << 20),
            ],
            &[
                ("0000:3b:00.0", "0"),
                ("0000:af:00.0", "1"),
                ("0000:5e:00.0", "-1"),
            ],
        );
        let t = Topology::read(dir.path());
        (dir, t)
    }

    /// A scratch directory without a new dependency.
    mod tempdir {
        pub struct Dir(std::path::PathBuf);
        impl Dir {
            pub fn new() -> Dir {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let p = std::env::temp_dir().join(format!(
                    "aie-numa-test-{}-{}",
                    std::process::id(),
                    N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                ));
                std::fs::create_dir_all(&p).unwrap();
                Dir(p)
            }
            pub fn path(&self) -> &std::path::Path {
                &self.0
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn cpulists() {
        assert_eq!(parse_cpulist("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpulist(""), Vec::<u32>::new());
        assert_eq!(list(&[0, 1, 2, 3, 8, 10, 11]), "0-3,8,10-11");
    }

    #[test]
    fn reads_a_fake_two_socket_machine() {
        let (_d, t) = two_sockets();
        assert!(t.multi());
        assert_eq!(t.nodes.len(), 2);
        assert_eq!(t.nodes[1].cpus.len(), 32);
        assert_eq!(t.nodes[1].free, 200 << 30);
        assert_eq!(t.node_of_cpu(50), Some(1));
        assert_eq!(t.node_of_cpu(3), Some(0));
    }

    #[test]
    fn no_numa_dirs_is_one_socket() {
        let d = tempdir::Dir::new();
        let t = Topology::read(d.path());
        assert!(!t.multi());
        assert_eq!(t.sockets().len(), 1);
        assert_eq!(t.sockets()[0].id, 0);
    }

    #[test]
    fn gpu_nodes_from_pci() {
        let (d, _) = two_sockets();
        assert_eq!(pci_node(d.path(), "0000:af:00.0"), 1);
        assert_eq!(pci_node(d.path(), "0000:3b:00.0"), 0);
        // -1 and unknown devices are node 0
        assert_eq!(pci_node(d.path(), "0000:5e:00.0"), 0);
        assert_eq!(pci_node(d.path(), "0000:ff:00.0"), 0);
    }

    #[test]
    fn auto_follows_the_gpu_with_most_layers() {
        let (_d, t) = two_sockets();
        let homes = |g0: usize, g1: usize| {
            let mut v = vec![Tier::Gpu(0); g0];
            v.extend(vec![Tier::Gpu(1); g1]);
            v.extend(vec![Tier::Ram; 3]);
            v
        };
        let c = choose(&t, None, &[0, 1], &homes(10, 30)).unwrap();
        assert_eq!(
            (c.node, c.auto, c.reason.as_str()),
            (1, true, "next to GPU 1")
        );
        // a tie goes to the lower GPU
        assert_eq!(choose(&t, None, &[0, 1], &homes(20, 20)).unwrap().node, 0);
        // no layers on GPUs: GPU 0's node
        assert_eq!(choose(&t, None, &[1, 0], &[Tier::Ram; 8]).unwrap().node, 1);
        // no GPUs: the socket with the most free memory
        let c = choose(&t, None, &[], &[Tier::Ram; 8]).unwrap();
        assert_eq!(c.node, 1);
        assert_eq!(c.reason, "the socket with the most free memory");
    }

    #[test]
    fn explicit_choice_and_refusals() {
        let (_d, t) = two_sockets();
        let c = choose(&t, Some(1), &[0], &[Tier::Gpu(0)]).unwrap();
        assert_eq!((c.node, c.auto), (1, false));
        let e = choose(&t, Some(2), &[0], &[]).unwrap_err();
        assert_eq!(
            e,
            "socket 2 doesn't exist: this machine has 2 sockets (socket 0, 1)"
        );
        let one = Topology::read(tempdir::Dir::new().path());
        let e = choose(&one, Some(1), &[], &[]).unwrap_err();
        assert!(
            e.contains("socket 1 doesn't exist") && e.contains("one socket"),
            "{e}"
        );
        assert_eq!(choose(&one, None, &[0], &[Tier::Gpu(0)]).unwrap().node, 0);
    }

    #[test]
    fn memory_only_nodes_cant_host() {
        let d = tempdir::Dir::new();
        fake(
            d.path(),
            &[(0, "0-7", 1 << 20, 1 << 19), (1, "", 1 << 22, 1 << 22)],
            &[("0000:01:00.0", "1")],
        );
        let t = Topology::read(d.path());
        assert!(!t.multi());
        assert!(
            choose(&t, Some(1), &[], &[])
                .unwrap_err()
                .contains("memory only")
        );
        // a GPU on a CPU-less node falls back to a socket that can run threads
        assert_eq!(choose(&t, None, &[1], &[Tier::Gpu(0)]).unwrap().node, 0);
    }

    #[test]
    fn numa_maps_sums_pages_by_node() {
        let text = "\
7f00 default anon=10 dirty=10 N0=6 N1=4 kernelpagesize_kB=4
7f10 bind:1 huge anon=2 N1=2 kernelpagesize_kB=2048
7f20 default file=/x mapped=3 N0=3 kernelpagesize_kB=4
";
        let m = parse_numa_maps(text);
        assert_eq!(m[&0], 9 * 4096);
        assert_eq!(m[&1], 4 * 4096 + 2 * 2048 * 1024);
    }

    #[test]
    fn numa_maps_splits_file_from_anon() {
        let text = "\
7f00 default anon=10 N0=6 N1=4 kernelpagesize_kB=4
7f20 default file=/m.gguf mapped=3 mapmax=1 N0=1 N1=2 kernelpagesize_kB=4
7f30 default file=/m.gguf mapped=2 mapmax=2 N1=2 kernelpagesize_kB=4
7f40 default file=/lib.so mapped=9 N1=9 kernelpagesize_kB=4
";
        let m = split_numa_maps(text, "/m.gguf");
        assert_eq!(m.file[&0], 4096);
        assert_eq!(m.file[&1], 4 * 4096);
        assert_eq!(m.anon[&0], 6 * 4096);
        assert_eq!(m.anon[&1], 4 * 4096);
    }

    fn facts<'a>(
        tier: &'a BTreeMap<u32, u64>,
        bufs: &'a BTreeMap<u32, u64>,
        errors: &'a [String],
    ) -> Facts<'a> {
        Facts {
            node: 0,
            multi: true,
            other_sockets_allowed: &[],
            ram_tier_by_node: tier,
            buffers_by_node: bufs,
            errors,
        }
    }

    #[test]
    fn problems_name_what_is_claimed_and_wrong() {
        let none = BTreeMap::new();
        let tier: BTreeMap<u32, u64> = [(0, 6_000_000_000), (1, 4_100_000_000)].into();
        let bufs: BTreeMap<u32, u64> = [(0, 1_000_000_000), (1, 900_000_000)].into();
        let errs = ["sched_setaffinity failed (errno 1)".to_string()];
        let mut f = facts(&tier, &bufs, &errs);
        f.other_sockets_allowed = &[1];
        let p = problems(&f);
        assert!(
            p.contains(&"4.1 GB of the RAM tier is on socket 1, not socket 0".to_string()),
            "{p:?}"
        );
        assert!(p.iter().any(|s| s.contains("may run on socket 1")), "{p:?}");
        assert!(
            p.iter()
                .any(|s| s.contains("0.9 GB of the host-side buffers")),
            "{p:?}"
        );
        assert!(p.iter().any(|s| s.contains("sched_setaffinity")), "{p:?}");
        // strays aren't problems: under 1% of the tier
        let tier: BTreeMap<u32, u64> = [(0, 10_000_000_000), (1, 10_000_000)].into();
        let p = problems(&facts(&tier, &none, &[]));
        assert!(p.is_empty(), "{p:?}");
        // one socket: never a problem, whatever is measured
        let mut f = facts(&tier, &none, &errs);
        f.multi = false;
        assert!(problems(&f).is_empty());
    }

    #[test]
    fn samples_pages_of_a_real_mapping() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let len = 64 * page;
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED);
        // untouched: nothing resident yet
        let (m, exact) = sample_pages(base as usize, len, page, 8192);
        assert!(exact && m.is_empty(), "{m:?}");
        unsafe { std::ptr::write_bytes(base as *mut u8, 1, len) };
        let (m, exact) = sample_pages(base as usize, len, page, 8192);
        assert!(exact);
        assert_eq!(m.values().sum::<u64>(), len as u64, "{m:?}");
        // a sparse sample estimates the same total
        let (m, exact) = sample_pages(base as usize, len, page, 8);
        assert!(!exact);
        assert_eq!(m.values().sum::<u64>(), len as u64);
        unsafe { libc::munmap(base, len) };
    }

    /// Runs in its own thread so the affinity/policy changes can't leak into
    /// other tests' threads (affinity is set for all of them, so restore it).
    #[test]
    fn applies_and_measures_on_this_machine() {
        let real = Topology::read(Path::new("/"));
        let before: Vec<(libc::pid_t, Vec<u32>)> = thread_ids()
            .into_iter()
            .map(|t| (t, affinity_of(t)))
            .collect();
        let n0 = real.sockets()[0].id;
        let choice = Choice {
            node: n0,
            auto: true,
            reason: "test".into(),
        };
        let a = apply(&real, &choice).unwrap();
        assert!(!a.cpus.is_empty());
        let r = report(&real, &a, default_threads(&a), Path::new("/nonexistent"));
        assert_eq!(r.node, n0);
        assert!(r.affinity.on_node, "{:?}", r.affinity);
        assert!(r.problems.is_empty() || real.multi(), "{:?}", r.problems);
        for (t, cpus) in before {
            let _ = set_affinity(t, &cpus);
        }
    }
}
