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

/// The NUMA node of a PCI device, or `None` when the kernel doesn't know
/// (`-1`, no file). Unlike [`pci_node`] this never guesses.
pub fn pci_node_opt(root: &Path, pci_bus_id: &str) -> Option<u32> {
    read_node(
        &root
            .join("sys/bus/pci/devices")
            .join(pci_bus_id)
            .join("numa_node"),
    )
}

/// A sysfs `numa_node` file's value; `-1` (unknown) and unreadable are `None`.
fn read_node(file: &Path) -> Option<u32> {
    std::fs::read_to_string(file)
        .ok()?
        .trim()
        .parse::<i64>()
        .ok()
        .filter(|n| *n >= 0)
        .map(|n| n as u32)
}

/// The NUMA node of the block device `major:minor` (what a store's filesystem
/// sits on). A partition or an NVMe namespace has no `numa_node` of its own:
/// the walk goes up through its parents (the disk, the controller, the PCI
/// device) and takes the first `numa_node` file that exists. `None` if there
/// is none or the first one says unknown (`-1`).
pub fn block_node(root: &Path, major: u32, minor: u32) -> Option<u32> {
    let link = root.join(format!("sys/dev/block/{major}:{minor}"));
    let mut dir = std::fs::canonicalize(&link).ok()?;
    let top = std::fs::canonicalize(root.join("sys")).ok()?;
    while dir.starts_with(&top) && dir != top {
        for f in [dir.join("device/numa_node"), dir.join("numa_node")] {
            if f.is_file() {
                return read_node(&f);
            }
        }
        dir = dir.parent()?.to_path_buf();
    }
    None
}

/// The node of the block device holding `path`, from its `st_dev`.
pub fn path_node(root: &Path, path: &Path) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    let dev = std::fs::metadata(path).ok()?.dev();
    block_node(root, libc::major(dev), libc::minor(dev))
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

fn errno(e: std::io::Error) -> i32 {
    e.raw_os_error().unwrap_or(-1)
}

fn affinity_of(tid: u32) -> Vec<u32> {
    aie_sys::get_affinity(tid).unwrap_or_default()
}

fn set_affinity(tid: u32, cpus: &[u32]) -> Result<(), i32> {
    aie_sys::set_affinity(tid, cpus).map_err(errno)
}

fn thread_ids() -> Vec<u32> {
    std::fs::read_dir("/proc/self/task")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().parse().ok())
        .collect()
}

/// The kernel's MPOL_PREFERRED, as reported by `get_mempolicy`.
const MPOL_PREFERRED: i32 = 1;

/// MPOL_PREFERRED for the calling thread (new threads inherit it).
fn set_policy(node: u32) -> Result<(), i32> {
    aie_sys::set_mempolicy_preferred(node).map_err(errno)
}

/// The calling thread's policy as (mode, first node of the mask).
fn get_policy() -> Option<(i32, Option<u32>)> {
    aie_sys::get_mempolicy().ok()
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
            get_policy() == Some((MPOL_PREFERRED, Some(node.id))),
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
    let pages: Vec<usize> = (0..n).map(|i| base + (i * total / n) * page).collect();
    let Ok(status) = aie_sys::move_pages_status(&pages) else {
        return (BTreeMap::new(), false);
    };
    let mut out = BTreeMap::new();
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
    /// `host_buffers_by_node` told apart: the KV cache, the compute scratch
    /// (both sized by llama.cpp's own breakdown of the live context, and
    /// counted only as far as measured resident on that node), and the rest
    /// of the worker's host allocations.
    pub kv_cache_by_node: BTreeMap<u32, u64>,
    pub compute_by_node: BTreeMap<u32, u64>,
    pub other_buffers_by_node: BTreeMap<u32, u64>,
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

/// What llama.cpp's breakdown of the live context says it allocated on the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostAlloc {
    pub kv_cache: u64,
    pub compute: u64,
}

/// Per node: the KV cache, the compute scratch and everything else, out of the
/// measured host buffers. They were allocated on the chosen node, so only that
/// node's measurement is told apart (and never beyond what was measured there,
/// pages count once touched); what other nodes hold is "other", not guessed.
pub fn attribute(
    buffers: &BTreeMap<u32, u64>,
    node: u32,
    alloc: HostAlloc,
) -> [BTreeMap<u32, u64>; 3] {
    let (mut kv, mut cp, mut other) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (&n, &measured) in buffers {
        let (k, c) = if n == node {
            let k = alloc.kv_cache.min(measured);
            (k, alloc.compute.min(measured - k))
        } else {
            (0, 0)
        };
        kv.insert(n, k);
        cp.insert(n, c);
        other.insert(n, measured - k - c);
    }
    [kv, cp, other]
}

/// Measure now. `threads` is what llama.cpp was told to use.
pub fn report(
    topo: &Topology,
    a: &Applied,
    threads: i32,
    model: &Path,
    alloc: HostAlloc,
) -> HostReport {
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
    let small = aie_sys::page_size();
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
    let [kv, cp, other] = attribute(&buffers, a.node, alloc);
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
        kv_cache_by_node: kv,
        compute_by_node: cp,
        other_buffers_by_node: other,
        host_buffers_by_node: buffers,
        problems,
    }
}

/// [`report`] as JSON, at most every two seconds: it walks the process's page
/// tables, and the UI polls.
pub fn report_json(
    topo: &Topology,
    a: &Applied,
    threads: i32,
    model: &Path,
    alloc: HostAlloc,
) -> String {
    static CACHE: Mutex<Option<(std::time::Instant, String)>> = Mutex::new(None);
    let mut c = CACHE.lock().unwrap();
    if let Some((t, s)) = c.as_ref()
        && t.elapsed() < std::time::Duration::from_secs(2)
    {
        return s.clone();
    }
    let s = serde_json::to_string(&report(topo, a, threads, model, alloc)).unwrap_or_default();
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
        let page = aie_sys::page_size();
        let len = 64 * page;
        let mut map = aie_sys::AnonMap::new(len).unwrap();
        let base = map.addr();
        // untouched: nothing resident yet
        let (m, exact) = sample_pages(base, len, page, 8192);
        assert!(exact && m.is_empty(), "{m:?}");
        map.as_mut_slice().fill(1);
        let (m, exact) = sample_pages(base, len, page, 8192);
        assert!(exact);
        assert_eq!(m.values().sum::<u64>(), len as u64, "{m:?}");
        // a sparse sample estimates the same total
        let (m, exact) = sample_pages(base, len, page, 8);
        assert!(!exact);
        assert_eq!(m.values().sum::<u64>(), len as u64);
    }

    /// Runs in its own thread so the affinity/policy changes can't leak into
    /// other tests' threads (affinity is set for all of them, so restore it).
    #[test]
    fn applies_and_measures_on_this_machine() {
        let real = Topology::read(Path::new("/"));
        let before: Vec<(u32, Vec<u32>)> = thread_ids()
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
        let r = report(
            &real,
            &a,
            default_threads(&a),
            Path::new("/nonexistent"),
            HostAlloc::default(),
        );
        assert_eq!(r.node, n0);
        assert!(r.affinity.on_node, "{:?}", r.affinity);
        assert!(r.problems.is_empty() || real.multi(), "{:?}", r.problems);
        for (t, cpus) in before {
            let _ = set_affinity(t, &cpus);
        }
    }

    #[test]
    fn attributes_host_buffers_to_kv_scratch_and_rest() {
        let bufs: BTreeMap<u32, u64> = [(0, 100), (1, 1000)].into();
        let alloc = HostAlloc {
            kv_cache: 300,
            compute: 200,
        };
        // only the chosen node's buffers are told apart: 300 + 200 fit, 500 are "other"
        let [kv, cp, other] = attribute(&bufs, 1, alloc);
        assert_eq!((kv[&1], cp[&1], other[&1]), (300, 200, 500));
        assert_eq!((kv[&0], cp[&0], other[&0]), (0, 0, 100));
        // less measured than allocated: counted only as far as measured
        let bufs: BTreeMap<u32, u64> = [(0, 250)].into();
        let [kv, cp, other] = attribute(&bufs, 0, alloc);
        assert_eq!((kv[&0], cp[&0], other[&0]), (250, 0, 0));
    }

    #[test]
    fn block_node_walks_up_from_a_partition_or_namespace() {
        let dir = tempdir::Dir::new();
        let root = dir.path();
        let w = |p: &str, c: &str| {
            let f = root.join(p);
            fs::create_dir_all(f.parent().unwrap()).unwrap();
            fs::write(f, c).unwrap();
        };
        let link = |target: &str, name: &str| {
            let l = root.join(name);
            fs::create_dir_all(l.parent().unwrap()).unwrap();
            std::os::unix::fs::symlink(root.join(target), l).unwrap();
        };
        // sda1 -> sda, whose device/ has numa_node 1
        fs::create_dir_all(root.join("sys/devices/pci0000:00/0000:00:1f.2/block/sda/sda1"))
            .unwrap();
        w(
            "sys/devices/pci0000:00/0000:00:1f.2/block/sda/device/numa_node",
            "1\n",
        );
        link(
            "sys/devices/pci0000:00/0000:00:1f.2/block/sda/sda1",
            "sys/dev/block/8:1",
        );
        link(
            "sys/devices/pci0000:00/0000:00:1f.2/block/sda",
            "sys/dev/block/8:0",
        );
        assert_eq!(block_node(root, 8, 1), Some(1));
        assert_eq!(block_node(root, 8, 0), Some(1));
        // NVMe namespace: only the PCI device up the tree knows
        fs::create_dir_all(
            root.join("sys/devices/pci0000:00/0000:00:04.0/nvme/nvme0/nvme0n1/nvme0n1p2"),
        )
        .unwrap();
        w("sys/devices/pci0000:00/0000:00:04.0/numa_node", "0\n");
        link(
            "sys/devices/pci0000:00/0000:00:04.0/nvme/nvme0/nvme0n1/nvme0n1p2",
            "sys/dev/block/259:2",
        );
        assert_eq!(block_node(root, 259, 2), Some(0));
        // unknown (-1) is unknown, not node 0
        fs::create_dir_all(root.join("sys/devices/pci0000:00/0000:00:05.0/virtio2/block/vda"))
            .unwrap();
        w("sys/devices/pci0000:00/0000:00:05.0/numa_node", "-1\n");
        link(
            "sys/devices/pci0000:00/0000:00:05.0/virtio2/block/vda",
            "sys/dev/block/252:0",
        );
        assert_eq!(block_node(root, 252, 0), None);
        // no such device
        assert_eq!(block_node(root, 1, 1), None);
    }

    #[test]
    fn pci_node_opt_never_guesses() {
        let (dir, _) = two_sockets();
        assert_eq!(pci_node_opt(dir.path(), "0000:af:00.0"), Some(1));
        assert_eq!(pci_node_opt(dir.path(), "0000:5e:00.0"), None);
        assert_eq!(pci_node_opt(dir.path(), "0000:ff:00.0"), None);
    }
}
