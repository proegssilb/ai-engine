//! A rolling picture of host memory for the machine view's RAM bar.
//!
//! Every 2 s the daemon reads /proc/meminfo (and each NUMA node's meminfo on
//! multi-socket machines) and asks each worker what it holds, measured
//! exactly per node: the RAM tier's weights, the KV cache and the compute
//! scratch. Those are "ours" and drawn sharp. The daemon's own memory is not
//! the models', so it stays in "other", like every other process. The
//! window keeps 60 s of samples and reports the latest value plus the min and
//! max for each quantity; the UI draws the borders of the segments that are
//! not exact (free, other, the kernel's cache) blurred by that spread.
//! The point is to show what placement choices cost (page cache the OS gives
//! up, memory other processes hold), not to limit anything.
//!
//! The daemon never touches CUDA; this is all /proc and /sys.

use crate::workers::Manager;
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

/// Fixed on purpose: the UI's blur is meaningful only against a known window.
pub const INTERVAL: Duration = Duration::from_secs(2);
pub const WINDOW: Duration = Duration::from_secs(60);

/// The /proc/meminfo fields the bar needs, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Meminfo {
    pub total: u64,
    pub free: u64,
    pub cached: u64,
    pub buffers: u64,
    pub sreclaimable: u64,
    pub shmem: u64,
}

/// Parse /proc/meminfo or a node's meminfo (whose lines start with `Node N`).
pub fn parse_meminfo(text: &str) -> Meminfo {
    let mut m = Meminfo::default();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let mut key = it.next().unwrap_or("");
        if key == "Node" {
            it.next();
            key = it.next().unwrap_or("");
        }
        let kb: u64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let slot = match key {
            "MemTotal:" => &mut m.total,
            "MemFree:" => &mut m.free,
            // A node's meminfo has no Cached: its page cache is FilePages.
            "Cached:" | "FilePages:" => &mut m.cached,
            "Buffers:" => &mut m.buffers,
            "SReclaimable:" => &mut m.sreclaimable,
            "Shmem:" => &mut m.shmem,
            _ => continue,
        };
        *slot = kb * 1024;
    }
    m
}

/// One process's (or model's) share of a box.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelMem {
    model: String,
    /// The RAM tier's weights, as measured on this node.
    weights: u64,
    /// KV cache on this node.
    kv: u64,
    /// Compute scratch on this node.
    scratch: u64,
    disk_tier_bytes: u64,
    disk_resident: u64,
}

impl ModelMem {
    /// Everything of the model that is its own memory (not page cache).
    fn ours(&self) -> u64 {
        self.weights + self.kv + self.scratch
    }
}

/// A worker as the sampler sees it, from its status.
#[derive(Clone, Debug, Default)]
struct Proc {
    model: String,
    /// Disk-tier bytes and how much of them is in RAM (mincore over the
    /// disk tier's own byte ranges).
    disk: u64,
    resident: u64,
    /// Where the model file's cached pages measurably are. Used only to say
    /// which socket the disk tier's resident share belongs to.
    file_by_node: BTreeMap<u32, u64>,
    /// The worker's exact measurements per node.
    tier_by_node: BTreeMap<u32, u64>,
    kv_by_node: BTreeMap<u32, u64>,
    compute_by_node: BTreeMap<u32, u64>,
}

fn node_map(h: &Value, key: &str) -> BTreeMap<u32, u64> {
    h.get(key)
        .and_then(|v| serde_json::from_value::<BTreeMap<String, u64>>(v.clone()).ok())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(k, v)| Some((k.parse().ok()?, v)))
        .collect()
}

/// One box (all RAM, or one NUMA node) at one instant.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BoxSample {
    mem: Meminfo,
    models: Vec<ModelMem>,
}

/// The bar's segments, in bytes, left to right.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segments {
    ours: u64,
    free: u64,
    /// Other processes and the kernel's memory that can't be dropped.
    other: u64,
    /// Reclaimable kernel slab: the fuzzy border between `other` and `cache`.
    slab: u64,
    /// Page cache the system can drop (file pages, not tmpfs/shared memory).
    cache: u64,
}

impl BoxSample {
    fn segments(&self) -> Segments {
        let m = &self.mem;
        let ours: u64 = self.models.iter().map(|x| x.ours()).sum();
        let cache = (m.cached + m.buffers).saturating_sub(m.shmem);
        let other = m
            .total
            .saturating_sub(ours + m.free + cache + m.sreclaimable);
        Segments {
            ours,
            free: m.free,
            other,
            slab: m.sreclaimable,
            cache,
        }
    }

    /// Where each of the four boundaries sits, as bytes from the left.
    fn edges(&self) -> [u64; 4] {
        let s = self.segments();
        let e1 = s.ours;
        let e2 = e1 + s.free;
        let e3 = e2 + s.other;
        [e1, e2, e3, e3 + s.slab]
    }
}

struct Sample {
    system: BoxSample,
    nodes: Vec<(u32, BoxSample)>,
}

fn stat(values: impl Iterator<Item = u64>) -> Value {
    let v: Vec<u64> = values.collect();
    json!({
        "latest": v.last().copied().unwrap_or(0),
        "min": v.iter().min().copied().unwrap_or(0),
        "max": v.iter().max().copied().unwrap_or(0),
    })
}

/// The rolling window of samples.
#[derive(Default)]
pub struct Sampler {
    window: Mutex<VecDeque<Sample>>,
}

impl Sampler {
    fn push(&self, s: Sample) {
        let mut w = self.window.lock().unwrap();
        w.push_back(s);
        while w.len() > (WINDOW.as_secs() / INTERVAL.as_secs()) as usize {
            w.pop_front();
        }
    }

    /// Take one sample now: read the kernel's numbers and ask each worker for
    /// its disk-tier residency.
    pub async fn sample(&self, workers: &Manager) {
        let mut procs: Vec<Proc> = Vec::new();
        for w in workers.list().await {
            let st = tokio::time::timeout(
                Duration::from_secs(1),
                w.client().status(aie_proto::worker::StatusRequest {}),
            )
            .await;
            let mut p = Proc {
                model: w.name.clone(),
                ..Default::default()
            };
            if let Ok(Ok(r)) = st {
                let r = r.into_inner();
                let h: Value = serde_json::from_str(&r.host_json).unwrap_or(Value::Null);
                p.disk = r.disk_tier_bytes;
                p.resident = r.disk_tier_resident;
                p.file_by_node = node_map(&h, "file_cache_by_node");
                p.tier_by_node = node_map(&h, "ram_tier_by_node");
                p.kv_by_node = node_map(&h, "kv_cache_by_node");
                p.compute_by_node = node_map(&h, "compute_by_node");
            }
            procs.push(p);
        }
        let sample = tokio::task::spawn_blocking(move || read_sample(Path::new("/"), &procs))
            .await
            .ok();
        if let Some(s) = sample {
            self.push(s);
        }
    }

    /// The window as JSON for GET /api/aie/machine.
    pub fn snapshot(&self) -> Value {
        let w = self.window.lock().unwrap();
        let Some(last) = w.back() else {
            return Value::Null;
        };
        let system = box_json(None, w.iter().map(|s| &s.system));
        let nodes: Vec<Value> = last
            .nodes
            .iter()
            .map(|(n, _)| {
                let boxes = w
                    .iter()
                    .filter_map(|s| s.nodes.iter().find(|(i, _)| i == n).map(|(_, b)| b));
                box_json(Some(*n), boxes)
            })
            .collect();
        json!({
            "interval_secs": INTERVAL.as_secs(),
            "window_secs": WINDOW.as_secs(),
            "samples": w.len(),
            "system": system,
            "nodes": nodes,
        })
    }
}

fn box_json<'a>(node: Option<u32>, boxes: impl Iterator<Item = &'a BoxSample> + Clone) -> Value {
    let last = boxes.clone().last().expect("at least one sample");
    let seg = |f: fn(&Segments) -> u64| stat(boxes.clone().map(|b| f(&b.segments())));
    let mem = |f: fn(&Meminfo) -> u64| stat(boxes.clone().map(|b| f(&b.mem)));
    let edge = |i: usize| stat(boxes.clone().map(|b| b.edges()[i]));
    let models: Vec<Value> = last
        .models
        .iter()
        .map(|m| {
            let mine = || {
                boxes
                    .clone()
                    .filter_map(|b| b.models.iter().find(|x| x.model == m.model))
            };
            json!({
                "model": m.model,
                "ours": stat(mine().map(|x| x.ours())),
                "weights": stat(mine().map(|x| x.weights)),
                "kv": stat(mine().map(|x| x.kv)),
                "scratch": stat(mine().map(|x| x.scratch)),
                "disk_tier_bytes": m.disk_tier_bytes,
                "disk_tier_resident": stat(mine().map(|x| x.disk_resident)),
            })
        })
        .collect();
    json!({
        "node": node,
        "total": last.mem.total,
        "segments": {
            "ours": seg(|s| s.ours),
            "free": seg(|s| s.free),
            "other": seg(|s| s.other),
            "slab": seg(|s| s.slab),
            "cache": seg(|s| s.cache),
        },
        // cumulative boundaries: ours | free | other | slab | cache
        "edges": [edge(0), edge(1), edge(2), edge(3)],
        "meminfo": {
            "free": mem(|m| m.free),
            "cached": mem(|m| m.cached),
            "buffers": mem(|m| m.buffers),
            "sreclaimable": mem(|m| m.sreclaimable),
            "shmem": mem(|m| m.shmem),
        },
        "models": models,
    })
}

fn read(path: impl AsRef<Path>) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// Read one sample from `root` (`/`; a fake tree in tests). `procs` is what
/// each worker reported.
fn read_sample(root: &Path, procs: &[Proc]) -> Sample {
    let node_dir = root.join("sys/devices/system/node");
    let mut node_ids: Vec<u32> = std::fs::read_dir(&node_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.strip_prefix("node")?.parse().ok())
        .collect();
    node_ids.sort_unstable();
    let multi = node_ids.len() > 1;

    let sum = |m: &BTreeMap<u32, u64>| m.values().sum::<u64>();
    let mem = parse_meminfo(&read(root.join("proc/meminfo")));
    let models = procs
        .iter()
        .map(|p| ModelMem {
            model: p.model.clone(),
            weights: sum(&p.tier_by_node),
            kv: sum(&p.kv_by_node),
            scratch: sum(&p.compute_by_node),
            disk_tier_bytes: p.disk,
            disk_resident: p.resident,
        })
        .collect();
    let system = BoxSample { mem, models };

    let nodes = if multi {
        let infos: Vec<(u32, Meminfo)> = node_ids
            .iter()
            .map(|&n| {
                (
                    n,
                    parse_meminfo(&read(node_dir.join(format!("node{n}/meminfo")))),
                )
            })
            .collect();
        let cache = |m: &Meminfo| (m.cached + m.buffers).saturating_sub(m.shmem);
        let total_cache: u64 = infos.iter().map(|(_, m)| cache(m)).sum();
        infos
            .into_iter()
            .map(|(n, mem)| {
                let node_share = cache(&mem) as f64 / total_cache.max(1) as f64;
                let models = procs
                    .iter()
                    .map(|p| {
                        // The worker measures which node holds the model file's
                        // pages; without that, spread by each node's page cache.
                        // This only says whose socket the resident share is
                        // reported under: it is the kernel's cache either way.
                        let measured: u64 = sum(&p.file_by_node);
                        let share = if measured > 0 {
                            p.file_by_node.get(&n).copied().unwrap_or(0) as f64 / measured as f64
                        } else {
                            node_share
                        };
                        let at = |m: &BTreeMap<u32, u64>| m.get(&n).copied().unwrap_or(0);
                        ModelMem {
                            model: p.model.clone(),
                            weights: at(&p.tier_by_node),
                            kv: at(&p.kv_by_node),
                            scratch: at(&p.compute_by_node),
                            disk_tier_bytes: (p.disk as f64 * share) as u64,
                            disk_resident: (p.resident as f64 * share) as u64,
                        }
                    })
                    .collect();
                (n, BoxSample { mem, models })
            })
            .collect()
    } else {
        Vec::new()
    };
    Sample { system, nodes }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMINFO: &str = "MemTotal:       1000 kB\nMemFree:         100 kB\nBuffers:          10 kB\nCached:          500 kB\nShmem:            50 kB\nSReclaimable:     40 kB\nSwapTotal: 0 kB\n";

    #[test]
    fn meminfo_system_and_node() {
        let m = parse_meminfo(MEMINFO);
        assert_eq!(
            (m.total, m.free, m.cached),
            (1000 * 1024, 100 * 1024, 500 * 1024)
        );
        assert_eq!(
            (m.buffers, m.shmem, m.sreclaimable),
            (10 * 1024, 50 * 1024, 40 * 1024)
        );
        let node =
            parse_meminfo("Node 1 MemTotal:  8 kB\nNode 1 MemFree: 2 kB\nNode 1 FilePages: 3 kB\n");
        assert_eq!((node.total, node.free, node.cached), (8192, 2048, 3072));
    }

    #[test]
    fn segments_add_up_and_edges_are_cumulative() {
        let b = BoxSample {
            mem: parse_meminfo(MEMINFO),
            models: vec![ModelMem {
                model: "m".into(),
                weights: 150 * 1024,
                kv: 30 * 1024,
                scratch: 20 * 1024,
                disk_tier_bytes: 0,
                disk_resident: 0,
            }],
        };
        let s = b.segments();
        // cache = cached + buffers - shmem = 460
        assert_eq!(s.cache, 460 * 1024);
        assert_eq!(s.ours + s.free + s.other + s.slab + s.cache, 1000 * 1024);
        assert_eq!(b.edges()[3] + s.cache, 1000 * 1024);
    }

    #[test]
    fn window_keeps_sixty_seconds_and_reports_spread() {
        let sampler = Sampler::default();
        for free in 0..40u64 {
            let mut mem = parse_meminfo(MEMINFO);
            mem.free = (free + 1) * 1024;
            sampler.push(Sample {
                system: BoxSample {
                    mem,
                    models: vec![],
                },
                nodes: vec![],
            });
        }
        let v = sampler.snapshot();
        assert_eq!(v["samples"], 30);
        let f = &v["system"]["segments"]["free"];
        // the last 30 of 40 samples: 11..=40 kB
        assert_eq!(
            (f["min"].as_u64(), f["max"].as_u64(), f["latest"].as_u64()),
            (Some(11 * 1024), Some(40 * 1024), Some(40 * 1024))
        );
    }

    #[test]
    fn reads_a_fake_two_node_tree() {
        let d = std::env::temp_dir().join(format!("aie-mem-{}", std::process::id()));
        let w = |p: &str, c: &str| {
            let f = d.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, c).unwrap();
        };
        w("proc/meminfo", MEMINFO);
        w(
            "sys/devices/system/node/node0/meminfo",
            "Node 0 MemTotal: 500 kB\nNode 0 FilePages: 100 kB\n",
        );
        w(
            "sys/devices/system/node/node1/meminfo",
            "Node 1 MemTotal: 500 kB\nNode 1 FilePages: 300 kB\n",
        );
        let mut p = Proc {
            model: "m".into(),
            disk: 1000,
            resident: 400,
            tier_by_node: [(1, 7000), (0, 5)].into(),
            kv_by_node: [(1, 300)].into(),
            compute_by_node: [(1, 200)].into(),
            ..Default::default()
        };
        let s = read_sample(&d, std::slice::from_ref(&p));
        // the system box sums the nodes; only the model's own memory is ours
        assert_eq!(s.system.models.len(), 1);
        assert_eq!(s.system.models[0].ours(), 7005 + 300 + 200);
        assert_eq!(s.nodes.len(), 2);
        let n1 = &s.nodes[1].1;
        assert_eq!(
            (n1.models[0].weights, n1.models[0].kv, n1.models[0].scratch),
            (7000, 300, 200)
        );
        assert_eq!(s.nodes[0].1.models[0].ours(), 5);
        // the disk tier's resident share is the kernel's cache, spread by
        // each node's page cache: node 1 holds 3/4
        assert_eq!(n1.models[0].disk_resident, 300);
        // with the worker's measurement, the split follows where the pages are
        p.file_by_node = [(0u32, 3u64), (1, 1)].into();
        let s = read_sample(&d, &[p]);
        assert_eq!(s.nodes[0].1.models[0].disk_resident, 300);
        assert_eq!(s.nodes[1].1.models[0].disk_resident, 100);
        std::fs::remove_dir_all(d).ok();
    }
}
