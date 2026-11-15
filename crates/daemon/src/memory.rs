//! A rolling picture of host memory for the machine view's RAM bar.
//!
//! Every 2 s the daemon reads /proc/meminfo (and each NUMA node's meminfo on
//! multi-socket machines) and how much memory its own processes hold: the
//! daemon and one worker per loaded model. It keeps 60 s of samples and
//! reports the latest value plus the window's min and max for each quantity.
//! The UI draws the bar with fuzzy edges between segments whose width is that
//! spread, so a value that jumps about looks blurred and a steady one sharp.
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
            "Cached:" => &mut m.cached,
            "Buffers:" => &mut m.buffers,
            "SReclaimable:" => &mut m.sreclaimable,
            "Shmem:" => &mut m.shmem,
            _ => continue,
        };
        *slot = kb * 1024;
    }
    m
}

/// Memory a process holds that is not a file mapping: `RssAnon + RssShmem`
/// from /proc/PID/status. File-backed pages (the mapped model file, the
/// binary) are page cache already counted in the cache segment, so counting
/// them here would show them twice.
pub fn parse_own_rss(status: &str) -> u64 {
    let kb = |key: &str| -> u64 {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|v| v.split_whitespace().next()?.parse().ok())
            .unwrap_or(0)
    };
    (kb("RssAnon:") + kb("RssShmem:")) * 1024
}

/// Anonymous memory per NUMA node from /proc/PID/numa_maps: mappings with no
/// `file=` and some `anon=` pages, counting each node's `N<i>=pages`.
pub fn parse_numa_maps_anon(text: &str) -> BTreeMap<u32, u64> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let mut anon = false;
        let mut file = false;
        let mut page = 4096u64;
        let mut nodes = Vec::new();
        for tok in line.split_whitespace().skip(2) {
            if tok.starts_with("file=") {
                file = true;
            } else if tok.starts_with("anon=") || tok == "heap" || tok == "stack" {
                anon = true;
            } else if let Some(kb) = tok.strip_prefix("kernelpagesize_kB=") {
                page = kb.parse::<u64>().unwrap_or(4) * 1024;
            } else if let Some((n, pages)) = tok.strip_prefix('N').and_then(|t| t.split_once('='))
                && let (Ok(n), Ok(p)) = (n.parse::<u32>(), pages.parse::<u64>())
            {
                nodes.push((n, p));
            }
        }
        if anon && !file {
            for (n, p) in nodes {
                *out.entry(n).or_default() += p * page;
            }
        }
    }
    out
}

/// One process's (or model's) share of a box.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelMem {
    /// Model name; empty for the daemon itself.
    model: String,
    ours: u64,
    disk_tier_bytes: u64,
    disk_resident: u64,
}

/// What one of our processes holds, as read from /proc.
struct Held {
    model: String,
    rss: u64,
    per_node: BTreeMap<u32, u64>,
    disk: u64,
    resident: u64,
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
        let ours: u64 = self.models.iter().map(|x| x.ours).sum();
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
        let mut procs: Vec<(String, Option<u32>, u64, u64)> = Vec::new();
        for w in workers.list().await {
            let st = tokio::time::timeout(
                Duration::from_secs(1),
                w.client().status(aie_proto::worker::StatusRequest {}),
            )
            .await;
            let (bytes, resident) = match st {
                Ok(Ok(r)) => {
                    let r = r.into_inner();
                    (r.disk_tier_bytes, r.disk_tier_resident)
                }
                _ => (0, 0),
            };
            procs.push((w.name.clone(), w.pid().await, bytes, resident));
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
                "ours": stat(mine().map(|x| x.ours)),
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

/// Read one sample from `root` (`/`; a fake tree in tests). `procs` is each
/// worker's model, pid, disk-tier bytes and how much of it is in RAM.
fn read_sample(root: &Path, procs: &[(String, Option<u32>, u64, u64)]) -> Sample {
    let node_dir = root.join("sys/devices/system/node");
    let mut node_ids: Vec<u32> = std::fs::read_dir(&node_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.strip_prefix("node")?.parse().ok())
        .collect();
    node_ids.sort_unstable();
    let multi = node_ids.len() > 1;

    let mut ours: Vec<Held> = Vec::new();
    let mut add = |model: String, pid: String, disk: u64, resident: u64| {
        let proc = root.join("proc").join(pid);
        let rss = parse_own_rss(&read(proc.join("status")));
        let per_node = if multi {
            parse_numa_maps_anon(&read(proc.join("numa_maps")))
        } else {
            BTreeMap::new()
        };
        ours.push(Held {
            model,
            rss,
            per_node,
            disk,
            resident,
        });
    };
    add(String::new(), "self".into(), 0, 0);
    for (model, pid, disk, resident) in procs {
        if let Some(pid) = pid {
            add(model.clone(), pid.to_string(), *disk, *resident);
        }
    }

    let mem = parse_meminfo(&read(root.join("proc/meminfo")));
    let models = ours
        .iter()
        .map(|h| ModelMem {
            model: h.model.clone(),
            ours: h.rss,
            disk_tier_bytes: h.disk,
            disk_resident: h.resident,
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
                // Which node holds a mapped file's pages isn't visible per
                // file, so a model's resident disk tier is spread over the
                // nodes in proportion to their page cache.
                let share = cache(&mem) as f64 / total_cache.max(1) as f64;
                let models = ours
                    .iter()
                    .map(|h| ModelMem {
                        model: h.model.clone(),
                        ours: h.per_node.get(&n).copied().unwrap_or(0),
                        disk_tier_bytes: (h.disk as f64 * share) as u64,
                        disk_resident: (h.resident as f64 * share) as u64,
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
            parse_meminfo("Node 1 MemTotal:  8 kB\nNode 1 MemFree: 2 kB\nNode 1 Cached: 3 kB\n");
        assert_eq!((node.total, node.free, node.cached), (8192, 2048, 3072));
    }

    #[test]
    fn own_rss_leaves_out_file_pages() {
        let s = "Name:\tx\nRssAnon:\t   100 kB\nRssFile:\t  900 kB\nRssShmem:\t 5 kB\n";
        assert_eq!(parse_own_rss(s), 105 * 1024);
    }

    #[test]
    fn numa_maps_counts_anonymous_mappings_only() {
        let t = "55d0 default heap anon=10 dirty=10 N0=6 N1=4 kernelpagesize_kB=4\n\
                 7f00 default file=/m.gguf mapped=100 mapmax=1 N0=100 kernelpagesize_kB=4\n\
                 7f10 bind:1 anon=2 dirty=2 N1=2 kernelpagesize_kB=2048\n";
        let m = parse_numa_maps_anon(t);
        assert_eq!(m[&0], 6 * 4096);
        assert_eq!(m[&1], 4 * 4096 + 2 * 2048 * 1024);
    }

    #[test]
    fn segments_add_up_and_edges_are_cumulative() {
        let b = BoxSample {
            mem: parse_meminfo(MEMINFO),
            models: vec![ModelMem {
                model: "m".into(),
                ours: 200 * 1024,
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
        w("proc/self/status", "RssAnon: 10 kB\n");
        w("proc/self/numa_maps", "1 default anon=2 N0=2\n");
        w("proc/7/status", "RssAnon: 100 kB\n");
        w("proc/7/numa_maps", "1 default anon=25 N0=5 N1=20\n");
        w(
            "sys/devices/system/node/node0/meminfo",
            "Node 0 MemTotal: 500 kB\nNode 0 Cached: 100 kB\n",
        );
        w(
            "sys/devices/system/node/node1/meminfo",
            "Node 1 MemTotal: 500 kB\nNode 1 Cached: 300 kB\n",
        );
        let s = read_sample(&d, &[("m".into(), Some(7), 1000, 400)]);
        assert_eq!(s.system.models.len(), 2);
        assert_eq!(s.nodes.len(), 2);
        let n1 = &s.nodes[1].1;
        assert_eq!(n1.models[1].ours, 20 * 4096);
        // disk tier spread by page cache: node 1 holds 3/4
        assert_eq!(n1.models[1].disk_resident, 300);
        std::fs::remove_dir_all(d).ok();
    }
}
