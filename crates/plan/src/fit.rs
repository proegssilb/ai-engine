//! Does a plan fit the machine?
//!
//! Weights, KV cache and expert cache are computed exactly from the anatomy.
//! The rest ([`Overheads`]) are estimates from the GPU spike, to be replaced by
//! llama.cpp's dry run (`no_alloc`) before a plan is applied (architecture.md).
//! The disk tier never counts against RAM: mapped pages can always be dropped.

use crate::{Plan, Tier};
use aie_anatomy::{Anatomy, Subsystem};
use std::fmt;

const MIB: u64 = 1 << 20;

/// Memory available on one device.
#[derive(Clone, Debug, PartialEq)]
pub struct Capacity {
    pub name: String,
    pub total: u64,
    /// Free right now, before this model loads.
    pub free: u64,
}

/// The machine, as far as placement cares.
#[derive(Clone, Debug, PartialEq)]
pub struct Machine {
    /// GPUs by index (`CUDA0` first).
    pub gpus: Vec<Capacity>,
    pub ram: Capacity,
}

/// Memory llama.cpp needs besides weights, KV cache and expert cache.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overheads {
    /// CUDA context on each GPU the worker uses. Not yet measured on the V100s.
    pub cuda_context: u64,
    /// Compute (scratch) buffer per GPU in use. Measured 80-360 MiB for a
    /// 512-token batch on Qwen3-30B-A3B.
    pub gpu_compute: u64,
    /// Extra scratch on GPU 0 when host weights are offloaded for big batches.
    /// Measured ~140 MiB.
    pub op_offload: u64,
    /// Host compute buffer.
    pub cpu_compute: u64,
    /// Bytes per KV-cache element (2 = f16, llama.cpp's default).
    pub kv_elem_bytes: u64,
}

impl Default for Overheads {
    fn default() -> Self {
        Overheads { cuda_context: 400 * MIB, gpu_compute: 512 * MIB, op_offload: 192 * MIB, cpu_compute: 256 * MIB, kv_elem_bytes: 2 }
    }
}

/// What a plan needs on one device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Need {
    pub weights: u64,
    pub kv_cache: u64,
    pub expert_cache: u64,
    pub overhead: u64,
}

impl Need {
    pub fn total(&self) -> u64 {
        self.weights + self.kv_cache + self.expert_cache + self.overhead
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeviceFit {
    pub tier: Tier,
    pub name: String,
    pub need: Need,
    pub free: u64,
}

impl DeviceFit {
    pub fn fits(&self) -> bool {
        self.need.total() <= self.free
    }
}

/// Per-device needs. Devices the plan doesn't touch are left out.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub devices: Vec<DeviceFit>,
    /// Bytes on the disk tier (mapped, evictable; not counted against RAM).
    pub disk_bytes: u64,
}

impl Report {
    pub fn fits(&self) -> bool {
        self.devices.iter().all(DeviceFit::fits)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = |b: u64| b as f64 / MIB as f64;
        for d in &self.devices {
            writeln!(
                f,
                "{:<8} {:>9.0} MiB needed of {:>9.0} free (weights {:.0}, KV {:.0}, expert cache {:.0}, overhead {:.0}){}",
                d.name,
                m(d.need.total()),
                m(d.free),
                m(d.need.weights),
                m(d.need.kv_cache),
                m(d.need.expert_cache),
                m(d.need.overhead),
                if d.fits() { "" } else { "  DOES NOT FIT" }
            )?;
        }
        write!(f, "disk     {:>9.0} MiB mapped (may be dropped from RAM)", m(self.disk_bytes))
    }
}

/// Work out what the plan needs on each device.
pub fn check(a: &Anatomy, plan: &Plan, machine: &Machine, o: &Overheads) -> Report {
    let n_gpus = machine.gpus.len();
    let mut gpu = vec![Need::default(); n_gpus];
    let mut gpu_used = vec![false; n_gpus];
    let mut ram = Need { overhead: o.cpu_compute, ..Need::default() };
    let mut disk = 0u64;

    let mut add = |tier: Tier, bytes: u64, f: fn(&mut Need) -> &mut u64| match tier {
        Tier::Gpu(i) => {
            if let Some(n) = gpu.get_mut(i as usize) {
                *f(n) += bytes;
                gpu_used[i as usize] = true;
            }
        }
        Tier::Ram => *f(&mut ram) += bytes,
        Tier::Disk => disk += bytes,
    };

    let mut host_weights = false;
    for p in &a.parts {
        let t = plan.tensors.get(&p.name).copied().unwrap_or(Tier::Disk);
        host_weights |= p.layer.is_some() && !matches!(t, Tier::Gpu(_));
        add(t, p.bytes, |n| &mut n.weights);
    }

    // KV cache lives with each layer's home; a RAM-homed layer keeps it in RAM.
    if let Some(kv) = a.kv {
        let per_layer = kv.bytes_per_token(o.kv_elem_bytes) / u64::from(kv.layers.max(1)) * plan.n_ctx;
        for &home in &plan.layer_home {
            add(home, per_layer, |n| &mut n.kv_cache);
        }
    }

    // llama.cpp's expert cache: only used if some layer has all its routed
    // experts in host memory; split across GPUs like the layers.
    if plan.expert_cache_bytes > 0 && expert_cache_active(a, plan) {
        let homes: Vec<u32> = plan.layer_home.iter().filter_map(|t| if let Tier::Gpu(i) = t { Some(*i) } else { None }).collect();
        for i in 0..n_gpus as u32 {
            let share = homes.iter().filter(|&&h| h == i).count() as u64;
            if share > 0 {
                add(Tier::Gpu(i), plan.expert_cache_bytes * share / homes.len() as u64, |n| &mut n.expert_cache);
            }
        }
    }

    for (i, n) in gpu.iter_mut().enumerate() {
        if gpu_used[i] {
            n.overhead += o.cuda_context + o.gpu_compute;
        }
    }
    if host_weights && n_gpus > 0 {
        // the scheduler offloads big batches to the first GPU
        gpu[0].overhead += o.op_offload;
        if !gpu_used[0] {
            gpu[0].overhead += o.cuda_context;
            gpu_used[0] = true;
        }
    }

    let mut devices: Vec<DeviceFit> = (0..n_gpus)
        .filter(|&i| gpu_used[i])
        .map(|i| DeviceFit { tier: Tier::Gpu(i as u32), name: machine.gpus[i].name.clone(), need: gpu[i], free: machine.gpus[i].free })
        .collect();
    devices.push(DeviceFit { tier: Tier::Ram, name: machine.ram.name.clone(), need: ram, free: machine.ram.free });
    Report { devices, disk_bytes: disk }
}

/// Whether llama.cpp will enable the expert cache: some layer must have all of
/// its routed experts in host memory (RAM or the mapped file).
pub fn expert_cache_active(a: &Anatomy, plan: &Plan) -> bool {
    (0..a.layers).any(|l| {
        let mut parts = a.parts_of(Subsystem::RoutedExperts).filter(|p| p.layer == Some(l)).peekable();
        parts.peek().is_some() && parts.all(|p| !matches!(plan.tensors.get(&p.name), Some(Tier::Gpu(_))))
    })
}
