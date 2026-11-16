// Fixture data that reproduces the product mockup's own example (docs/mockup.png),
// rendered by the same code as real data: /ui/?demo. Two sockets with 32 cores
// and 64 GiB RAM each; GPUs 0, 1 and 4 on socket 0, GPUs 2 and 3 on socket 1;
// four 4 TB NVMe drives, two per socket; four models placed as in the mockup.
// Models and subsystem kinds the backend can't run (speech) exist only here.

import type { Catalog, CatalogEntry } from './view'
import { anatomyCatalog, GiB } from './view'
import type { HostReport, Machine, MachineModel, Measured, MemBox, ModelTag, Placement, RamTier, Spread, Stats } from './api'

const g = (n: number) => Math.round(n * GiB)

const A = 'Qwen3.8-Flash-Next'
const B = 'Qwen3.8-27B'
const C = 'Whisper large-v3'
const D = 'Kokoro-82M'

const tag = (name: string, summary: string, gb: number): ModelTag => ({
  name,
  size: g(gb),
  modified_at: '2026-11-15T00:00:00Z',
  details: { summary },
})

export const demoTags: ModelTag[] = [
  tag(A, 'Planner · sparse experts · 3-bit', 103),
  tag(B, 'Implementer · dense · 4-bit', 28.9),
  tag(C, 'Speech to text', 3.1),
  tag(D, 'Text to speech · runs on CPU', 0.4),
]

const entries: Record<string, CatalogEntry> = {
  'core layers': { id: 'core', label: 'Core layers', desc: 'The always-on path. Every token passes through it, so it wants the fastest memory.', role: 'core', order: 0 },
  'lookup table': { id: 'lookup', label: 'Lookup table', desc: 'A phrase memory that is read, never computed. Fine on a fast SSD.', role: 'other', order: 2 },
  'draft head': { id: 'draft', label: 'Draft head', desc: 'Guesses a few tokens ahead so the core can check them in one pass.', role: 'other', order: 4 },
  'vision encoder': { id: 'vision', label: 'Vision encoder', desc: 'Turns images into tokens. Idle unless an image arrives.', role: 'other', order: 5 },
  'transformer layers': { id: 'layers', label: 'Transformer layers', short: 'Layers', desc: '', role: 'core', order: 0 },
  'embeddings + output head': { id: 'embed', label: 'Embeddings + output head', short: 'Embed + head', desc: '', role: 'other', order: 2 },
  'audio encoder': { id: 'audio', label: 'Audio encoder', desc: '', role: 'other', order: 0 },
  'text decoder': { id: 'text', label: 'Text decoder', desc: '', role: 'other', order: 1 },
  'acoustic model': { id: 'acoustic', label: 'Acoustic model', desc: '', role: 'other', order: 0 },
  vocoder: { id: 'vocoder', label: 'Vocoder', desc: '', role: 'other', order: 1 },
}

export const demoCatalog: Catalog = (label, model) => {
  if (label === 'kv') {
    const k = anatomyCatalog('kv', model)
    return { ...k, order: model === A ? 3 : 1 }
  }
  if (label === 'routed experts') return { ...anatomyCatalog(label, model), order: 1 }
  // the vision encoder sits after the draft head on B and before it on nothing else
  if (model === B && label === 'vision encoder') return { ...entries[label], order: 3 }
  if (model === B && label === 'draft head') return { ...entries[label], order: 4 }
  return entries[label] ?? anatomyCatalog(label, model)
}

const row = (n: number, kv: number, free: number): Measured => ({ name: `CUDA${n}`, kind: 'device', total: 0, free, model: 0, context: g(kv), compute: 0 })
const gpuRow = (n: number, kv: number, total: number, used: number): Measured => ({ ...row(n, kv, g(total - used)), total: g(total) })

const sp = (v: number, d = 0): Spread => ({ latest: g(v), min: g(v - d), max: g(v + d) })
const spb = (v: number): Spread => ({ latest: v, min: v, max: v })

function host(node: number, tier: number, threads: number): HostReport {
  return {
    node,
    auto: false,
    reason: 'chosen',
    sockets: 2,
    threads,
    affinity: { cpus: 32, nodes: [node], on_node: true, error: null },
    policy: { mode: 'bind', node, verified: true, error: null },
    ram_tier_by_node: { [node]: g(tier) },
    ram_tier_exact: true,
    memory_by_node: { [node]: g(tier) },
    file_cache_by_node: { 0: g(4) },
    host_buffers_by_node: {},
    kv_cache_by_node: {},
    compute_by_node: {},
    other_buffers_by_node: {},
    problems: [],
  }
}

const mm = (model: string, planned: Record<string, Record<string, number>>, memory: Measured[], h: HostReport | null, store: string | null): MachineModel => ({
  model,
  planned: Object.fromEntries(Object.entries(planned).map(([k, v]) => [k, Object.fromEntries(Object.entries(v).map(([t, b]) => [t, g(b)]))])),
  bytes_by_tier: {},
  memory,
  n_ctx: 8192,
  misplaced: 0,
  store,
  host: h,
})

function memBox(node: number, total: number, ours: number, free: number, other: number, cache: number, tier: { model: string; resident: number }[]): MemBox {
  const slab = 0.5
  const e0 = ours
  const e1 = e0 + free
  const e2 = e1 + other
  const e3 = e2 + slab
  return {
    node,
    total: g(total),
    segments: { ours: sp(ours), free: sp(free, 1.2), other: sp(other, 0.6), slab: sp(slab), cache: sp(cache, 1.5) },
    edges: [sp(e0), sp(e1, 1.2), sp(e2, 1.8), sp(e3, 1.8)],
    meminfo: { free: sp(free), cached: sp(cache), buffers: sp(0.1), sreclaimable: sp(slab), shmem: sp(0.1) },
    models: tier.map((t) => ({
      model: t.model,
      ours: sp(0),
      weights: sp(0),
      kv: sp(0),
      scratch: sp(0),
      disk_tier_bytes: g(35),
      disk_tier_resident: sp(t.resident, 0.5),
    })),
  }
}

const disk = (i: number, node: number, used: number) => ({ path: `/mnt/nvme${i}`, name: `NVMe ${i}`, total: 4e12, free: 4e12 - g(used), numa_node: node })

export const demoMachine: Machine = {
  devices: [
    { name: 'CUDA0', description: 'GPU 32 GiB', total: g(32), free: g(15), gpu: true, numa_node: 0 },
    { name: 'CUDA1', description: 'GPU 32 GiB', total: g(32), free: g(8), gpu: true, numa_node: 0 },
    { name: 'CUDA2', description: 'GPU 32 GiB', total: g(32), free: g(17.5), gpu: true, numa_node: 1 },
    { name: 'CUDA3', description: 'GPU 32 GiB', total: g(32), free: g(17.6), gpu: true, numa_node: 1 },
    { name: 'CUDA4', description: 'GPU 16 GiB', total: g(16), free: g(11.9), gpu: true, numa_node: 0 },
  ],
  ram: { total: g(128), available: g(100) },
  numa: [
    { node: 0, total: g(64), free: g(30), cpus: '0-31' },
    { node: 1, total: g(64), free: g(58), cpus: '32-63' },
  ],
  disk: disk(0, 0, 35),
  disks: [disk(0, 0, 35), disk(1, 0, 0), disk(2, 1, 0), disk(3, 1, 0)],
  cuda_build: true,
  memory: {
    interval_secs: 2,
    window_secs: 60,
    samples: 30,
    system: memBox(0, 128, 20.4, 88, 10, 8, []),
    nodes: [memBox(0, 64, 20, 30, 6, 7.5, [{ model: A, resident: 3 }]), memBox(1, 64, 0.4, 58, 4.6, 0.5, [])],
  },
  models: [
    mm(
      A,
      { 'core layers': { gpu0: 5 }, 'routed experts': { gpu1: 24, ram: 20, disk: 6 }, 'lookup table': { disk: 29 }, 'draft head': { gpu0: 6 }, 'vision encoder': { gpu4: 1 } },
      [gpuRow(0, 6, 32, 17), gpuRow(1, 0, 32, 24), gpuRow(4, 0, 16, 4.1)],
      host(0, 20, 32),
      '/mnt/nvme0',
    ),
    mm(
      B,
      { 'transformer layers': { gpu2: 8, gpu3: 8 }, 'embeddings + output head': { gpu2: 1.5 }, 'vision encoder': { gpu3: 0.9 }, 'draft head': { gpu3: 0.5 } },
      [gpuRow(2, 5, 32, 14.5), gpuRow(3, 5, 32, 14.4)],
      null,
      null,
    ),
    mm(C, { 'audio encoder': { gpu4: 1.3 }, 'text decoder': { gpu4: 1.8 } }, [], null, null),
    mm(D, { 'acoustic model': { ram: 0.3 }, vocoder: { ram: 0.1 } }, [], host(1, 0.4, 32), null),
  ],
}

const placement = (model: string, arch: string, layers: number, bytes: number, subs: Record<string, number>, moe: boolean): Placement => ({
  model,
  anatomy: {
    architecture: arch,
    supported: true,
    layers,
    moe: moe ? { experts: 256, experts_used: 8 } : null,
    bytes: g(bytes),
    subsystems: Object.fromEntries(Object.entries(subs).map(([k, v]) => [k, g(v)])),
  },
  intent: null,
  loaded: null,
})

export const demoPlacements: Record<string, Placement> = {
  [A]: placement(A, 'demo', 48, 103, { 'core layers': 5, 'routed experts': 50, 'lookup table': 29, 'draft head': 6, 'vision encoder': 1 }, true),
  [B]: placement(B, 'demo', 64, 28.9, { 'transformer layers': 16, 'embeddings + output head': 1.5, 'vision encoder': 0.9, 'draft head': 0.5 }, false),
  [C]: placement(C, 'demo', 32, 3.1, { 'audio encoder': 1.3, 'text decoder': 1.8 }, false),
  [D]: placement(D, 'demo', 12, 0.4, { 'acoustic model': 0.3, vocoder: 0.1 }, false),
}
// the demo is a picture: nothing to verify, so show the models as loaded without a report
for (const m of demoMachine.models) {
  demoPlacements[m.model].loaded = {
    description: 'demo fixture',
    n_ctx: 8192,
    misplaced: 0,
    load_ms: 0,
    planned: m.planned ?? {},
    measured: m.memory ?? null,
    expert_cache: [],
    stats: null,
    host: m.host ?? null,
    warnings: [],
  }
}

// What the demo shows besides placement: A is held and verified with a few
// requests behind it, B has weights verify found elsewhere than planned, D's
// RAM tier is not held (with the fixes), C is loaded but idle.
const stats = (requests: number, pre: number, dec: number, tier: RamTier | null, diskBytes = 0, diskResident = 0): Stats => ({
  requests,
  prompt_tokens: requests * 120,
  completion_tokens: requests * 80,
  prefill_tps: pre,
  decode_tps: dec,
  last_prefill_tps: pre,
  last_decode_tps: dec,
  disk_tier_bytes: diskBytes,
  disk_tier_resident: diskResident,
  ram_tier: tier,
})
const heldTier: RamTier = {
  bytes: g(20),
  resident: g(20),
  swapped: 0,
  held: true,
  held_by_os: true,
  mechanisms: [
    { layer: 'os', name: 'no swap', ok: true, detail: 'swap is off' },
    { layer: 'os', name: 'swap forbidden', ok: true, detail: 'pages are locked' },
  ],
  environment: 'demo',
  fixes: [],
}
const looseTier: RamTier = {
  bytes: g(0.4),
  resident: g(0.4),
  swapped: 0,
  held: false,
  held_by_os: false,
  mechanisms: [
    { layer: 'os', name: 'no swap', ok: false, detail: 'swap is on' },
    { layer: 'os', name: 'swap forbidden', ok: false, detail: 'the memlock limit is too low' },
  ],
  environment: 'demo',
  fixes: ['Turn swap off (sudo swapoff -a), or', 'raise the memlock limit for the daemon (LimitMEMLOCK=infinity in its service file)'],
}
const L = (m: string) => demoPlacements[m].loaded!
L(A).stats = stats(12, 450, 38.2, heldTier, g(35), g(3))
L(B).stats = stats(4, 1210, 22.4, null)
L(B).misplaced = 8
L(B).misplaced_detail = [{ subsystem: 'transformer layers', planned: 'gpu3', actual: 'a CUDA_Host buffer', tensors: 8, bytes: g(8) }]
L(C).stats = stats(0, 0, 0, null)
L(D).stats = stats(31, 95, 61.5, looseTier)
L(D).warnings = ['0.4 GiB of the RAM tier may be swapped out']
