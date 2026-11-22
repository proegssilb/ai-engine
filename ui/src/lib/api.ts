// Typed access to the daemon: ollama-compatible /api and placement /api/aie.

import type { Intent, Tier } from './pending'
export type { Amount, Intent, Share, Tier } from './pending'

export interface ModelTag {
  name: string
  size: number
  modified_at: string
  details: { family?: string; parameter_size?: string; quantization_level?: string; /** replaces the generated one-line description (demo fixtures) */ summary?: string }
}

export interface Running {
  name: string
  size: number
  size_vram: number
  context_length: number
  expires_at: string
}

export interface Device {
  name: string
  description: string
  total: number
  free: number
  gpu: boolean
  pci_bus_id?: string | null
  /** the PCI device's NUMA node; null/absent when the kernel doesn't say */
  numa_node?: number | null
}

/** Latest value plus the min and max over the daemon's 60 s window (bytes). */
export interface Spread {
  latest: number
  min: number
  max: number
}

/** One RAM box (all RAM, or one NUMA node) over the window. Our models' own memory is exact (from the workers); the rest is fuzzy. */
export interface MemBox {
  node: number | null
  total: number
  /** ours | free | other (processes, kernel) | slab (reclaimable) | cache (droppable page cache) */
  segments: Record<'ours' | 'free' | 'other' | 'slab' | 'cache', Spread>
  /** where the four borders between those segments sit, bytes from the left */
  edges: [Spread, Spread, Spread, Spread]
  meminfo: Record<'free' | 'cached' | 'buffers' | 'sreclaimable' | 'shmem', Spread>
  /** loaded models: their disk tier's resident share is page cache (the kernel's) */
  models: {
    model: string
    ours: Spread
    weights: Spread
    kv: Spread
    scratch: Spread
    disk_tier_bytes: number
    disk_tier_resident: Spread
  }[]
}

export interface MemoryWindow {
  interval_secs: number
  window_secs: number
  samples: number
  system: MemBox
  /** only on machines with several NUMA nodes */
  nodes: MemBox[]
}

/** Where a model's host side is pinned and where it really is (aie_worker::numa::HostReport). */
export interface HostReport {
  node: number
  auto: boolean
  reason: string
  sockets: number
  threads: number
  affinity: { cpus: number; nodes: number[]; on_node: boolean; error: string | null }
  policy: { mode: string; node: number; verified: boolean; error: string | null }
  /** node -> bytes of the RAM tier (estimated from sampled pages) */
  ram_tier_by_node: Record<string, number>
  memory_by_node: Record<string, number>
  /** the model file's pages in the page cache, per node (reported, not placed) */
  file_cache_by_node?: Record<string, number>
  /** KV cache, compute buffers and other host allocations per node */
  host_buffers_by_node?: Record<string, number>
  /** host_buffers_by_node told apart (the KV cache and compute scratch as far as measured on that node) */
  kv_cache_by_node?: Record<string, number>
  compute_by_node?: Record<string, number>
  other_buffers_by_node?: Record<string, number>
  ram_tier_exact?: boolean
  problems: string[]
}

export interface DiskInfo {
  path: string
  total: number
  free: number
  /** NUMA node of the block device; null when unknown */
  numa_node?: number | null
  name?: string
}

export interface Machine {
  devices: Device[]
  ram: { total: number; available: number }
  numa: { node: number; total: number; free: number; cpus: string }[]
  disk: DiskInfo
  disks: DiskInfo[]
  cuda_build: boolean
  /** null until the daemon has taken its first sample */
  memory: MemoryWindow | null
  models: MachineModel[]
}

export interface MachineModel {
  model: string
  bytes_by_tier: Record<string, number>
  /** subsystem label -> tier key (gpu0, ram, disk) -> bytes */
  planned?: Record<string, Record<string, number>>
  /** what the loaded context allocated, per device and on the host */
  memory?: Measured[] | null
  n_ctx?: number
  misplaced: number
  store: string | null
  host?: HostReport | null
}

/** One row of llama.cpp's memory breakdown (aie_llama::fit::Measured). */
export interface Measured {
  name: string
  kind: 'device' | 'host' | 'buffer_type'
  total: number
  free: number
  /** weights */
  model: number
  /** KV cache and state */
  context: number
  /** compute scratch */
  compute: number
}

export interface Placement {
  model: string
  anatomy: {
    architecture: string
    supported: boolean
    layers: number
    moe: { experts: number; experts_used: number } | null
    bytes: number
    subsystems: Record<string, number>
  }
  intent: Intent | null
  /** a saved placement that isn't loaded: subsystem -> tier key -> bytes it would use */
  intended?: Record<string, Record<string, number>>
  loaded: null | {
    description: string
    n_ctx: number
    misplaced: number
    load_ms: number
    /** subsystem label -> tier key (gpu0, ram, disk) -> bytes */
    planned: Record<string, Record<string, number>>
    /** llama.cpp's dry-run breakdown per device, before the load (null if not available) */
    measured: Measured[] | null
    /** weights verify found elsewhere than planned, by subsystem and planned tier */
    misplaced_detail?: { subsystem: string; planned: string; actual: string; tensors: number; bytes: number }[]
    /** how the expert cache is split over GPUs (empty when it isn't in use) */
    expert_cache: { gpu: number; bytes: number }[]
    stats: Stats | null
    host: HostReport | null
    warnings: string[]
  }
}

export interface Stats {
  requests: number
  prompt_tokens: number
  completion_tokens: number
  prefill_tps: number
  decode_tps: number
  last_prefill_tps: number
  last_decode_tps: number
  disk_tier_bytes: number
  disk_tier_resident: number
  ram_tier: RamTier | null
}

/** Is the RAM tier staying in RAM: OS-level backstop plus driver features. */
export interface RamTier {
  bytes: number
  resident: number
  swapped: number
  held: boolean
  held_by_os: boolean
  mechanisms: { layer: string; name: string; ok: boolean; detail: string }[]
  environment: string
  fixes: string[]
}

/** One model's pending placement: an intent, or null for automatic (llama.cpp's own fit). */
export interface Edit {
  model: string
  intent: Intent | null
}

export interface DeviceCheck {
  /** CUDA0, ... or RAM */
  name: string
  total: number
  free: number
  reclaimed: number
  available: number
  need: number
  by_model: Record<string, number>
  fits: boolean
}

export interface CheckResult {
  fits: boolean
  problems: string[]
  devices: DeviceCheck[]
  models: Record<string, { planned: Record<string, Record<string, number>>; need: Record<string, number> | null; error: string | null }>
}

export interface ApplyResult {
  applied: boolean
  check: CheckResult
  results: { model: string; ok: boolean; error?: string; misplaced?: number; warnings?: string[]; restored?: boolean }[]
}

export class ApiError extends Error {}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: body === undefined ? {} : { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const text = await r.text()
  const v = text ? JSON.parse(text) : null
  if (!r.ok) {
    const msg = v?.error?.message ?? v?.error ?? `HTTP ${r.status}`
    throw new ApiError(String(msg))
  }
  return v as T
}

export const api = {
  tags: () => call<{ models: ModelTag[] }>('GET', '/api/tags').then((v) => v.models),
  ps: () => call<{ models: Running[] }>('GET', '/api/ps').then((v) => v.models),
  machine: () => call<Machine>('GET', '/api/aie/machine'),
  placement: (model: string) => call<Placement>('GET', `/api/aie/placement?model=${encodeURIComponent(model)}`),
  /** Check pending edits together without changing anything (a dry run per model). */
  check: (edits: Edit[]) => call<CheckResult>('POST', '/api/aie/placement/check', { edits }),
  /** Save, reload and verify every edited model, if the edits fit together. */
  apply: (edits: Edit[]) => call<ApplyResult>('POST', '/api/aie/placement/apply', { edits }),
  savePlacement: (model: string, intent: Intent | null, reload: boolean) =>
    call<{ saved: boolean; unloaded: boolean }>('PUT', '/api/aie/placement', { model, intent, reload }),
  /** Load a model (an empty chat loads it, as in ollama). */
  load: (model: string) => call<unknown>('POST', '/api/chat', { model, messages: [] }),
  unload: (model: string) => call<unknown>('POST', '/api/chat', { model, messages: [], keep_alive: 0 }),
  remove: (model: string) => call<unknown>('DELETE', '/api/delete', { model }),

  /** Pull with progress; resolves when done, rejects on error. */
  async pull(model: string, progress: (status: string, done?: number, total?: number) => void) {
    const r = await fetch('/api/pull', { method: 'POST', body: JSON.stringify({ model, stream: true }) })
    if (!r.ok || !r.body) throw new ApiError(`HTTP ${r.status}`)
    const reader = r.body.getReader()
    const dec = new TextDecoder()
    let buf = ''
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let i
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i)
        buf = buf.slice(i + 1)
        if (!line.trim()) continue
        const v = JSON.parse(line)
        if (v.error) throw new ApiError(v.error)
        progress(v.status, v.completed, v.total)
      }
    }
  },
}

/** Nodes that can host a model (those with CPUs); several means the socket choice matters. */
export function sockets(numa: Machine['numa']): Machine['numa'] {
  return numa.filter((n) => n.cpus.trim() !== '')
}

export function bytes(b: number): string {
  if (b >= 1 << 30) return `${(b / (1 << 30)).toFixed(1)} GiB`
  if (b >= 1 << 20) return `${(b / (1 << 20)).toFixed(0)} MiB`
  return `${(b / 1024).toFixed(0)} KiB`
}

export function tierKey(t: Tier): string {
  return typeof t === 'object' ? `gpu${t.Gpu}` : t.toLowerCase()
}

export function tierLabel(key: string): string {
  if (key.startsWith('gpu')) return `GPU ${key.slice(3)}`
  return key === 'ram' ? 'RAM' : key === 'disk' ? 'Disk' : key
}

export function tierFromKey(key: string): Tier {
  if (key.startsWith('gpu')) return { Gpu: Number(key.slice(3)) }
  return key === 'ram' ? 'Ram' : 'Disk'
}
