// Typed access to the daemon: ollama-compatible /api and placement /api/aie.

export type Tier = { Gpu: number } | 'Ram' | 'Disk'
export type Amount = { Fraction: number } | { Bytes: number } | 'Rest'
export interface Share {
  tier: Tier
  amount: Amount
}

/** aie_plan::Intent (decision D6: shares per tier, by size). */
export interface Intent {
  n_ctx: number
  layers: Share[]
  subsystems: Record<string, Share[]>
  embeddings: Tier | null
  output: Tier | null
  expert_cache_bytes: number
  allow_repack: boolean
  /** socket (NUMA node) for the host side; null/absent = next to its GPU */
  host_node?: number | null
}

export interface ModelTag {
  name: string
  size: number
  modified_at: string
  details: { family?: string; parameter_size?: string; quantization_level?: string }
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
}

/** Latest value plus the min and max over the daemon's 60 s window (bytes). */
export interface Spread {
  latest: number
  min: number
  max: number
}

/** One RAM box (all RAM, or one NUMA node) over the window. */
export interface MemBox {
  node: number | null
  total: number
  /** ours | free | other (processes, kernel) | slab (reclaimable) | cache (droppable page cache) */
  segments: Record<'ours' | 'free' | 'other' | 'slab' | 'cache', Spread>
  /** where the four borders between those segments sit, bytes from the left */
  edges: [Spread, Spread, Spread, Spread]
  meminfo: Record<'free' | 'cached' | 'buffers' | 'sreclaimable' | 'shmem', Spread>
  /** our processes; model '' is the daemon itself */
  models: { model: string; ours: Spread; disk_tier_bytes: number; disk_tier_resident: Spread }[]
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
  problems: string[]
}

export interface Machine {
  devices: Device[]
  ram: { total: number; available: number }
  numa: { node: number; total: number; free: number; cpus: string }[]
  disk: { path: string; total: number; free: number }
  disks: { path: string; total: number; free: number }[]
  cuda_build: boolean
  /** null until the daemon has taken its first sample */
  memory: MemoryWindow | null
  models: { model: string; bytes_by_tier: Record<string, number>; misplaced: number; store: string | null; host?: HostReport | null }[]
}

/** One row of llama.cpp's dry-run memory breakdown (aie_llama::fit::Measured). */
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
  loaded: null | {
    description: string
    n_ctx: number
    misplaced: number
    load_ms: number
    /** subsystem label -> tier key (gpu0, ram, disk) -> bytes */
    planned: Record<string, Record<string, number>>
    /** llama.cpp's dry-run breakdown per device, before the load (null if not available) */
    measured: Measured[] | null
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
