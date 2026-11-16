// Pending placement edits of one model: shares per tier for its layers and for
// its big per-layer subsystem (experts or feed-forward), by size or share (D6).
// Nothing is saved until "Apply placement".

import { tierFromKey, tierKey, type Intent, type Placement, type Share } from './api'
import type { ModelView } from './view'

export type Pct = Record<string, number>

export function toPercents(shares: Share[], keys: string[], total: number): Pct {
  const out: Pct = Object.fromEntries(keys.map((k) => [k, 0]))
  let rest: string | null = null
  for (const s of shares) {
    const k = tierKey(s.tier)
    if (s.amount === 'Rest') rest = k
    else if ('Fraction' in s.amount) out[k] = Math.round(s.amount.Fraction * 1000) / 10
    else out[k] = Math.round((s.amount.Bytes / Math.max(total, 1)) * 1000) / 10
  }
  if (rest) out[rest] = Math.max(0, Math.round((100 - Object.values(out).reduce((a, b) => a + b, 0)) * 10) / 10)
  return out
}

export function toShares(pct: Pct, keys: string[]): Share[] {
  const used = keys.filter((k) => (pct[k] ?? 0) > 0)
  return used.map((k, i) => ({
    tier: tierFromKey(k),
    amount: i === used.length - 1 ? ('Rest' as const) : { Fraction: (pct[k] ?? 0) / 100 },
  }))
}

export const pctOk = (pct: Pct, keys: string[]) => Math.abs(keys.reduce((s, k) => s + (pct[k] ?? 0), 0) - 100) < 0.5

/** A box key's tier key: gpuN, ram (any socket), disk (any store). */
export const tierOfBox = (box: string) => (box.startsWith('gpu') ? box : box.startsWith('ram') ? 'ram' : 'disk')

export class ModelEdit {
  mode = $state<'auto' | 'custom'>('auto')
  layerPct = $state<Pct>({})
  bulkPct = $state<Pct>({})
  cacheGiB = $state(0)
  nCtx = $state(4096)
  /** socket for the host side; null = automatic (next to its GPU) */
  hostNode = $state<number | null>(null)
  /** edited since the last load of the saved placement: refreshes keep their hands off */
  dirty = $state(false)

  /** Take the saved placement, or what the model is placed as now. */
  seed(p: Placement, gpus: number, mv: ModelView | undefined) {
    if (this.dirty) return
    const gpuKeys = Array.from({ length: gpus }, (_, i) => `gpu${i}`)
    const layerTiers = [...gpuKeys, 'ram']
    const bulkTiers = [...gpuKeys, 'ram', 'disk']
    const bulkLabel = p.anatomy.moe ? 'routed experts' : 'dense ffn'
    const bulkKey = p.anatomy.moe ? 'RoutedExperts' : 'DenseFfn'
    const layerBytes = Object.entries(p.anatomy.subsystems)
      .filter(([k]) => !['embeddings', 'output head', 'other', bulkLabel].includes(k))
      .reduce((s, [, b]) => s + b, 0)
    const i = p.intent
    if (i) {
      this.mode = 'custom'
      this.nCtx = i.n_ctx
      this.hostNode = i.host_node ?? null
      this.cacheGiB = i.expert_cache_bytes / (1 << 30)
      this.layerPct = toPercents(i.layers, layerTiers, layerBytes)
      const bulkTotal = p.anatomy.subsystems[bulkLabel] ?? 0
      this.bulkPct = i.subsystems[bulkKey] ? toPercents(i.subsystems[bulkKey], bulkTiers, bulkTotal) : { ...this.layerPct, disk: 0 }
      return
    }
    this.mode = 'auto'
    this.hostNode = null
    this.nCtx = p.loaded?.n_ctx ?? 4096
    // start from where the pieces are now, so an edit changes what you see
    const from = (role: 'core' | 'bulk', keys: string[]): Pct | null => {
      const sv = mv?.subs.find((s) => s.role === role)
      if (!sv || !sv.dests.length) return null
      const out: Pct = Object.fromEntries(keys.map((k) => [k, 0]))
      for (const d of sv.dests) out[tierOfBox(d.box)] = (out[tierOfBox(d.box)] ?? 0) + (100 * d.bytes) / Math.max(sv.bytes, 1)
      for (const k of keys) out[k] = Math.round(out[k] * 10) / 10
      const sum = keys.reduce((s, k) => s + out[k], 0)
      const big = keys.reduce((a, k) => (out[k] > out[a] ? k : a), keys[0])
      out[big] = Math.round((out[big] + 100 - sum) * 10) / 10
      return out
    }
    const home = gpus > 0 ? 'gpu0' : 'ram'
    const dflt = (keys: string[]): Pct => Object.fromEntries(keys.map((k) => [k, k === home ? 100 : 0]))
    this.layerPct = from('core', layerTiers) ?? dflt(layerTiers)
    this.bulkPct = from('bulk', bulkTiers) ?? dflt(bulkTiers)
  }

  /** The intent these edits mean (custom mode), or null for automatic. */
  intent(p: Placement, gpus: number, sockets: number): Intent | null {
    if (this.mode !== 'custom') return null
    const gpuKeys = Array.from({ length: gpus }, (_, i) => `gpu${i}`)
    const layerTiers = [...gpuKeys, 'ram']
    const bulkTiers = [...gpuKeys, 'ram', 'disk']
    if (!pctOk(this.layerPct, layerTiers) || !pctOk(this.bulkPct, bulkTiers)) {
      throw new Error('Each split has to add up to 100%.')
    }
    return {
      n_ctx: this.nCtx,
      layers: toShares(this.layerPct, layerTiers),
      subsystems: { [p.anatomy.moe ? 'RoutedExperts' : 'DenseFfn']: toShares(this.bulkPct, bulkTiers) },
      embeddings: null,
      output: null,
      expert_cache_bytes: Math.round(this.cacheGiB * (1 << 30)),
      allow_repack: false,
      host_node: sockets > 1 ? this.hostNode : null,
    }
  }
}
