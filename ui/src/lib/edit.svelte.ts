// Pending placement edits of one model, as reactive state. The rules (what an
// edit means as an intent, moving pieces, putting a whole model somewhere) are
// in pending.ts. Nothing is saved until "Apply placement".

import type { Placement } from './api'
import { bulkTiers, layerTiers, newState, tierKey, tierOfBox, toIntent, toPercents, type Ctx, type EditState, type Intent, type Pct } from './pending'
import type { ModelView } from './view'

export { tierOfBox }
export type { Pct }

export class ModelEdit implements EditState {
  mode = $state<'auto' | 'custom'>('auto')
  layerPct = $state<Pct>({})
  bulkPct = $state<Pct>({})
  cacheGiB = $state(0)
  nCtx = $state(4096)
  /** socket for the host side; null = automatic (next to its GPU) */
  hostNode = $state<number | null>(null)
  embeddings = $state<string | null>(null)
  output = $state<string | null>(null)
  /** edited since the saved placement was loaded: refreshes keep their hands off */
  dirty = $state(false)

  /** Take the saved placement, or what the model is placed as now. */
  seed(p: Placement, gpus: number, mv: ModelView | undefined) {
    if (this.dirty) return
    const lt = layerTiers(gpus)
    const bt = bulkTiers(gpus)
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
      this.layerPct = toPercents(i.layers, lt, layerBytes)
      const bulkTotal = p.anatomy.subsystems[bulkLabel] ?? 0
      this.bulkPct = i.subsystems[bulkKey] ? toPercents(i.subsystems[bulkKey], bt, bulkTotal) : { ...this.layerPct, disk: 0 }
      this.embeddings = i.embeddings ? tierKey(i.embeddings as never) : null
      this.output = i.output ? tierKey(i.output as never) : null
      return
    }
    this.mode = 'auto'
    this.hostNode = null
    this.embeddings = null
    this.output = null
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
    this.layerPct = from('core', lt) ?? dflt(lt)
    this.bulkPct = from('bulk', bt) ?? dflt(bt)
  }

  /** The intent these edits mean (custom mode), or null for automatic. */
  intent(ctx: Ctx): Intent | null {
    return toIntent(this, ctx)
  }

  /** The state as plain data (what a check or a test sees). */
  plain(): EditState {
    return {
      mode: this.mode,
      nCtx: this.nCtx,
      cacheGiB: this.cacheGiB,
      hostNode: this.hostNode,
      layerPct: { ...this.layerPct },
      bulkPct: { ...this.bulkPct },
      embeddings: this.embeddings,
      output: this.output,
    }
  }
}

export { newState }
