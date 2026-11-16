// Pending placement edits and the intent they mean, as plain data and pure
// functions (no components, no network), so they can be exercised on their
// own. An edit state describes one model's placement the way the model card
// edits it: shares per tier for its layers and for its big per-layer
// subsystem (experts or feed-forward), a tier for the embeddings and for the
// output head, a context length and a socket. `toIntent` turns it into the
// intent the daemon takes (aie_plan::Intent); `null` means automatic.

export type Pct = Record<string, number>
export type Role = 'layers' | 'bulk' | 'embeddings' | 'output'

/** aie_plan::Tier as JSON */
export type Tier = { Gpu: number } | 'Ram' | 'Disk'
export type Amount = { Fraction: number } | { Bytes: number } | 'Rest'
export interface Share {
  tier: Tier
  amount: Amount
}
export interface Intent {
  n_ctx: number
  layers: Share[]
  subsystems: Record<string, Share[]>
  embeddings: Tier | null
  output: Tier | null
  expert_cache_bytes: number
  allow_repack: boolean
  host_node?: number | null
}

export interface EditState {
  mode: 'auto' | 'custom'
  nCtx: number
  cacheGiB: number
  /** socket for the host side; null = automatic (next to its GPU) */
  hostNode: number | null
  layerPct: Pct
  bulkPct: Pct
  /** tier key (gpuN, ram, disk) or null = llama.cpp's choice */
  embeddings: string | null
  output: string | null
}

export function newState(): EditState {
  return { mode: 'auto', nCtx: 4096, cacheGiB: 0, hostNode: null, layerPct: {}, bulkPct: {}, embeddings: null, output: null }
}

/** What the planner needs to know about the model and the machine. */
export interface Ctx {
  /** sparse experts (the bulk subsystem is the expert bank) rather than dense feed-forward */
  moe: boolean
  gpus: number
  /** sockets that can host a model; the host socket only matters with several */
  sockets: number
}

export const gpuKeys = (gpus: number) => Array.from({ length: gpus }, (_, i) => `gpu${i}`)
export const layerTiers = (gpus: number) => [...gpuKeys(gpus), 'ram']
export const bulkTiers = (gpus: number) => [...gpuKeys(gpus), 'ram', 'disk']

export function tierFromKey(key: string): Tier {
  if (key.startsWith('gpu')) return { Gpu: Number(key.slice(3)) }
  return key === 'ram' ? 'Ram' : 'Disk'
}
export function tierKey(t: Tier): string {
  return typeof t === 'object' ? `gpu${t.Gpu}` : t.toLowerCase()
}

/** A box key's tier: gpuN, ram (any socket), disk (any store). */
export const tierOfBox = (box: string) => (box.startsWith('gpu') ? box : box.startsWith('ram') ? 'ram' : 'disk')
/** The socket a RAM or CPU box belongs to. */
export const nodeOfBox = (box: string): number | null => {
  const m = /^(?:ram|cpu)(\d+)$/.exec(box)
  return m ? Number(m[1]) : null
}

const round1 = (v: number) => Math.round(v * 10) / 10

export function toPercents(shares: Share[], keys: string[], total: number): Pct {
  const out: Pct = Object.fromEntries(keys.map((k) => [k, 0]))
  let rest: string | null = null
  for (const s of shares) {
    const k = tierKey(s.tier)
    if (s.amount === 'Rest') rest = k
    else if ('Fraction' in s.amount) out[k] = round1(s.amount.Fraction * 100)
    else out[k] = round1((s.amount.Bytes / Math.max(total, 1)) * 100)
  }
  if (rest) out[rest] = Math.max(0, round1(100 - Object.values(out).reduce((a, b) => a + b, 0)))
  return out
}

/** Shares for the tiers that have any; the last one takes what the others leave. */
export function toShares(pct: Pct, keys: string[]): Share[] {
  const used = keys.filter((k) => (pct[k] ?? 0) > 0)
  return used.map((k, i) => ({
    tier: tierFromKey(k),
    amount: i === used.length - 1 ? ('Rest' as const) : { Fraction: (pct[k] ?? 0) / 100 },
  }))
}

export const pctOk = (pct: Pct, keys: string[]) => Math.abs(keys.reduce((s, k) => s + (pct[k] ?? 0), 0) - 100) < 0.5

/** The intent these edits mean: null for automatic (llama.cpp's own fit). Throws when a split doesn't make 100%. */
export function toIntent(st: EditState, ctx: Ctx): Intent | null {
  if (st.mode !== 'custom') return null
  const lt = layerTiers(ctx.gpus)
  const bt = bulkTiers(ctx.gpus)
  if (!pctOk(st.layerPct, lt) || !pctOk(st.bulkPct, bt)) throw new Error('Each split has to add up to 100%.')
  return {
    n_ctx: st.nCtx,
    layers: toShares(st.layerPct, lt),
    subsystems: { [ctx.moe ? 'RoutedExperts' : 'DenseFfn']: toShares(st.bulkPct, bt) },
    embeddings: st.embeddings ? tierFromKey(st.embeddings) : null,
    output: st.output ? tierFromKey(st.output) : null,
    expert_cache_bytes: Math.round(st.cacheGiB * (1 << 30)),
    allow_repack: false,
    host_node: ctx.sockets > 1 ? st.hostNode : null,
  }
}

/** What is being dragged, or put somewhere by a menu. */
export type Subject = Role | 'model' | 'none'

export interface BoxInfo {
  kind: 'cpu' | 'ram' | 'gpu' | 'disk'
}

/** Why `subject` can't go onto `toBox`, or null when it can. */
export function dropReason(
  subject: Subject,
  fromBox: string | null,
  toBox: string,
  boxes: Record<string, BoxInfo | undefined>,
  opts: { storeBox: string | null },
): string | null {
  const b = boxes[toBox]
  if (!b) return 'There is no such device.'
  if (b.kind === 'cpu') return 'CPU boxes only run what sits in RAM; they hold nothing.'
  if (subject === 'none') return 'This can not be moved on its own.'
  if (fromBox === toBox) return 'It is already here.'
  if (b.kind === 'disk') {
    if (subject === 'model') return 'A whole model can not be put on disk: its layers need a GPU or RAM.'
    if (subject === 'layers') return 'Core layers can not go on disk: attention and the KV cache need a GPU or RAM.'
    if (opts.storeBox && opts.storeBox !== toBox) return `The model's file is on ${opts.storeBox.replace('disk', 'disk ')}; its disk tier can only be there.`
  }
  return null
}

function setNode(st: EditState, toBox: string, sockets: number) {
  const n = nodeOfBox(toBox)
  if (n !== null && sockets > 1) st.hostNode = n
}

/** Move one piece of a subsystem (or all of it, when it is whole) to another box; pieces already there merge. */
export function movePiece(st: EditState, role: Role, fromBox: string, toBox: string, sockets = 1): void {
  st.mode = 'custom'
  const from = tierOfBox(fromBox)
  const to = tierOfBox(toBox)
  if (role === 'embeddings' || role === 'output') {
    st[role] = to
  } else {
    const pct = role === 'layers' ? st.layerPct : st.bulkPct
    if (from !== to) {
      pct[to] = round1((pct[to] ?? 0) + (pct[from] ?? 0))
      pct[from] = 0
    }
  }
  // All RAM-tier memory of a model is on one socket: moving to another socket's RAM moves the socket.
  if (to === 'ram') setNode(st, toBox, sockets)
}

export type Target = { kind: 'gpu'; index: number } | { kind: 'ram'; node: number } | { kind: 'gpus'; weights: number[] }

/** Put the whole model on one tier: layers and everything that follows them, the embeddings and the output head. */
export function putAll(st: EditState, target: Target, ctx: Ctx): void {
  st.mode = 'custom'
  const lt = layerTiers(ctx.gpus)
  const bt = bulkTiers(ctx.gpus)
  const zero = (keys: string[]): Pct => Object.fromEntries(keys.map((k) => [k, 0]))
  const layers = zero(lt)
  let tier: string
  if (target.kind === 'gpu') {
    layers[`gpu${target.index}`] = 100
    tier = `gpu${target.index}`
  } else if (target.kind === 'ram') {
    layers.ram = 100
    tier = 'ram'
    if (ctx.sockets > 1) st.hostNode = target.node
  } else {
    // across all GPUs in proportion to the memory each has available
    const w = target.weights.slice(0, ctx.gpus)
    const sum = w.reduce((a, b) => a + b, 0) || 1
    let left = 100
    let big = 0
    w.forEach((x, i) => {
      layers[`gpu${i}`] = round1((100 * x) / sum)
      left -= layers[`gpu${i}`]
      if (x > w[big]) big = i
    })
    layers[`gpu${big}`] = round1(layers[`gpu${big}`] + left)
    tier = `gpu${big}`
  }
  st.layerPct = layers
  st.bulkPct = { ...zero(bt), ...layers }
  st.embeddings = tier
  st.output = tier
}

/** A subsystem's current placement, for planning locally. */
export interface PlanLabel {
  label: string
  bytes: number
  edit?: Role
  current: Record<string, number>
}

/**
 * Where each subsystem's bytes would go under these edits, estimated locally
 * (shares of bytes, not whole layers): used to redraw the machine at once and
 * in the demo, which has no daemon. The daemon's check gives the real plan.
 */
export function localPlan(labels: PlanLabel[], st: EditState): Record<string, Record<string, number>> {
  const out: Record<string, Record<string, number>> = {}
  const spread = (bytes: number, pct: Pct) => {
    const tiers = Object.keys(pct).filter((k) => pct[k] > 0)
    const res: Record<string, number> = {}
    let left = bytes
    tiers.forEach((k, i) => {
      const b = i === tiers.length - 1 ? left : Math.round((bytes * pct[k]) / 100)
      res[k] = b
      left -= b
    })
    return res
  }
  for (const l of labels) {
    if (st.mode !== 'custom' || !l.edit) out[l.label] = l.current
    else if (l.edit === 'layers') out[l.label] = spread(l.bytes, st.layerPct)
    else if (l.edit === 'bulk') out[l.label] = spread(l.bytes, st.bulkPct)
    else out[l.label] = st[l.edit] ? { [st[l.edit]!]: l.bytes } : l.current
  }
  return out
}
