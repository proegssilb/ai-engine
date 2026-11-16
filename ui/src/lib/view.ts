// Turns what the daemon reports (machine, per-model placement) into what the
// placement view draws: one panel per socket holding that socket's CPU, RAM,
// GPUs and disks, and one chip per (model, subsystem) piece placed on a box.
// Pure functions, so the real UI and the demo page share them.

import { sockets, type Machine, type MachineModel, type ModelTag, type Placement } from './api'

export const GiB = 1 << 30
export const MiB = 1 << 20

/** Model colours, as in the mockup: A blue, B orange, C purple, D lime. */
export const PALETTE = ['#5b9cff', '#f2a43a', '#b88cff', '#c8dd4a', '#ff6b6b', '#4fd1c5', '#57c785', '#ff8fb3']
export const letter = (i: number) => String.fromCharCode(65 + (i % 26))

export type BoxKind = 'cpu' | 'ram' | 'gpu' | 'disk'

export interface HwBox {
  key: string
  kind: BoxKind
  title: string
  /** small text on the right of the title */
  detail?: string
  total: number
  /** the socket it hangs off; null = unknown (multi-socket machines only) */
  node: number | null
  cpus?: number
  cpulist?: string
  path?: string
  /** what the device reports in use, all processes */
  deviceUsed?: number
}

export interface Panel {
  key: string
  title: string
  tag: string
  /** null = the "Not attached to a socket" area */
  node: number | null
  boxes: HwBox[]
}

/** What a chip is: weights of a subsystem, KV cache, compute scratch, or "this model's threads run here". */
export type ChipKind = 'weights' | 'kv' | 'scratch' | 'runs'

export interface Chip {
  id: string
  model: string
  /** the subsystem id this piece belongs to (for the white selection outline) */
  sub: string
  label: string
  bytes: number
  /** the whole subsystem, when this is one piece of a split */
  whole?: number
  piece: boolean
  box: string
  kind: ChipKind
  /** extra tooltip text */
  title?: string
}

export interface SubDest {
  box: string
  bytes: number
}

export interface SubView {
  id: string
  label: string
  /** short label for chips */
  short: string
  desc: string
  bytes: number
  role: 'core' | 'bulk' | 'other' | 'kv'
  dests: SubDest[]
  split: boolean
  /** anatomy subsystems merged into this one */
  members: string[]
  order?: number
}

export interface ModelView {
  name: string
  letter: string
  color: string
  title: string
  desc: string
  size: number
  loaded: boolean
  moe: boolean
  subs: SubView[]
}

export interface View {
  panels: Panel[]
  models: ModelView[]
  chips: Record<string, Chip[]>
  boxes: Record<string, HwBox>
  summary: string
  /** "GPU 1", "RAM 0", "Disk 2" for a box key */
  name(box: string): string
  /** exact RAM-box numbers: model -> node -> {weights, kv, scratch} (from the worker's own measurements) */
  ram: Record<string, Record<number, { weights: number; kv: number; scratch: number }>>
}

/** How the subsystems the backend knows are grouped and described. */
export interface CatalogEntry {
  id: string
  label: string
  short?: string
  desc: string
  role: SubView['role']
  /** position in the model's list (default: by role, then size) */
  order?: number
}
/** The catalog entry for an anatomy subsystem label ('kv' asks for the KV cache entry) of a model. */
export type Catalog = (label: string, model: string) => CatalogEntry

export const KV: CatalogEntry = {
  id: 'kv',
  label: 'KV cache',
  desc: 'Working memory for the conversation. Grows with context length.',
  role: 'kv',
}

const CORE: CatalogEntry = {
  id: 'core',
  label: 'Core layers',
  desc: 'The always-on path: attention, norms, router and shared experts. Every token passes through it, so it wants the fastest memory.',
  role: 'core',
}

/** The backend's subsystems (aie_anatomy) grouped like the mockup's. */
export const anatomyCatalog: Catalog = (label) => {
  switch (label) {
    case 'kv':
      return KV
    case 'attention':
    case 'norms':
    case 'router':
    case 'shared experts':
      return CORE
    case 'routed experts':
      return {
        id: 'bulk',
        label: 'Expert bank',
        short: 'Experts',
        desc: 'Thousands of specialists; only a handful fire per token. Keep the busy ones on a GPU and the rest somewhere cheaper.',
        role: 'bulk',
      }
    case 'dense ffn':
      return {
        id: 'bulk',
        label: 'Feed-forward',
        desc: 'The dense feed-forward blocks, the biggest part of every layer. Split them by share between GPUs, RAM and disk.',
        role: 'bulk',
      }
    case 'embeddings':
      return { id: 'embeddings', label: 'Embeddings', desc: 'Turns tokens into vectors: a lookup, never computed.', role: 'other' }
    case 'output head':
      return { id: 'head', label: 'Output head', desc: 'Final norm and the projection to the vocabulary.', role: 'other' }
    default:
      return { id: label, label: label.charAt(0).toUpperCase() + label.slice(1), desc: '', role: 'other' }
  }
}

export function parseCpulist(s: string): number {
  let n = 0
  for (const part of s.trim().split(',').filter(Boolean)) {
    const [a, b] = part.split('-').map((x) => Number(x))
    n += b === undefined ? 1 : Math.max(0, b - a + 1)
  }
  return n
}

export function gib(b: number, digits = 1): string {
  return (b / GiB).toFixed(digits)
}

/** GiB/MiB, as the placement amounts are shown. */
export function size(b: number): string {
  if (b >= GiB * 0.095) return `${gib(b)} GiB`
  if (b >= MiB / 10) return `${(b / MiB).toFixed(b >= 10 * MiB ? 0 : 1)} MiB`
  return `${(b / 1024).toFixed(0)} KiB`
}

/** A size without its unit, in the unit `whole` is shown in. */
function num(b: number, unitOf: number): string {
  return unitOf >= GiB * 0.095 ? gib(b) : (b / MiB).toFixed(b >= 10 * MiB ? 0 : 1)
}

/** Decimal capacity, as drives and the mockup's header give it ("4 TB"). */
export function capacity(b: number): string {
  if (b >= 1e12) return `${+(b / 1e12).toFixed(1)} TB`
  if (b >= 1e9) return `${+(b / 1e9).toFixed(0)} GB`
  return `${+(b / 1e6).toFixed(0)} MB`
}

/** Memory boxes by the binary size the machine reports ("64 GiB"). */
export function binCapacity(b: number): string {
  return b >= GiB ? `${+(b / GiB).toFixed(b >= 10 * GiB ? 0 : 1)} GiB` : `${(b / MiB).toFixed(0)} MiB`
}

/** The amount shown on a chip: "5.0 GiB", or "20.0 of 50.0 GiB" for a piece of a split. */
export function amount(c: Pick<Chip, 'bytes' | 'whole' | 'piece'>): string {
  if (c.piece && c.whole) {
    const n = (b: number) => {
      const v = num(b, c.whole!)
      return v.endsWith('.0') ? v.slice(0, -2) : v
    }
    return `${n(c.bytes)} of ${size(c.whole).replace('.0 ', ' ')}`
  }
  return size(c.bytes)
}

/** Box height: one step per doubling of capacity (the "Box height scale" legend). */
export const BASE_GIB = 16
export const STEP_PX = 36
export function boxHeight(total: number): number {
  return 96 + STEP_PX * Math.max(0, Math.log2(Math.max(total, 1) / (BASE_GIB * GiB)))
}
export function scaleBarHeight(total: number): number {
  return 10 + 6 * Math.max(0, Math.log2(total / (BASE_GIB * GiB)))
}

export interface ViewInput {
  machine: Machine
  tags: ModelTag[]
  placements: Record<string, Placement | undefined>
  catalog?: Catalog
}

/** Temperature words for the pieces of a split expert bank: busy on the GPU, rest in RAM, cold on disk. */
function pieceWord(role: SubView['role'], moe: boolean, tier: string, n: number, bytes: number, whole: number): string {
  if (role === 'bulk' && moe) return tier.startsWith('gpu') ? 'hot' : tier === 'ram' ? 'warm' : 'cold'
  return n === 2 && Math.abs(bytes / Math.max(whole, 1) - 0.5) < 0.1 ? 'half' : 'part'
}

export function buildView(input: ViewInput): View {
  const { machine, tags, placements } = input
  const catalog = input.catalog ?? anatomyCatalog

  // ---- hardware -------------------------------------------------------
  let socks = sockets(machine.numa)
  if (!socks.length) {
    const n0 = machine.numa[0]
    socks = [n0 ?? { node: 0, total: machine.ram.total, free: machine.ram.available, cpus: '' }]
  }
  const multi = socks.length > 1
  const nodeOf = (n: number | null | undefined): number | null => {
    if (n !== null && n !== undefined && n >= 0 && socks.some((s) => s.node === n)) return n
    return multi ? null : socks[0].node
  }

  const boxes: Record<string, HwBox> = {}
  const panels: Panel[] = socks.map((s) => ({
    key: `socket${s.node}`,
    title: `Socket ${s.node}`,
    tag: `NUMA node ${s.node}`,
    node: s.node,
    boxes: [],
  }))
  const unattached: Panel = { key: 'unattached', title: 'Not attached to a socket', tag: 'NUMA node unknown', node: null, boxes: [] }
  const panelFor = (node: number | null) => (node === null ? unattached : panels.find((p) => p.node === node)!)
  const add = (b: HwBox) => {
    boxes[b.key] = b
    panelFor(b.node).boxes.push(b)
  }

  for (const s of socks) {
    const ncpu = parseCpulist(s.cpus)
    add({ key: `cpu${s.node}`, kind: 'cpu', title: `CPU ${s.node}`, detail: ncpu ? `${ncpu} CPUs` : '', total: 0, node: s.node, cpus: ncpu, cpulist: s.cpus })
    add({ key: `ram${s.node}`, kind: 'ram', title: multi ? `RAM ${s.node}` : 'RAM', total: multi ? s.total : machine.ram.total || s.total, node: s.node })
  }
  let g = 0
  machine.devices.forEach((d) => {
    if (!d.gpu) return
    add({ key: `gpu${g}`, kind: 'gpu', title: `GPU ${g}`, detail: d.description, total: d.total, node: nodeOf(d.numa_node), deviceUsed: d.total - d.free })
    g++
  })
  const disks = machine.disks?.length ? machine.disks : machine.disk ? [machine.disk] : []
  disks.forEach((d, i) =>
    add({ key: `disk${i}`, kind: 'disk', title: d.name ?? (disks.length > 1 ? `Disk ${i}` : 'Disk'), detail: '', path: d.path, total: d.total, node: nodeOf(d.numa_node), deviceUsed: d.total - d.free }),
  )
  // cpu, ram, then gpus, then disks, in that order within a panel
  const order: Record<BoxKind, number> = { cpu: 0, ram: 1, gpu: 2, disk: 3 }
  for (const p of [...panels, unattached]) p.boxes.sort((a, b) => order[a.kind] - order[b.kind])
  const allPanels = unattached.boxes.length ? [...panels, unattached] : panels

  const gpuTotal = Object.values(boxes).filter((b) => b.kind === 'gpu')
  const diskGroups = new Map<number, number>()
  disks.forEach((d) => diskGroups.set(d.total, (diskGroups.get(d.total) ?? 0) + 1))
  const noun = disks.length && disks.every((d) => d.name?.startsWith('NVMe')) ? 'NVMe' : disks.length === 1 ? 'disk' : 'disks'
  const diskText = disks.length
    ? [...diskGroups].map(([t, n]) => `${n} × ${capacity(t)}`).join(' + ') + ` ${noun}`
    : ''
  const ramTotal = Object.values(boxes)
    .filter((b) => b.kind === 'ram')
    .reduce((s, b) => s + b.total, 0)
  const summary = [
    `${socks.length} socket${socks.length === 1 ? '' : 's'}`,
    gpuTotal.length ? `${gpuTotal.length} GPU${gpuTotal.length === 1 ? '' : 's'} (${binCapacity(gpuTotal.reduce((s, b) => s + b.total, 0))})` : 'no GPUs',
    `${binCapacity(ramTotal)} RAM`,
    diskText,
  ]
    .filter(Boolean)
    .join(' · ')

  const name = (box: string) => {
    const b = boxes[box]
    return b ? b.title : box
  }

  // ---- models and their pieces ---------------------------------------
  const chips: Record<string, Chip[]> = {}
  const ramExact: View['ram'] = {}
  const put = (c: Chip) => (chips[c.box] ??= []).push(c)
  const models: ModelView[] = []

  tags.forEach((tag, mi) => {
    const p = placements[tag.name]
    const mm: MachineModel | undefined = machine.models.find((m) => m.model === tag.name)
    const planned = mm?.planned ?? p?.loaded?.planned ?? p?.intended
    const moe = !!p?.anatomy.moe
    const host = mm?.host ?? p?.loaded?.host ?? null
    const hostNode = host && socks.some((s) => s.node === host.node) ? host.node : socks[0].node

    // which box a tier key means for this model
    const diskIndex = Math.max(0, disks.findIndex((d) => d.path === (mm?.store ?? disks[0]?.path)))
    const boxOf = (t: string): string => (t === 'ram' ? `ram${hostNode}` : t === 'disk' ? `disk${diskIndex}` : t)

    // subsystems, merged as the catalog says
    const merged = new Map<string, SubView>()
    const sizes = p?.anatomy.subsystems ?? {}
    const labels = new Set([...Object.keys(sizes), ...Object.keys(planned ?? {})])
    for (const label of labels) {
      const e = catalog(label, tag.name)
      let sv = merged.get(e.id)
      if (!sv) {
        sv = { id: e.id, label: e.label, short: e.short ?? e.label, desc: e.desc, bytes: 0, role: e.role, dests: [], split: false, members: [], order: e.order }
        merged.set(e.id, sv)
      }
      sv.bytes += sizes[label] ?? Object.values(planned?.[label] ?? {}).reduce((a, b) => a + b, 0)
      sv.members.push(label)
      for (const [t, b] of Object.entries(planned?.[label] ?? {})) {
        if (b <= 0) continue
        const box = boxOf(t)
        const d = sv.dests.find((x) => x.box === box)
        if (d) d.bytes += b
        else sv.dests.push({ box, bytes: b })
      }
    }

    // The RAM tier's weights as the worker measured them: per node, they
    // replace the planned bytes of the RAM-tier subsystems (same total,
    // shared out in proportion), so chips, bar and table are one number.
    const measuredTier = Object.entries(host?.ram_tier_by_node ?? {})
      .map(([n, b]) => [Number(n), b] as const)
      .filter(([, b]) => b > 0)
    const tierSubs = [...merged.values()].filter((s) => s.dests.some((d) => d.box === `ram${hostNode}`))
    const plannedRam = tierSubs.reduce((s, sv) => s + (sv.dests.find((d) => d.box === `ram${hostNode}`)?.bytes ?? 0), 0)
    const measuredTotal = measuredTier.reduce((s, [, b]) => s + b, 0)
    if (measuredTotal > 0 && plannedRam > 0) {
      for (const sv of tierSubs) {
        const d = sv.dests.find((x) => x.box === `ram${hostNode}`)!
        const share = d.bytes / plannedRam
        sv.dests = sv.dests.filter((x) => x !== d)
        let left = Math.round(measuredTotal * share)
        measuredTier.forEach(([n, b], i) => {
          const v = i === measuredTier.length - 1 ? left : Math.round(b * share)
          left -= v
          if (v > 0) sv.dests.push({ box: `ram${n}`, bytes: v })
        })
      }
    }

    // KV cache and compute scratch, from the live breakdown (devices) and the host report (RAM)
    const kvDests: SubDest[] = []
    const scratch: SubDest[] = []
    for (const r of mm?.memory ?? p?.loaded?.measured ?? []) {
      if (r.kind === 'device') {
        const m = /(\d+)$/.exec(r.name)
        const box = m ? `gpu${m[1]}` : ''
        if (!boxes[box]) continue
        if (r.context > 0) kvDests.push({ box, bytes: r.context })
        if (r.compute > 0) scratch.push({ box, bytes: r.compute })
      }
    }
    // a stray page or two on another socket isn't a piece worth drawing
    for (const [n, b] of Object.entries(host?.kv_cache_by_node ?? {})) if (b >= MiB) kvDests.push({ box: `ram${n}`, bytes: b })
    for (const [n, b] of Object.entries(host?.compute_by_node ?? {})) if (b >= MiB) scratch.push({ box: `ram${n}`, bytes: b })
    if (kvDests.length) {
      const k = catalog('kv', tag.name)
      merged.set('kv', { id: k.id, label: k.label, short: k.short ?? k.label, desc: k.desc, role: 'kv', order: k.order, bytes: kvDests.reduce((s, d) => s + d.bytes, 0), dests: kvDests, split: false, members: [] })
    }

    const subs = [...merged.values()].sort((a, b) => {
      const o = { core: 0, bulk: 1, other: 2, kv: 3 }
      if (a.order !== undefined || b.order !== undefined) return (a.order ?? 99) - (b.order ?? 99)
      return o[a.role] - o[b.role] || b.bytes - a.bytes
    })
    for (const sv of subs) sv.split = sv.dests.length > 1

    // chips: one per (subsystem, box) piece, for models that are loaded
    const loaded = !!mm || !!p?.loaded
    if (loaded) {
      for (const sv of subs) {
        sv.dests.forEach((d) => {
          const word = sv.split ? pieceWord(sv.role, moe, d.box.startsWith('gpu') ? d.box : d.box.startsWith('ram') ? 'ram' : 'disk', sv.dests.length, d.bytes, sv.bytes) : ''
          const label = sv.split ? (sv.role === 'bulk' && moe ? `${sv.short} · ${word}` : `${sv.short}, ${word}`) : sv.short
          put({
            id: `${tag.name}/${sv.id}/${d.box}`,
            model: tag.name,
            sub: sv.id,
            label,
            bytes: d.bytes,
            whole: sv.split ? sv.bytes : undefined,
            piece: sv.split,
            box: d.box,
            kind: sv.role === 'kv' ? 'kv' : 'weights',
            title: sv.members.length > 1 ? `${sv.label}: ${sv.members.join(', ')}` : undefined,
          })
        })
      }
      for (const d of scratch) {
        put({ id: `${tag.name}/scratch/${d.box}`, model: tag.name, sub: 'scratch', label: 'Compute scratch', bytes: d.bytes, piece: false, box: d.box, kind: 'scratch', title: 'Temporary working memory for computing a batch' })
      }
      // "runs" chips: the model's threads run on its socket
      if (host && socks.some((s) => s.node === host.node)) {
        const here = subs.filter((s) => s.role !== 'kv' && s.dests.some((d) => d.box === `ram${host.node}`))
        const parts = subs.filter((s) => s.role !== 'kv')
        const short = tag.name.replace(/^.*\//, '').split(/[-:]/)[0]
        const label = here.length
          ? here.length === parts.length && here.length > 1
            ? `${short}, ${here.length === 2 ? 'both' : 'all'} parts`
            : here.map((s) => (s.split ? chipLabel(s, moe, `ram${host.node}`) : s.short)).join(' + ')
          : 'Host side'
        put({ id: `${tag.name}/runs`, model: tag.name, sub: here.length === 1 ? here[0].id : '', label, bytes: 0, piece: false, box: `cpu${host.node}`, kind: 'runs' })
      }
      // exact RAM-box numbers per node, for the bar
      if (host) {
        const nodes = new Set([...Object.keys(host.ram_tier_by_node ?? {}), ...Object.keys(host.kv_cache_by_node ?? {}), ...Object.keys(host.compute_by_node ?? {})])
        for (const n of nodes) {
          ;(ramExact[tag.name] ??= {})[Number(n)] = {
            weights: host.ram_tier_by_node?.[n] ?? 0,
            kv: (host.kv_cache_by_node?.[n] ?? 0) >= MiB ? host.kv_cache_by_node![n] : 0,
            scratch: (host.compute_by_node?.[n] ?? 0) >= MiB ? host.compute_by_node![n] : 0,
          }
        }
      }
    }

    const desc = tag.details.summary ?? [
      tag.details.family,
      moe ? 'sparse experts' : p ? 'dense' : '',
      tag.details.quantization_level,
    ]
      .filter(Boolean)
      .join(' · ')
    models.push({
      name: tag.name,
      letter: letter(mi),
      color: PALETTE[mi % PALETTE.length],
      title: tag.name.replace(/^hf\.co\//, ''),
      desc,
      size: p?.anatomy.bytes ?? tag.size,
      loaded,
      moe,
      subs,
    })
  })

  return { panels: allPanels, models, chips, boxes, summary, name, ram: ramExact }
}

function chipLabel(s: SubView, moe: boolean, box: string): string {
  const d = s.dests.find((x) => x.box === box)
  const word = pieceWord(s.role, moe, 'ram', s.dests.length, d?.bytes ?? 0, s.bytes)
  return s.role === 'bulk' && moe ? `${s.short} · ${word}` : `${s.short}, ${word}`
}
