<script lang="ts">
  // One RAM box as a bar on a linear scale, left to right:
  //   our models, exact: per model its RAM-tier weights, KV cache and compute scratch
  //   | free | other processes and kernel | disk cache (the kernel's)
  // Our memory is drawn with sharp edges from the workers' own measurements.
  // Everything else is fuzzy on purpose: each of its borders fades from the
  // smallest to the largest position it had in the daemon's last minute. The
  // part of the cache that is a model's disk tier is shaded in the model's
  // colour at the right end, as "may be dropped": it is the kernel's cache, not
  // ours, and never counted as model memory.
  import { bytes, type MemBox } from './api'
  import { size, type ModelView } from './view'

  let {
    mem,
    total,
    node,
    models,
    exact,
  }: {
    mem?: MemBox
    total: number
    node: number
    models: ModelView[]
    /** model -> node -> exact bytes */
    exact: Record<string, Record<number, { weights: number; kv: number; scratch: number }>>
  } = $props()

  const T = $derived(Math.max(total, 1))
  const pct = (b: number) => (100 * Math.min(Math.max(b, 0), T)) / T

  // our exact pieces, in model order
  const pieces = $derived(
    models.flatMap((m) => {
      const e = exact[m.name]?.[node]
      if (!e) return []
      return [
        { key: `${m.name}/w`, color: m.color, bytes: e.weights, what: 'RAM-tier weights', m, style: '' },
        { key: `${m.name}/kv`, color: m.color, bytes: e.kv, what: 'KV cache', m, style: 'kv' },
        { key: `${m.name}/s`, color: m.color, bytes: e.scratch, what: 'compute scratch', m, style: 'scratch' },
      ].filter((p) => p.bytes > 0)
    }),
  )
  const ours = $derived(pieces.reduce((s, p) => s + p.bytes, 0))

  // the fuzzy borders, shifted so they start where our exact memory ends
  const delta = $derived(mem ? ours - mem.edges[0].latest : 0)
  const gradient = $derived.by(() => {
    if (!mem) return 'none'
    const colours = ['var(--seg-free)', 'var(--seg-other)', 'var(--seg-slab)', 'var(--seg-cache)']
    const stops = [`${colours[0]} 0%`]
    let at = pct(ours)
    for (let i = 1; i < 4; i++) {
      const e = mem.edges[i]
      const lo = Math.max(at, pct(e.min + delta))
      const hi = Math.max(lo, pct(e.max + delta))
      stops.push(`${colours[i - 1]} ${lo}%`, `${colours[i]} ${hi}%`)
      at = hi
    }
    return `linear-gradient(to right, ${stops.join(', ')})`
  })

  // the disk tier's pages in RAM, right-aligned (the cache is the last segment)
  const tier = $derived((mem?.models ?? []).filter((m) => m.disk_tier_resident.latest > 0))
  const tierSum = $derived(tier.reduce((s, m) => s + m.disk_tier_resident.latest, 0))
  const tierMax = $derived(Math.min(total, Math.max(tierSum, tier.reduce((s, m) => s + m.disk_tier_resident.max, 0))))
  const tierMin = $derived(Math.min(tierMax, tier.reduce((s, m) => s + m.disk_tier_resident.min, 0)))
  const colorOf = (name: string) => models.find((m) => m.name === name)?.color ?? 'var(--disk)'
  const letterOf = (name: string) => models.find((m) => m.name === name)?.letter ?? '?'

  const spread = (s: { latest: number; min: number; max: number }) =>
    s.max - s.min > 16 << 20 ? `${bytes(s.latest)} (${bytes(s.min)} to ${bytes(s.max)})` : bytes(s.latest)

  const tooltip = $derived(
    [
      `Our models' memory in this box, exact (measured by the workers): ${size(ours)}`,
      ...(mem
        ? [
            `Free: ${spread(mem.segments.free)}, memory nobody uses`,
            `Other: ${spread(mem.segments.other)}, other programs, the daemon, the kernel, shared memory`,
            `Disk cache: ${spread(mem.segments.cache)}, file pages the kernel can drop at any time${tierSum > 0 ? `; ${size(tierSum)} of it is a model's disk tier (may be dropped)` : ''}`,
            `The blur at the borders after our memory is how much each moved in the last minute.`,
          ]
        : []),
    ].join('\n'),
  )
</script>

<div class="ram" title={tooltip}>
  <div class="track" style="background: {gradient}">
    <div class="ours">
      {#each pieces as p (p.key)}
        <span class={p.style} style="width: {pct(p.bytes)}%; --c: {p.color}" title="{p.m.letter}: {p.what} {size(p.bytes)}"></span>
      {/each}
    </div>
    {#if tierSum > 0}
      <div
        class="tier"
        style="width: {pct(tierMax)}%; mask-image: linear-gradient(to right, transparent 0%, black {tierMax ? (100 * (tierMax - tierMin)) / tierMax : 0}%); -webkit-mask-image: linear-gradient(to right, transparent 0%, black {tierMax ? (100 * (tierMax - tierMin)) / tierMax : 0}%)"
      >
        {#each tier as m (m.model)}
          <span style="width: {(100 * m.disk_tier_resident.latest) / Math.max(tierSum, 1)}%; --c: {colorOf(m.model)}" title="{letterOf(m.model)}: disk tier in cache {size(m.disk_tier_resident.latest)}, may be dropped"></span>
        {/each}
      </div>
    {/if}
  </div>
  {#if mem}
    <div class="legend muted">
      <span>free {size(mem.segments.free.latest)}</span>
      <span>other {size(mem.segments.other.latest)}</span>
      <span>disk cache {size(mem.segments.cache.latest)}{#if tierSum > 0}, {size(tierSum)} of it a disk tier: may be dropped{/if}</span>
    </div>
  {/if}
</div>

<style>
  .ram {
    --seg-free: #15171c;
    --seg-other: #7d8597;
    --seg-slab: #6b7e8f;
    --seg-cache: #3f5a73;
  }
  .track {
    position: relative;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background-color: var(--bg);
  }
  .ours {
    position: absolute;
    inset: 0;
    display: flex;
  }
  .ours span,
  .tier span {
    display: block;
    height: 100%;
    flex: none;
    background: var(--c);
  }
  .ours span.kv {
    background: color-mix(in srgb, var(--c) 55%, white);
  }
  .ours span.scratch {
    background: repeating-linear-gradient(45deg, var(--c) 0 3px, color-mix(in srgb, var(--c) 40%, black) 3px 6px);
  }
  .tier {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    display: flex;
  }
  .tier span {
    background: repeating-linear-gradient(135deg, var(--c) 0 2px, transparent 2px 5px);
    border-top: 1px solid var(--c);
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 10px;
    margin-top: 4px;
    font-size: 11px;
  }
</style>
