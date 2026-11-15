<script lang="ts">
  // One RAM box as a bar, left to right:
  //   our models | free | other processes + kernel | (reclaimable slab) | disk cache
  // The borders are deliberately blurry. Each border fades from the smallest
  // to the largest position it had in the daemon's last minute, so a value
  // that jumps about is a wide blur and a steady one a sharp edge. The part of
  // the cache that is our models' disk-tier pages is shaded in the model's
  // colour at the right end, so it isn't counted twice.
  import { bytes, type MemBox } from './api'

  let { box, colors }: { box: MemBox; colors: Record<string, string> } = $props()

  const total = $derived(Math.max(box.total, 1))
  const pct = (b: number) => (100 * Math.min(Math.max(b, 0), total)) / total

  // Colour stops: solid segment colours with a fade across each border's
  // min..max. Positions never go backwards, so overlapping fades just merge.
  const gradient = $derived.by(() => {
    const colours = ['var(--seg-ours)', 'var(--seg-free)', 'var(--seg-other)', 'var(--seg-slab)', 'var(--seg-cache)']
    const stops: string[] = []
    let at = 0
    stops.push(`${colours[0]} 0%`)
    box.edges.forEach((e, i) => {
      const lo = Math.max(at, pct(e.min))
      const hi = Math.max(lo, pct(e.max))
      stops.push(`${colours[i]} ${lo}%`, `${colours[i + 1]} ${hi}%`)
      at = hi
    })
    return `linear-gradient(to right, ${stops.join(', ')})`
  })

  const fade = (lo: number, hi: number) => `linear-gradient(to right, black ${pct(lo)}%, transparent ${Math.max(pct(hi), pct(lo) + 0.01)}%)`

  // our models, side by side from the left, fading out across the first border
  const ours = $derived(box.models.filter((m) => m.ours.latest > 0))
  const oursSum = $derived(ours.reduce((s, m) => s + m.ours.latest, 0))

  // the disk-tier pages in RAM, right-aligned; the group's left edge moves
  // between total - max and total - min of the summed residency
  const tier = $derived(box.models.filter((m) => m.disk_tier_resident.latest > 0))
  const tierSum = $derived(tier.reduce((s, m) => s + m.disk_tier_resident.latest, 0))
  const tierMax = $derived(Math.min(box.total, Math.max(tierSum, box.models.reduce((s, m) => s + m.disk_tier_resident.max, 0))))
  const tierMin = $derived(Math.min(tierMax, box.models.reduce((s, m) => s + m.disk_tier_resident.min, 0)))

  const name = (m: string) => (m === '' ? 'ai-engine itself' : m.replace(/^hf\.co\//, ''))
  const mib = (s: { latest: number; min: number; max: number }) =>
    s.max - s.min > 16 << 20 ? `${bytes(s.latest)} (${bytes(s.min)} to ${bytes(s.max)})` : bytes(s.latest)

  const tooltip = $derived(
    [
      `Memory over the last minute (the blur is how much each border moved):`,
      `Our models: ${mib(box.segments.ours)}, the memory ai-engine and its workers hold (not counting mapped model files, which are in the cache)`,
      `Free: ${mib(box.segments.free)}, memory nobody uses`,
      `Other: ${mib(box.segments.other)}, other programs, the kernel, shared memory`,
      `Reclaimable kernel caches: ${mib(box.segments.slab)}, between "other" and the cache`,
      `Disk cache: ${mib(box.segments.cache)}, file pages the system can drop at any time${tierSum > 0 ? `; ${bytes(tierSum)} of it is our models' disk tier, shaded in the model's colour` : ''}`,
    ].join('\n'),
  )
</script>

<div class="ram" title={tooltip}>
  <div class="track" style="background: {gradient}">
    <div class="layer" style="mask-image: {fade(box.edges[0].min, box.edges[0].max)}; -webkit-mask-image: {fade(box.edges[0].min, box.edges[0].max)}">
      {#each ours as m (m.model)}
        <span style="width: {pct((oursSum ? m.ours.latest / oursSum : 0) * box.edges[0].latest)}%; background: {m.model === '' ? 'var(--accent)' : (colors[m.model] ?? 'var(--accent)')}"></span>
      {/each}
    </div>
    {#if tierSum > 0}
      <div
        class="tier"
        style="width: {pct(tierMax)}%; mask-image: linear-gradient(to right, transparent 0%, black {tierMax ? (100 * (tierMax - tierMin)) / tierMax : 0}%); -webkit-mask-image: linear-gradient(to right, transparent 0%, black {tierMax ? (100 * (tierMax - tierMin)) / tierMax : 0}%)"
      >
        {#each tier as m (m.model)}
          <span style="width: {(100 * m.disk_tier_resident.latest) / Math.max(tierSum, 1)}%; background: {colors[m.model] ?? 'var(--disk)'}"></span>
        {/each}
      </div>
    {/if}
  </div>
  <div class="legend muted">
    <span><i style="background: var(--accent)"></i>ours {bytes(box.segments.ours.latest)}</span>
    <span><i class="free"></i>free {bytes(box.segments.free.latest)}</span>
    <span><i style="background: var(--seg-other)"></i>other {bytes(box.segments.other.latest)}</span>
    <span><i style="background: var(--seg-cache)"></i>disk cache {bytes(box.segments.cache.latest)}</span>
  </div>
</div>

<style>
  .ram {
    --seg-ours: var(--accent);
    --seg-free: var(--panel);
    --seg-other: #7d8597;
    --seg-slab: #6b7e8f;
    --seg-cache: #3f5a73;
  }
  .track {
    position: relative;
    height: 18px;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--line);
  }
  .layer {
    position: absolute;
    inset: 0;
    display: flex;
  }
  .layer span,
  .tier span {
    display: block;
    height: 100%;
    flex: none;
  }
  .tier {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    display: flex;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 10px;
    margin-top: 6px;
    font-size: 11.5px;
  }
  .legend i {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 2px;
    margin-right: 4px;
    border: 1px solid var(--line);
  }
  .legend i.free {
    background: var(--seg-free);
  }
</style>
