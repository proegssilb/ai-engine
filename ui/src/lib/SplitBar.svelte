<script lang="ts">
  // A subsystem split across tiers by share (decision D6), drawn like the
  // mockup's expert bank: one segment per tier, in shades of the model's
  // colour, with the tier under each segment. Edit by dragging a divider, or
  // with the share or size fields: the last tier with a share takes the rest.
  import { type Pct } from './edit.svelte'
  import { gib, size } from './view'

  let {
    tiers,
    total,
    pct = $bindable(),
    color,
    name,
    word = () => '',
    onedit,
    openEdit = false,
  }: {
    tiers: string[]
    /** bytes of the whole subsystem */
    total: number
    pct: Pct
    color: string
    /** display name of a tier key: "GPU 1", "RAM 0", "Disk 0" */
    name: (tier: string) => string
    /** hot / warm / cold for an expert bank */
    word?: (tier: string) => string
    onedit: () => void
    openEdit?: boolean
  } = $props()

  const shown = $derived(tiers.filter((k) => (pct[k] ?? 0) > 0))
  // tiers with a field: those with a share, and ones added by hand
  let added = $state<string[]>([])
  const fields = $derived(tiers.filter((k) => (pct[k] ?? 0) > 0 || added.includes(k)))
  const free = $derived(tiers.filter((k) => !fields.includes(k)))
  const fmt = (b: number) => { const v = b / (1 << 30); return Math.abs(v - Math.round(v)) < 0.05 ? String(Math.round(v)) : v.toFixed(v < 10 ? 1 : 0) }
  const sum = $derived(tiers.reduce((s, k) => s + (pct[k] ?? 0), 0))
  const shade = (i: number) => `color-mix(in srgb, ${color} ${Math.max(35, 100 - 28 * i)}%, white)`
  const round1 = (v: number) => Math.round(v * 10) / 10

  let bar: HTMLDivElement | undefined = $state()

  function drag(e: PointerEvent, i: number) {
    // divider i sits between shown[i] and shown[i+1]
    const el = bar
    if (!el) return
    const left = shown[i]
    const right = shown[i + 1]
    const start = shown.slice(0, i).reduce((s, k) => s + pct[k], 0)
    const end = start + pct[left] + pct[right]
    const target = e.currentTarget as HTMLElement
    target.setPointerCapture(e.pointerId)
    const move = (ev: PointerEvent) => {
      const r = el.getBoundingClientRect()
      const at = Math.min(end, Math.max(start, (100 * (ev.clientX - r.left)) / r.width))
      const l = Math.round(at - start)
      pct[left] = l
      pct[right] = round1(end - start - l)
      onedit()
    }
    const up = () => {
      target.removeEventListener('pointermove', move)
      target.removeEventListener('pointerup', up)
    }
    target.addEventListener('pointermove', move)
    target.addEventListener('pointerup', up)
  }

  function setGiB(k: string, v: number) {
    pct[k] = round1(Math.min(100, Math.max(0, (100 * v * (1 << 30)) / Math.max(total, 1))))
    onedit()
  }
</script>

<div class="split">
  <div class="bar" bind:this={bar}>
    {#each shown as k, i (k)}
      <div class="seg" style="flex: {pct[k]} 1 0; background: {shade(i)}" title="{name(k)}: {pct[k]}% = {size((pct[k] / 100) * total)}">
        <span>{fmt((pct[k] / 100) * total)}</span>
        {#if i < shown.length - 1}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div class="handle" onpointerdown={(e) => drag(e, i)} title="Drag to rebalance"></div>
        {/if}
      </div>
    {/each}
  </div>
  <div class="labels">
    {#each shown as k (k)}
      <span>{#if word(k)}{word(k)}{' '}{/if}→{' '}{name(k)}</span>
    {/each}
  </div>
  <details class="edit" open={openEdit}>
    <summary>Edit by size or share</summary>
  <div class="fields">
    {#each fields as k (k)}
      <label>
        <span class="n">{name(k)}</span>
        <input type="number" min="0" max="100" step="1" value={pct[k] ?? 0} oninput={(e) => { pct[k] = Math.min(100, Math.max(0, +e.currentTarget.value || 0)); onedit() }} aria-label="{name(k)} share in percent" />%
        <input type="number" min="0" step="0.5" value={+gib(((pct[k] ?? 0) / 100) * total)} oninput={(e) => setGiB(k, +e.currentTarget.value || 0)} aria-label="{name(k)} size in GiB" />GiB
      </label>
    {/each}
  </div>
  {#if free.length}
    <select class="add" aria-label="Add a tier" onchange={(e) => { if (e.currentTarget.value) added = [...added, e.currentTarget.value]; e.currentTarget.value = '' }}>
      <option value="">+ add a tier</option>
      {#each free as k (k)}<option value={k}>{name(k)}</option>{/each}
    </select>
  {/if}
  </details>
  {#if Math.abs(sum - 100) >= 0.5}
    <div class="error small">Shares add up to {round1(sum)}%; they need to make 100%.</div>
  {/if}
</div>

<style>
  .split {
    margin-top: 8px;
  }
  .bar {
    display: flex;
    height: 28px;
    gap: 2px;
    background: transparent;
    border-radius: 5px;
    overflow: visible;
  }
  .seg {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 22px;
    color: #10131a;
    font-weight: 600;
    font-size: 12px;
    font-family: ui-monospace, 'JetBrains Mono', Menlo, Consolas, 'Liberation Mono', monospace;
  }
  .seg:first-child {
    border-radius: 5px 0 0 5px;
  }
  .seg:last-child {
    border-radius: 0 5px 5px 0;
  }
  .seg:only-child {
    border-radius: 5px;
  }
  .handle {
    position: absolute;
    right: -7px;
    top: -2px;
    bottom: -2px;
    width: 12px;
    cursor: ew-resize;
    z-index: 2;
    touch-action: none;
  }
  .handle::after {
    content: '';
    position: absolute;
    left: 5px;
    top: 4px;
    bottom: 4px;
    width: 2px;
    background: #10131a;
    opacity: 0.55;
    border-radius: 1px;
  }
  .labels {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 18px;
    margin-top: 6px;
    color: var(--muted);
    font-family: ui-monospace, 'JetBrains Mono', Menlo, Consolas, 'Liberation Mono', monospace;
    font-size: 12px;
  }
  .fields {
    display: grid;
    grid-template-columns: 1fr;
    gap: 6px 10px;
    margin-top: 8px;
  }
  label {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--muted);
  }
  .n {
    width: 50px;
    color: var(--text);
  }
  input {
    width: 58px;
    padding: 3px 5px;
    font-size: 12px;
  }
  .add {
    margin-top: 6px;
    font-size: 12px;
    padding: 3px 5px;
  }
  .edit {
    margin-top: 6px;
    font-size: 12.5px;
  }
  summary {
    cursor: pointer;
    color: var(--muted);
  }
  .small {
    font-size: 12px;
    margin-top: 6px;
  }
</style>
