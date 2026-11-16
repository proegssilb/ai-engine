<script lang="ts">
  // One model as in the mockup: letter badge, name, a line about it, its total,
  // and its subsystems each with where it goes. A split subsystem (the expert
  // bank) opens into a bar with a segment per tier: this is where placement is
  // edited, by size or share per tier (D6).
  import SplitBar from './SplitBar.svelte'
  import { tierOfBox, type ModelEdit } from './edit.svelte'
  import { short, size, type ModelView, type SubView, type View } from './view'

  let {
    mv,
    view,
    gpus,
    selected,
    edit,
    running,
    openEdit = false,
    sockets,
    busy,
    onselect,
    onunload,
    ondiscard,
  }: {
    mv: ModelView
    view: View
    gpus: number
    selected: { model: string; sub: string | null } | null
    edit: ModelEdit
    running: boolean
    openEdit?: boolean
    /** NUMA nodes that can host a model; the socket choice only shows with several */
    sockets: number[]
    busy: string
    onselect: (model: string, sub: string | null) => void
    onunload: () => void
    ondiscard: () => void
  } = $props()

  const isSel = $derived(selected?.model === mv.name)

  // "GPU 2 + 3", "RAM 0", "NVMe 0 + RAM 0"
  function where(s: SubView): string {
    if (!s.dests.length) return mv.loaded ? '' : 'not loaded'
    const names = s.dests.map((d) => view.name(d.box))
    const gpu = names.filter((n) => n.startsWith('GPU '))
    const rest = names.filter((n) => !n.startsWith('GPU '))
    const parts = [...(gpu.length ? [`GPU ${gpu.map((n) => n.slice(4)).join(' + ')}`] : []), ...rest]
    return parts.join(' + ')
  }

  const gpuKeys = $derived(Array.from({ length: gpus }, (_, i) => `gpu${i}`))
  const tiersFor = (s: SubView) => (s.role === 'core' ? [...gpuKeys, 'ram'] : [...gpuKeys, 'ram', 'disk'])
  // the display name of a tier key on this machine: RAM on the socket the model's host side is on
  const ramBox = $derived(Object.keys(view.boxes).find((k) => view.boxes[k].kind === 'ram' && (view.chips[k] ?? []).some((c) => c.model === mv.name)) ?? Object.keys(view.boxes).find((k) => view.boxes[k].kind === 'ram') ?? 'ram0')
  const diskBox = $derived(Object.keys(view.boxes).find((k) => view.boxes[k].kind === 'disk') ?? 'disk0')
  const tierName = (t: string) => (t === 'ram' ? view.name(ramBox) : t === 'disk' ? view.name(diskBox) : view.name(t))
  const word = (s: SubView) => (t: string) => (s.role === 'bulk' && mv.moe ? (t.startsWith('gpu') ? 'hot' : t === 'ram' ? 'warm' : 'cold') : '')
  const editable = (s: SubView) => (s.role === 'core' || s.role === 'bulk')

  function touched() {
    edit.mode = 'custom'
    edit.dirty = true
  }
</script>

<div class="card" class:sel={isSel} style="--c: {mv.color}">
  <button class="head" onclick={() => onselect(mv.name, null)}>
    <span class="badge">{mv.letter}</span>
    <span class="who">
      <span class="name">{mv.title}</span>
      <span class="sub-line">{mv.desc}{#if running}<span class="ok" title="loaded"> ●</span>{/if}</span>
    </span>
    <span class="total mono">{size(mv.size)}</span>
  </button>
  <div class="status">
    <span class="mark {mv.status.kind}" title={mv.status.kind === 'verified' ? 'Every weight verified where planned' : mv.status.kind === 'misplaced' ? 'Some weights are not where planned; the chips say which' : 'Not loaded'}>{mv.status.kind === 'verified' ? '✓ ' : mv.status.kind === 'misplaced' ? '! ' : ''}{mv.status.text}</span>
    {#if mv.statsLine}<span class="muted mono">{mv.statsLine}</span>{/if}
  </div>
  {#each mv.warnings as w (w)}<div class="warn bad">{w}</div>{/each}
  {#if mv.fixes.length}
    <div class="warn amber">
      The RAM tier is not held by the OS and may be swapped out. To fix:
      <ul>{#each mv.fixes as f (f)}<li>{f}</li>{/each}</ul>
    </div>
  {/if}
  {#if isSel}
    <details class="controls" open={openEdit || edit.dirty}>
      <summary>Placement: {edit.mode === 'auto' ? 'automatic' : 'custom split'} · context {edit.nCtx}</summary>
      <div class="modes">
        <label><input type="radio" bind:group={edit.mode} value="auto" onchange={() => (edit.dirty = true)} /> Automatic</label>
        <label><input type="radio" bind:group={edit.mode} value="custom" onchange={() => (edit.dirty = true)} /> Custom split</label>
      </div>
      <div class="row">
        <label>Context <input type="number" min="512" step="512" bind:value={edit.nCtx} oninput={touched} /> tokens</label>
        {#if mv.moe}
          <label>Expert cache <input type="number" min="0" step="0.5" bind:value={edit.cacheGiB} oninput={touched} /> GiB</label>
        {/if}
      </div>
      {#if sockets.length > 1}
        <label class="row">Host socket
          <select bind:value={edit.hostNode} onchange={touched}>
            <option value={null}>Automatic: next to its GPU</option>
            {#each sockets as n (n)}<option value={n}>Socket {n}</option>{/each}
          </select>
        </label>
      {/if}
      <div class="row">
        {#if mv.loaded}<button onclick={onunload} disabled={!!busy}>Unload</button>{/if}
        {#if edit.dirty}<button onclick={ondiscard} disabled={!!busy}>Discard edits</button>{/if}
      </div>
    </details>
  {/if}
  {#each mv.subs as s (s.id)}
    {@const sel = selected?.model === mv.name && selected.sub === s.id}
    <div class="sub" class:sel>
      <button class="row" onclick={() => onselect(mv.name, s.id)}>
        <span class="nm">{s.label}</span>
        <span class="amt mono">{#if sel && s.split}{size(s.bytes) + " · split in " + s.dests.length}{:else}{short(s.bytes)}{#if where(s)}{" → " + where(s)}{/if}{/if}</span>
      </button>
      {#if isSel && s.desc}<p class="desc">{s.desc}</p>{/if}
      {#if sel && editable(s)}
        {#if s.role === 'core'}
          <SplitBar tiers={tiersFor(s)} total={s.bytes} bind:pct={edit.layerPct} color={mv.color} name={tierName} word={word(s)} onedit={touched} {openEdit} />
        {:else}
          <SplitBar tiers={tiersFor(s)} total={s.bytes} bind:pct={edit.bulkPct} color={mv.color} name={tierName} word={word(s)} onedit={touched} {openEdit} />
        {/if}
        <p class="hint">Drag a divider to rebalance.</p>
      {/if}
    </div>
  {/each}
  {#if !mv.subs.length}<div class="muted small">No anatomy yet.</div>{/if}
</div>

<style>
  .card {
    border: 1.5px solid color-mix(in srgb, var(--c) 70%, transparent);
    border-radius: 12px;
    padding: 8px;
    background: color-mix(in srgb, var(--c) 6%, var(--panel));
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .card.sel {
    border-color: var(--c);
    box-shadow: 0 0 0 1px var(--c);
  }
  .head {
    display: flex;
    gap: 10px;
    align-items: center;
    text-align: left;
    background: none;
    border: none;
    padding: 2px 2px 4px;
    width: 100%;
  }
  .badge {
    width: 26px;
    height: 26px;
    border-radius: 7px;
    background: var(--c);
    color: #10131a;
    font-weight: 700;
    display: grid;
    place-items: center;
    flex: none;
  }
  .who {
    display: flex;
    flex-direction: column;
    min-width: 0;
    flex: 1;
  }
  .name {
    font-weight: 600;
    font-size: 15px;
    overflow-wrap: anywhere;
  }
  .sub-line {
    color: var(--muted);
    font-size: 12.5px;
  }
  .total {
    color: var(--muted);
    align-self: flex-start;
  }
  .sub {
    background: var(--panel-2);
    border-radius: 7px;
    padding: 4px 8px;
    border: 1.5px solid transparent;
  }
  .sub.sel {
    border-color: #fff;
    background: color-mix(in srgb, var(--c) 14%, var(--panel-2));
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    background: none;
    border: none;
    padding: 0;
    width: 100%;
    text-align: left;
    font-size: 13.5px;
  }
  .row:hover:not(:disabled) {
    border: none;
  }
  .nm {
    font-weight: 500;
    min-width: 0;
  }
  .amt {
    color: var(--muted);
    text-align: right;
    white-space: nowrap;
    flex: none;
  }
  .desc {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 12.5px;
    line-height: 1.25;
  }
  .hint {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 12.5px;
  }
  .small {
    font-size: 12px;
  }
  .status {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 10px;
    padding: 0 4px;
    font-size: 12px;
  }
  .mark.verified {
    color: var(--ok);
  }
  .mark.misplaced {
    color: var(--bad);
    font-weight: 600;
  }
  .mark.not {
    color: var(--muted);
  }
  .mark {
    white-space: nowrap;
  }
  .warn {
    font-size: 12.5px;
    border: 1px solid;
    border-radius: 7px;
    padding: 5px 8px;
    line-height: 1.3;
  }
  .warn.bad {
    color: var(--bad);
    border-color: var(--bad);
  }
  .warn.amber {
    color: #f2c14a;
    border-color: #f2c14a;
  }
  .warn ul {
    margin: 4px 0 0;
    padding-left: 16px;
  }
  .controls {
    background: var(--panel-2);
    border-radius: 7px;
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    font-size: 12.5px;
  }
  .controls > :not(summary) {
    margin-top: 6px;
  }
  .controls summary {
    cursor: pointer;
    color: var(--muted);
  }
  .controls .modes,
  .controls .row {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
    align-items: center;
  }
  .controls input[type='number'] {
    width: 74px;
    padding: 3px 5px;
  }
  .controls select {
    padding: 3px 5px;
    max-width: 100%;
  }
</style>
