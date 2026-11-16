<script lang="ts">
  // One model as in the mockup: letter badge, name, a line about it, its total,
  // and its subsystems each with where it goes. A split subsystem (the expert
  // bank) opens into a bar with a segment per tier: this is where placement is
  // edited, by size or share per tier (D6).
  import SplitBar from './SplitBar.svelte'
  import { tierOfBox, type ModelEdit } from './edit.svelte'
  import { size, type ModelView, type SubView, type View } from './view'

  let {
    mv,
    view,
    gpus,
    selected,
    edit,
    running,
    openEdit = false,
    onselect,
  }: {
    mv: ModelView
    view: View
    gpus: number
    selected: { model: string; sub: string | null } | null
    edit: ModelEdit
    running: boolean
    openEdit?: boolean
    onselect: (model: string, sub: string | null) => void
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
  {#each mv.subs as s (s.id)}
    {@const sel = selected?.model === mv.name && selected.sub === s.id}
    <div class="sub" class:sel>
      <button class="row" onclick={() => onselect(mv.name, s.id)}>
        <span class="nm">{s.label}</span>
        <span class="amt mono">{size(s.bytes)}{#if sel && s.split}{" · split in " + s.dests.length}{:else if where(s)}{" → " + where(s)}{/if}</span>
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
  }
  .amt {
    color: var(--muted);
    text-align: right;
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
</style>
