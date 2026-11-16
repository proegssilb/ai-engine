<script lang="ts">
  // The machine as in the mockup: one panel per socket, each with that
  // socket's CPU, RAM, GPUs and disks; every box holds one chip per
  // (model, subsystem) piece placed there. Box height grows by one step per
  // doubling of capacity; the usage bars inside stay linear.
  import RamBar from './RamBar.svelte'
  import type { Machine } from './api'
  import { dnd } from './dnd.svelte'
  import { dropReason, type Subject } from './pending'
  import { amount, bigsize, boxHeight, pair, type Chip, type HwBox, type ModelView, type View } from './view'

  let {
    view,
    machine,
    selected,
    over,
    projected,
    onselect,
    onmove,
  }: {
    view: View
    machine: Machine
    selected: { model: string; sub: string | null } | null
    /** boxes the pending plan doesn't fit on: what it would need */
    over: Record<string, string>
    /** used memory under the pending plan, for GPUs (empty when nothing is pending) */
    projected: Record<string, number>
    onselect: (model: string, sub: string | null) => void
    /** a chip or a model was dropped on a box */
    onmove: (subject: Subject, model: string, fromBox: string | null, toBox: string) => void
  } = $props()

  const boxInfo = $derived(Object.fromEntries(Object.entries(view.boxes).map(([k, b]) => [k, { kind: b.kind }])))
  /** why the thing being dragged can't go on this box, or null */
  const reason = (box: string): string | null => {
    const d = dnd.active
    if (!d) return null
    const store = view.models.find((m) => m.name === d.model)?.storeBox ?? null
    return dropReason(d.subject, d.fromBox, box, boxInfo, { storeBox: store })
  }
  let hover = $state<string | null>(null)

  const byName = $derived(Object.fromEntries(view.models.map((m) => [m.name, m])) as Record<string, ModelView>)
  const chipsOf = (b: HwBox): Chip[] => (view.chips[b.key] ?? []).filter((c) => c.kind !== 'runs')
  const runsOf = (b: HwBox): Chip[] => (view.chips[b.key] ?? []).filter((c) => c.kind === 'runs')
  const ours = (b: HwBox) => chipsOf(b).reduce((s, c) => s + c.bytes, 0)
  const isSel = (c: Chip) => !!selected && selected.model === c.model && selected.sub !== null && selected.sub === c.sub
  // header: used / capacity. Devices and disks: what the device reports (the same number
  // their bar draws); RAM: our models' exact memory.
  const used = (b: HwBox) => projected[b.key] ?? Math.max(b.deviceUsed ?? 0, ours(b))
  const headText = (b: HwBox) => (b.kind === 'ram' ? pair(ours(b), b.total) : pair(used(b), b.total))
  // linear usage bar: our pieces in model colours, then whatever else the device reports used
  function segs(b: HwBox) {
    const T = Math.max(b.total, 1)
    const out = chipsOf(b).map((c) => ({ w: (100 * c.bytes) / T, color: byName[c.model]?.color ?? '#888', kind: c.kind }))
    const other = Math.max(0, used(b) - ours(b))
    return { out, other: (100 * other) / T }
  }
  const memBox = (node: number | null) => (node === null ? undefined : machine.memory ? (machine.memory.nodes.length ? machine.memory.nodes.find((n) => n.node === node) : machine.memory.system) : undefined)
</script>

<div class="machine">
  {#each view.panels as p (p.key)}
    <section class="socket" class:detached={p.node === null}>
      <header>
        <h3>{p.title}</h3>
        <span class="muted mono">{p.tag}</span>
      </header>
      {#each [['cpu', 'ram'], ['gpu'], ['disk']] as group (group.join())}
        {@const boxes = p.boxes.filter((b) => group.includes(b.kind))}
        {#if boxes.length}
          <div class="row {group[0]}" style="--n: {boxes.length}">
            {#each boxes as b (b.key)}
              {@const s = b.kind === 'gpu' || b.kind === 'disk' ? segs(b) : null}
              <div
                class="box {b.kind}"
                class:drop={!!dnd.active && b.kind !== 'cpu' && !reason(b.key)}
                class:refuse={!!dnd.active && !!reason(b.key) && reason(b.key) !== 'It is already here.'}
                class:hover={hover === b.key}
                class:over={!!over[b.key]}
                role="group"
                ondragover={(e) => {
                  if (dnd.active && !reason(b.key)) {
                    e.preventDefault()
                    hover = b.key
                  }
                }}
                ondragleave={() => (hover = hover === b.key ? null : hover)}
                ondrop={(e) => {
                  e.preventDefault()
                  hover = null
                  const d = dnd.active
                  if (d && !reason(b.key)) onmove(d.subject, d.model, d.fromBox, b.key)
                  dnd.active = null
                }}
                style="min-height: {boxHeight(b.kind === 'cpu' ? (view.boxes[`ram${b.node}`]?.total ?? 0) : b.total)}px" title={b.path ?? b.detail ?? ''}>
                <div class="bh">
                  <h4>{b.title}</h4>
                  {#if b.kind === 'cpu'}
                    <span class="muted mono">{b.detail}</span>
                  {:else}
                    <span class="muted mono">{headText(b)}</span>
                  {/if}
                </div>
                {#if dnd.active && reason(b.key) && reason(b.key) !== 'It is already here.'}
                  <div class="why">{reason(b.key)}</div>
                {/if}
                {#if over[b.key]}<div class="why over">{over[b.key]}</div>{/if}
                {#if b.kind === 'cpu'}
                  <p class="cpu-note muted">Compute only. Runs what sits in RAM {b.node}.</p>
                  {#each runsOf(b) as c (c.id)}
                    {@render chip(c)}
                  {/each}
                {:else}
                  {#if b.kind === 'ram'}
                    <RamBar mem={memBox(b.node)} total={b.total} node={b.node ?? 0} models={view.models} exact={view.ram} />
                  {:else if s}
                    <div class="bar">
                      {#each s.out as x, i (i)}<span class={x.kind} style="width: {x.w}%; background: {x.color}"></span>{/each}
                      {#if s.other > 0}<span style="width: {s.other}%; background: var(--seg-other, #7d8597)" title="used by other processes"></span>{/if}
                    </div>
                  {/if}
                  {#each chipsOf(b) as c (c.id)}
                    {@render chip(c)}
                  {/each}
                  {#if b.kind === 'disk'}
                    <div class="free">{chipsOf(b).length ? '' : 'Empty. '}{bigsize(Math.max(0, b.total - (b.deviceUsed ?? 0)))} free</div>
                  {:else if !chipsOf(b).length}
                    <div class="empty">Empty.</div>
                  {/if}
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      {/each}
    </section>
  {/each}
  {#if !machine.cuda_build}
    <div class="muted small note">This build has no CUDA backend, so no GPUs are shown.</div>
  {/if}
</div>

{#snippet chip(c: Chip)}
  {@const m = byName[c.model]}
  <button
    class="chip {c.kind}"
    class:piece={c.piece}
    class:sel={isSel(c)}
    class:wrong={!!c.problem}
    class:pending={!!c.pending}
    draggable={c.kind === 'weights' && !!c.edit}
    ondragstart={(e) => {
      if (!(c.kind === 'weights' && c.edit)) return
      e.dataTransfer?.setData('text/plain', `${c.model}: ${c.label}`)
      dnd.active = { subject: c.edit, model: c.model, sub: c.sub, fromBox: c.box }
    }}
    ondragend={() => (dnd.active = null)}
    style="--c: {m?.color}" title={[c.problem ? `Not where planned: ${c.problem}` : '', c.title ?? `${m?.title}: ${c.label}`].filter(Boolean).join('\n')} onclick={() => onselect(c.model, c.sub || null)}>
    <span class="badge">{m?.letter}</span>
    <span class="lbl">{c.problem ? '! ' : ''}{c.label}</span>
    {#if c.badge}<span class="held" class:no={!c.badge.ok}>{c.badge.label}</span>{/if}
    <span class="amt mono">{c.kind === 'runs' ? 'runs' : amount(c)}</span>
  </button>
{/snippet}

<style>
  .machine {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr));
    gap: 14px;
    align-items: stretch;
  }
  .socket {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }
  .socket.detached {
    border-style: dashed;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 2px 4px;
  }
  header h3 {
    font-size: 17px;
  }
  .row {
    display: grid;
    gap: 10px;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 135px), 1fr));
    align-items: start;
  }
  .row.cpu {
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr));
  }
  .box {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    min-width: 0;
  }
  .box.cpu {
    background: transparent;
    border-style: dashed;
  }
  .bh {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0 6px;
    min-width: 0;
  }
  .bh .mono {
    font-size: 11px;
    white-space: nowrap;
    flex: none;
  }
  h4 {
    white-space: nowrap;
    flex: none;
    margin: 0;
    font-size: 14.5px;
    font-weight: 600;
  }
  .cpu-note {
    margin: 0;
    font-size: 13px;
    line-height: 1.3;
  }
  .bar {
    display: flex;
    height: 6px;
    background: var(--bg);
    border-radius: 3px;
    overflow: hidden;
  }
  .bar span {
    display: block;
    height: 100%;
  }
  .bar span.kv {
    opacity: 0.6;
  }
  .chip {
    display: flex;
    align-items: center;
    gap: 6px;
    text-align: left;
    padding: 3px 6px;
    border: 1.5px solid var(--c);
    border-radius: 7px;
    background: color-mix(in srgb, var(--c) 14%, var(--bg));
    font-size: 12.5px;
    width: 100%;
  }
  .chip:hover:not(:disabled) {
    border-color: var(--c);
    filter: brightness(1.15);
  }
  .chip.piece {
    border-style: dashed;
  }
  .chip.sel {
    border-color: #fff;
    border-style: solid;
    box-shadow: 0 0 0 1px #fff;
  }
  .chip.runs .lbl {
    white-space: normal;
  }
  .chip[draggable='true'] {
    cursor: grab;
  }
  .chip.pending {
    background:
      repeating-linear-gradient(135deg, color-mix(in srgb, var(--c) 35%, transparent) 0 3px, transparent 3px 7px),
      color-mix(in srgb, var(--c) 10%, var(--bg));
  }
  .pend {
    flex: none;
    font-size: 10px;
    padding: 0 5px;
    border-radius: 8px;
    border: 1px dotted var(--c);
    color: var(--text);
    white-space: nowrap;
  }
  .box.drop {
    outline: 2px dashed #7bd88f;
    outline-offset: -2px;
  }
  .box.hover {
    outline-style: solid;
    background: color-mix(in srgb, #7bd88f 10%, var(--panel-2));
  }
  .box.refuse {
    outline: 2px dashed var(--bad);
    outline-offset: -2px;
    opacity: 0.8;
  }
  .box.over {
    border-color: var(--bad);
  }
  .why {
    font-size: 11.5px;
    color: var(--bad);
    line-height: 1.25;
  }
  .why.over {
    font-weight: 600;
  }
  .chip.wrong {
    border-color: var(--bad);
    border-style: solid;
    box-shadow: 0 0 0 1px var(--bad);
  }
  .chip.wrong .lbl {
    color: var(--bad);
  }
  .held {
    flex: none;
    font-size: 10.5px;
    padding: 0 5px;
    border-radius: 8px;
    border: 1px solid var(--ok);
    color: var(--ok);
    white-space: nowrap;
  }
  .held.no {
    border-color: #f2c14a;
    color: #f2c14a;
  }
  .chip.scratch .lbl {
    font-style: italic;
  }
  .badge {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    background: var(--c);
    color: #10131a;
    font-weight: 700;
    font-size: 11px;
    display: grid;
    place-items: center;
    flex: none;
  }
  .lbl {
    flex: 1 1 0;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .amt {
    color: var(--muted);
    white-space: nowrap;
    font-size: 12px;
    margin-left: auto;
    flex: none;
  }
  .free,
  .empty {
    flex: 1;
    display: grid;
    place-items: center;
    border: 1px dashed var(--line);
    border-radius: 8px;
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px;
    min-height: 40px;
    text-align: center;
  }
  .note {
    grid-column: 1 / -1;
  }
</style>
