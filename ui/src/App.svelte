<script lang="ts">
  import { untrack } from 'svelte'
  import Details from './lib/Details.svelte'
  import MachineView from './lib/MachineView.svelte'
  import ModelCard from './lib/ModelCard.svelte'
  import { ModelEdit } from './lib/edit.svelte'
  import { api, bytes, sockets, type Machine, type ModelTag, type Placement, type Running } from './lib/api'
  import { demoCatalog, demoMachine, demoPlacements, demoTags } from './lib/demo'
  import { BASE_GIB, buildView, scaleBarHeight } from './lib/view'

  // /ui/?demo renders the mockup's own example from fixtures
  const qs = new URLSearchParams(location.search)
  const demo = qs.has('demo')
  // ?edit opens the size and share fields of the selected subsystem
  const openEdit = qs.has('edit')

  let models = $state<ModelTag[]>(demo ? demoTags : [])
  let running = $state<Running[]>([])
  let machine = $state<Machine | null>(demo ? demoMachine : null)
  let placements = $state<Record<string, Placement | undefined>>(demo ? demoPlacements : {})
  let selected = $state<{ model: string; sub: string | null } | null>(null)
  let error = $state('')
  let busy = $state('')
  let pullName = $state('')
  let pullStatus = $state('')

  const edits = new Map<string, ModelEdit>()
  const edit = (name: string) => {
    let e = edits.get(name)
    if (!e) edits.set(name, (e = new ModelEdit()))
    return e
  }

  const view = $derived(machine ? buildView({ machine, tags: models, placements, catalog: demo ? demoCatalog : undefined }) : null)
  const gpuCount = $derived(machine ? machine.devices.filter((d) => d.gpu).length : 0)
  const socketIds = $derived(machine ? sockets(machine.numa).map((n) => n.node) : [])
  const selModel = $derived(view && selected ? (view.models.find((m) => m.name === selected!.model) ?? null) : null)

  async function refresh() {
    if (demo) return
    try {
      const [tags, ps, m] = await Promise.all([api.tags(), api.ps(), api.machine()])
      const pl = await Promise.all(tags.map((t) => api.placement(t.name).catch(() => undefined)))
      models = tags
      running = ps
      machine = m
      placements = Object.fromEntries(tags.map((t, i) => [t.name, pl[i]]))
      error = ''
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
    }
  }

  // Take each model's saved placement (or where it is now) as the starting point for edits.
  $effect(() => {
    const v = view
    const pl = placements
    if (!v) return
    untrack(() => {
      for (const m of v.models) {
        const p = pl[m.name]
        if (p) edit(m.name).seed(p, gpuCount, m)
      }
      if (!selected && v.models.length) {
        const m = v.models[0]
        selected = { model: m.name, sub: m.subs.find((s) => s.role === 'bulk')?.id ?? null }
      }
    })
  })

  $effect(() => {
    refresh()
    if (demo) return
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  })

  const select = (model: string, sub: string | null) => (selected = { model, sub })

  async function save(name: string, auto: boolean) {
    const p = placements[name]
    if (!p) return
    const e = edit(name)
    const intent = auto ? null : e.intent(p, gpuCount, socketIds.length)
    await api.savePlacement(name, intent, true)
    await api.load(name)
    e.dirty = false
  }

  /** Apply placement: every model with edits is saved, unloaded and loaded with them. */
  async function apply() {
    if (demo) {
      error = 'This is the demo page: nothing is applied.'
      return
    }
    error = ''
    try {
      const todo = [...edits].filter(([, e]) => e.dirty).map(([n]) => n)
      if (!todo.length) {
        error = 'No edits to apply. Select a subsystem and change its split first.'
        return
      }
      for (const name of todo) {
        busy = `Saving and loading ${name}, then checking every weight…`
        await save(name, edit(name).mode === 'auto')
      }
      await refresh()
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
    } finally {
      busy = ''
    }
  }

  /** Auto-place: the selected model goes back to llama.cpp's own fit for the memory free right now. */
  async function autoPlace() {
    if (demo) {
      error = 'This is the demo page: nothing is applied.'
      return
    }
    if (!selected) return
    error = ''
    try {
      busy = `Auto-placing ${selected.model}…`
      edit(selected.model).mode = 'auto'
      await save(selected.model, true)
      await refresh()
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
    } finally {
      busy = ''
    }
  }

  async function unload(name: string) {
    busy = 'Unloading…'
    try {
      await api.unload(name)
      await refresh()
    } finally {
      busy = ''
    }
  }

  function discard(name: string) {
    const e = edit(name)
    e.dirty = false
    const p = placements[name]
    const m = view?.models.find((x) => x.name === name)
    if (p) e.seed(p, gpuCount, m)
  }

  async function pull() {
    const name = pullName.trim()
    if (!name || demo) return
    pullStatus = 'starting…'
    try {
      await api.pull(name, (status, done, total) => {
        pullStatus = total ? `${status}: ${((100 * (done ?? 0)) / total).toFixed(1)}% of ${bytes(total)}` : status
      })
      pullStatus = ''
      pullName = ''
      await refresh()
    } catch (e) {
      pullStatus = `failed: ${e instanceof Error ? e.message : e}`
    }
  }

  async function remove(name: string) {
    if (demo || !confirm(`Delete ${name} from the store?`)) return
    await api.remove(name)
    if (selected?.model === name) selected = null
    await refresh()
  }

  const isRunning = (name: string) => running.some((r) => r.name === name)
  // the legend of box heights: one step per doubling
  const scale = [
    ['16G', 16],
    ['32G', 32],
    ['64G', 64],
    ['128G', 128],
    ['1T', 1024],
    ['2T', 2048],
    ['4T', 4096],
  ] as const
</script>

<div class="app">
  <header>
    <div class="title">
      <h1>Placement</h1>
      {#if view}<div class="muted mono">{view.summary}</div>{/if}
      {#if demo}<div class="muted small">Demo: the mockup's own example, drawn from fixture data.</div>{/if}
    </div>
    <div class="scale" aria-label="Box height scale">
      <div class="st">
        <strong>Box height scale</strong>
        <span class="muted">One step taller per doubling of capacity</span>
      </div>
      <div class="bars">
        {#each scale as [label, gb] (label)}
          <div class="b">
            <span style="height: {scaleBarHeight(gb * (1 << 30))}px"></span>
            <em class="muted mono">{label}</em>
          </div>
        {/each}
      </div>
    </div>
    <div class="actions">
      <button onclick={autoPlace} disabled={!!busy} title="Put the selected model back to llama.cpp's own placement for the memory free right now">Auto-place</button>
      <button class="primary" onclick={apply} disabled={!!busy} title="Save, reload and verify every model with edits">Apply placement</button>
    </div>
  </header>
  {#if error || busy}
    <div class="status">
      {#if busy}<span class="muted">{busy}</span>{/if}
      {#if error}<span class="error">{error}</span>{/if}
    </div>
  {/if}

  <main>
    <aside>
      <h2>Models</h2>
      {#if view}
        {#each view.models as mv (mv.name)}
          <ModelCard {mv} {view} gpus={gpuCount} {selected} edit={edit(mv.name)} running={isRunning(mv.name)} {openEdit} onselect={select} />
        {/each}
      {/if}
      {#if !models.length}<div class="muted">No models yet. Pull one below.</div>{/if}
      {#if !demo}
        <div class="pull">
          <input placeholder="hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M" bind:value={pullName} onkeydown={(e) => e.key === 'Enter' && pull()} />
          <button onclick={pull} disabled={!!pullStatus && !pullStatus.startsWith('failed')}>Pull</button>
        </div>
        {#if pullStatus}<div class="muted small">{pullStatus}</div>{/if}
        {#if selected}
          <button class="link" onclick={() => selected && remove(selected.model)}>Delete selected model</button>
        {/if}
      {/if}
    </aside>

    <section class="content">
      <h2>Machine</h2>
      {#if view && machine}
        <MachineView {view} {machine} {selected} onselect={select} />
        <p class="legend muted">
          <span>Letter and colour = which model.</span>
          <span>Dashed outline = one piece of a split subsystem.</span>
          <span>White outline = the subsystem selected on the left.</span>
          <span>Thin bar = how full, on a linear scale.</span>
        </p>
        {#if selModel}
          <Details
            mv={selModel}
            {view}
            p={placements[selModel.name] ?? null}
            edit={edit(selModel.name)}
            sockets={socketIds}
            {busy}
            onunload={() => unload(selModel.name)}
            ondiscard={() => discard(selModel.name)}
          />
        {/if}
      {/if}
    </section>
  </main>
</div>

<style>
  .app {
    max-width: 1700px;
    margin: 0 auto;
  }
  header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    padding: 20px 24px 8px;
    gap: 16px;
  }
  .title {
    flex: 1 1 320px;
    padding-top: 8px;
  }
  .scale {
    display: flex;
    gap: 16px;
    align-items: flex-end;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 10px 14px;
    flex: 0 1 auto;
    max-width: 100%;
  }
  .st {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 13px;
    max-width: 190px;
  }
  .bars {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  .b {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-end;
    gap: 3px;
  }
  .b span {
    display: block;
    width: 22px;
    background: #3a4050;
    border-radius: 3px 3px 0 0;
  }
  .b em {
    font-style: normal;
    font-size: 11px;
  }
  .actions {
    display: flex;
    gap: 10px;
    padding-top: 10px;
  }
  .status {
    padding: 0 24px;
    display: flex;
    gap: 16px;
  }
  main {
    display: grid;
    grid-template-columns: minmax(300px, 320px) 1fr;
    gap: 18px;
    padding: 8px 24px 28px;
    align-items: start;
  }
  aside,
  .content {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 22px;
    margin: 2px 4px;
    font-size: 13px;
  }
  .pull {
    display: flex;
    gap: 6px;
    margin-top: 8px;
  }
  .pull input {
    flex: 1;
    min-width: 0;
  }
  .small {
    font-size: 12px;
  }
  .link {
    background: none;
    border: none;
    color: var(--muted);
    text-align: left;
    padding: 0;
    font-size: 12px;
  }
  @media (max-width: 900px) {
    main {
      grid-template-columns: 1fr;
    }
    header,
    main,
    .status {
      padding-left: 16px;
      padding-right: 16px;
    }
    .scale {
      overflow-x: auto;
    }
  }
</style>
