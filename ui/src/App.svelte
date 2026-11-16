<script lang="ts">
  import { untrack } from 'svelte'
  import MachineView from './lib/MachineView.svelte'
  import ModelCard from './lib/ModelCard.svelte'
  import { ModelEdit } from './lib/edit.svelte'
  import { api, bytes, sockets, type ApplyResult, type CheckResult, type Edit, type Machine, type ModelTag, type Placement, type Running } from './lib/api'
  import { demoCatalog, demoMachine, demoPlacements, demoTags, fakeCheck } from './lib/demo'
  import { localPlan, movePiece, putAll, tierOfBox, type Ctx, type PlanLabel, type Role, type Subject, type Target } from './lib/pending'
  import { anatomyCatalog, buildView, size, type PendingPlan } from './lib/view'

  // /ui/?demo renders the mockup's own example from fixtures; &pending preloads two pending edits
  const qs = new URLSearchParams(location.search)
  const demo = qs.has('demo')
  const demoPending = demo && qs.has('pending')
  // ?edit opens the size and share fields of the selected subsystem
  const openEdit = qs.has('edit')
  const catalog = demo ? demoCatalog : anatomyCatalog

  let models = $state<ModelTag[]>(demo ? demoTags : [])
  let running = $state<Running[]>([])
  let machine = $state<Machine | null>(demo ? demoMachine : null)
  let placements = $state<Record<string, Placement | undefined>>(demo ? demoPlacements : {})
  let selected = $state<{ model: string; sub: string | null } | null>(null)
  let error = $state('')
  let busy = $state('')
  let applyResult = $state<ApplyResult | null>(null)
  let pullName = $state('')
  let pullStatus = $state('')

  // ---- pending edits ------------------------------------------------------
  const edits = new Map<string, ModelEdit>()
  /** bumped when an edit object is created, so what depends on the set of edits notices */
  let editsVersion = $state(0)
  const edit = (name: string) => {
    let e = edits.get(name)
    if (!e) edits.set(name, (e = new ModelEdit()))
    return e
  }

  const gpuCount = $derived(machine ? machine.devices.filter((d) => d.gpu).length : 0)
  const socketIds = $derived(machine ? sockets(machine.numa).map((n) => n.node) : [])
  const ctxFor = (name: string): Ctx => ({ moe: !!placements[name]?.anatomy.moe, gpus: gpuCount, sockets: socketIds.length })

  /** the pending edits as one string: changes when any edit does */
  const sig = $derived.by(() => {
    void editsVersion
    return JSON.stringify([...edits].filter(([, e]) => e.dirty).map(([n, e]) => [n, e.plain()]))
  })
  const dirtyNames = $derived.by(() => {
    void sig
    return [...edits].filter(([, e]) => e.dirty).map(([n]) => n)
  })

  const baseView = $derived(machine ? buildView({ machine, tags: models, placements, catalog }) : null)

  // what the check last said (for the edits it was asked about)
  let check = $state<{ sig: string; result: CheckResult | null; error: string; state: 'idle' | 'checking' | 'done' }>({ sig: '', result: null, error: '', state: 'idle' })
  let checkSeq = 0
  let timer: ReturnType<typeof setTimeout> | undefined

  /** subsystem label -> tiers whose bytes differ between two plans */
  const changedOf = (a: Record<string, Record<string, number>>, b: Record<string, Record<string, number>>) => {
    const out: Record<string, string[]> = {}
    for (const l of new Set([...Object.keys(a), ...Object.keys(b)])) {
      const tiers = new Set([...Object.keys(a[l] ?? {}), ...Object.keys(b[l] ?? {})])
      const d = [...tiers].filter((t) => Math.abs((a[l]?.[t] ?? 0) - (b[l]?.[t] ?? 0)) > 1 << 20)
      if (d.length) out[l] = d
    }
    return out
  }
  const currentPlanned = (name: string) => machine?.models.find((m) => m.model === name)?.planned ?? placements[name]?.loaded?.planned ?? placements[name]?.intended ?? {}

  /** each pending model as the check (or, until it answers, a local estimate) says it would be placed */
  const pendingPlans = $derived.by((): Record<string, PendingPlan> => {
    const out: Record<string, PendingPlan> = {}
    const bv = baseView
    if (!bv) return out
    for (const name of dirtyNames) {
      const p = placements[name]
      if (!p) continue
      const e = edit(name)
      const srv = check.sig === sig && !check.error ? check.result?.models[name] : undefined
      // what the model's KV cache and scratch hold now, per tier, as measured
      const curExtra: Record<string, number> = {}
      for (const [box, chips] of Object.entries(bv.chips))
        for (const c of chips) if (c.model === name && (c.kind === 'kv' || c.kind === 'scratch')) curExtra[tierOfBox(box)] = (curExtra[tierOfBox(box)] ?? 0) + c.bytes
      if (srv && !srv.error && Object.keys(srv.planned).length) {
        const changed = changedOf(srv.planned, currentPlanned(name))
        // the check's dry run says what the KV cache and scratch would add up to; when the layers
        // don't move they stay as measured now (the dry run reserves more than is resident yet)
        const layersMoved = Object.keys(changed).some((l) => catalog(l, name).edit === 'layers')
        let extra: Record<string, number> = layersMoved || e.mode === 'auto' ? {} : curExtra
        if (layersMoved || e.mode === 'auto') {
          for (const d of check.result!.devices) {
            const by = d.by_model[name]
            if (!by) continue
            const tier = d.name === 'RAM' ? 'ram' : `gpu${d.name.replace(/^CUDA/, '')}`
            const w = Object.values(srv.planned).reduce((s, t) => s + (t[tier] ?? 0), 0)
            if (by > w) extra[tier] = by - w
          }
        }
        out[name] = { planned: srv.planned, extra, hostNode: e.hostNode, changed }
        continue
      }
      const cur = currentPlanned(name)
      const labels: PlanLabel[] = Object.entries(p.anatomy.subsystems).map(([label, b]) => ({ label, bytes: b, edit: catalog(label, name).edit, current: cur[label] ?? {} }))
      const extra = curExtra
      const planned = localPlan(labels, e.plain())
      // the KV cache follows the layers: shared out over the tiers the layers go to
      const lay = labels.filter((l) => l.edit === 'layers')
      const layTotal = lay.reduce((sum, l) => sum + l.bytes, 0)
      const kvTotal = Object.values(extra).reduce((a, b) => a + b, 0)
      let moved = extra
      if (e.mode === 'custom' && layTotal > 0 && kvTotal > 0) {
        moved = {}
        for (const l of lay) for (const [t, b] of Object.entries(planned[l.label] ?? {})) moved[t] = (moved[t] ?? 0) + Math.round((kvTotal * b) / layTotal)
      }
      out[name] = { planned, extra: moved, hostNode: e.hostNode, changed: changedOf(planned, cur) }
    }
    return out
  })

  const view = $derived(machine ? buildView({ machine, tags: models, placements, catalog, pending: pendingPlans }) : null)

  /**
   * Each GPU's used memory under the pending plan: what it holds now, minus what the
   * edited models hold there now, plus what the plan puts there (the same chips the boxes draw,
   * so header, bar and chips agree; for GPUs this is the check's total - available + need).
   * RAM boxes already add up their own chips. Disks are not here: a disk's used space is the
   * filesystem's, and moving a piece off the model file doesn't change it.
   */
  const projected = $derived.by((): Record<string, number> => {
    const out: Record<string, number> = {}
    const bv = baseView
    const v = view
    if (!bv || !v || !dirtyNames.length) return out
    const held = (vw: typeof v, box: string) => (vw.chips[box] ?? []).filter((c) => dirtyNames.includes(c.model) && c.kind !== 'runs').reduce((sum, c) => sum + c.bytes, 0)
    for (const [k, b] of Object.entries(v.boxes)) {
      if (b.kind === 'gpu') out[k] = Math.max(0, (b.deviceUsed ?? 0) - held(bv, k) + held(v, k))
    }
    return out
  })

  function runCheck(forSig: string) {
    const seq = ++checkSeq
    const list: Edit[] = []
    try {
      for (const n of dirtyNames) list.push({ model: n, intent: edit(n).intent(ctxFor(n)) })
    } catch (e) {
      check = { sig: forSig, result: null, error: String(e instanceof Error ? e.message : e), state: 'done' }
      return
    }
    const ask = demo
      ? Promise.resolve(
          fakeCheck(
            machine!,
            dirtyNames.map((n) => {
              const pp = pendingPlans[n]
              const cur: Record<string, number> = {}
              for (const [box, chips] of Object.entries(baseView?.chips ?? {}))
                for (const c of chips) if (c.model === n && c.kind !== 'runs') cur[tierOfBox(box)] = (cur[tierOfBox(box)] ?? 0) + c.bytes
              return { name: n, planned: pp.planned, extra: pp.extra, current: cur }
            }),
          ),
        )
      : api.check(list)
    ask.then(
      (result) => {
        if (seq === checkSeq) check = { sig: forSig, result, error: '', state: 'done' }
      },
      (e) => {
        if (seq === checkSeq) check = { sig: forSig, result: null, error: String(e instanceof Error ? e.message : e), state: 'done' }
      },
    )
  }

  // every edit re-checks, after a short pause; answers to older edits are dropped
  $effect(() => {
    const s = sig
    untrack(() => {
      clearTimeout(timer)
      checkSeq++
      if (!dirtyNames.length) {
        check = { sig: s, result: null, error: '', state: 'idle' }
        return
      }
      check = { ...check, state: 'checking' }
      timer = setTimeout(() => runCheck(s), demo ? 50 : 400)
    })
  })

  /** boxes the pending plan doesn't fit on, with what it would need */
  const over = $derived.by((): Record<string, string> => {
    const out: Record<string, string> = {}
    if (!view || check.sig !== sig || !check.result) return out
    for (const d of check.result.devices) {
      if (d.fits) continue
      const text = `Needs ${size(d.need)}; ${size(d.available)} available`
      if (d.name === 'RAM') {
        for (const [k, b] of Object.entries(view.boxes)) if (b.kind === 'ram' && (view.chips[k] ?? []).some((c) => c.pending)) out[k] = text
      } else out[`gpu${d.name.replace(/^CUDA/, '')}`] = text
    }
    return out
  })
  const problems = $derived(check.sig === sig && check.result ? [...check.result.problems, ...Object.entries(check.result.models).flatMap(([n, m]) => (m.error ? [`${n}: ${m.error}`] : []))] : [])

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
  let preloaded = false
  $effect(() => {
    const v = baseView
    const pl = placements
    if (!v) return
    untrack(() => {
      let made = false
      for (const m of v.models) {
        const p = pl[m.name]
        if (!p) continue
        if (!edits.has(m.name)) made = true
        edit(m.name).seed(p, gpuCount, m)
      }
      if (made) editsVersion++
      if (!selected && v.models.length) {
        const m = v.models[0]
        selected = { model: m.name, sub: m.subs.find((s) => s.role === 'bulk')?.id ?? null }
      }
      if (demoPending && !preloaded && v.models.length > 3) {
        preloaded = true
        // one edit that fits (A's core layers to GPU 2) and one that overflows GPU 4 (all of B on it)
        onmove('layers', v.models[0].name, 'gpu0', 'gpu2')
        putAll(edit(v.models[1].name), { kind: 'gpu', index: 4 }, ctxFor(v.models[1].name))
        edit(v.models[1].name).dirty = true
        editsVersion++
      }
    })
  })

  $effect(() => {
    refresh()
    if (demo) return
    // polling must not overwrite what is being edited: seeding skips edited models
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  })

  const select = (model: string, sub: string | null) => (selected = { model, sub })

  /** a chip or a model was dropped on a box, or a menu said where */
  function onmove(subject: Subject, model: string, fromBox: string | null, toBox: string) {
    if (subject === 'none') return
    const e = edit(model)
    const ctx = ctxFor(model)
    if (subject === 'model') {
      if (toBox.startsWith('gpu')) putAll(e, { kind: 'gpu', index: Number(toBox.slice(3)) }, ctx)
      else if (toBox.startsWith('ram')) putAll(e, { kind: 'ram', node: Number(toBox.slice(3)) }, ctx)
    } else movePiece(e, subject, fromBox ?? toBox, toBox, ctx.sockets)
    e.dirty = true
    editsVersion++
  }
  function onputall(model: string, target: Target) {
    putAll(edit(model), target, ctxFor(model))
    edit(model).dirty = true
    editsVersion++
  }
  function onmovesingle(model: string, role: Role, fromBox: string | null, toBox: string) {
    onmove(role, model, fromBox, toBox)
  }

  /** Apply placement: every pending edit goes in one call; it is checked together and applied only if it fits. */
  async function apply() {
    if (demo) {
      error = 'This is the demo page: nothing is applied.'
      return
    }
    error = ''
    applyResult = null
    const list: Edit[] = []
    try {
      for (const n of dirtyNames) list.push({ model: n, intent: edit(n).intent(ctxFor(n)) })
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
      return
    }
    if (!list.length) {
      error = 'No pending edits. Drag a chip, move a divider or pick a destination first.'
      return
    }
    try {
      busy = `Applying ${list.length} model${list.length === 1 ? '' : 's'}: checking they fit together, then unloading, loading and verifying each…`
      const res = await api.apply(list)
      applyResult = res
      if (res.applied) {
        for (const r of res.results) if (r.ok) edit(r.model).dirty = false
        editsVersion++
      }
      await refresh()
      for (const m of view?.models ?? []) if (placements[m.name] && !edit(m.name).dirty) edit(m.name).seed(placements[m.name]!, gpuCount, m)
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
    } finally {
      busy = ''
    }
  }

  /** drop every pending edit */
  function discardAll() {
    for (const n of dirtyNames) discard(n)
    applyResult = null
    error = ''
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
    const m = baseView?.models.find((x) => x.name === name)
    if (p) e.seed(p, gpuCount, m)
    editsVersion++
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
</script>

<div class="app">
  <header>
    <div class="title">
      <h1>Placement</h1>
      {#if view}<div class="muted mono">{view.summary}</div>{/if}
      {#if demo}<div class="muted small">Demo: the mockup's own example, drawn from fixture data.</div>{/if}
    </div>
    <div class="actions">
      {#if dirtyNames.length}<button onclick={discardAll} disabled={!!busy} title="Drop every pending edit">Discard edits</button>{/if}
      <button class="primary" onclick={apply} disabled={!!busy} title="Check that every pending edit fits together, then save, reload and verify those models">
        Apply placement{dirtyNames.length ? ` (${dirtyNames.length})` : ''}
      </button>
    </div>
  </header>
  <div class="status" aria-live="polite">
    {#if busy}<span class="muted">{busy}</span>{/if}
    {#if error}<span class="error">{error}</span>{/if}
    {#if dirtyNames.length && !busy}
      {#if check.state === 'checking' || check.sig !== sig}
        <span class="muted">checking {dirtyNames.length} pending edit{dirtyNames.length === 1 ? '' : 's'}…</span>
      {:else if check.error}
        <span class="error">{check.error}</span>
      {:else if problems.length}
        <span class="error">Pending edits don't fit: {problems.join('; ')}</span>
      {:else}
        <span class="ok">{dirtyNames.length} pending edit{dirtyNames.length === 1 ? '' : 's'} fit.</span>
      {/if}
    {/if}
    {#if applyResult}
      {#if !applyResult.applied}
        <span class="error">Nothing was applied: {applyResult.check.problems.join('; ') || 'the edits did not fit'}</span>
      {:else}
        {#each applyResult.results as r (r.model)}
          <span class={r.ok ? (r.misplaced ? 'error' : 'ok') : 'error'}>
            {r.model.replace(/^hf\.co\//, '')}: {r.ok ? (r.misplaced ? `loaded, ${r.misplaced} weights not where planned` : 'applied and verified') : `failed: ${r.error}${r.restored ? ' (previous placement restored)' : ''}`}
          </span>
        {/each}
      {/if}
    {/if}
  </div>

  <main>
    <aside>
      {#if !demo}
        <div class="pull">
          <input placeholder="Pull a model: hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M" bind:value={pullName} onkeydown={(e) => e.key === 'Enter' && pull()} />
          <button onclick={pull} disabled={!!pullStatus && !pullStatus.startsWith('failed')}>Pull</button>
        </div>
        {#if pullStatus}<div class="muted small">{pullStatus}</div>{/if}
        {#if selected}
          <button class="link" onclick={() => selected && remove(selected.model)}>Delete selected model</button>
        {/if}
      {/if}
      <h2>Models</h2>
      {#if view}
        {#each view.models as mv (mv.name)}
          <ModelCard
            {mv}
            {view}
            gpus={gpuCount}
            {selected}
            edit={edit(mv.name)}
            running={isRunning(mv.name)}
            {openEdit}
            sockets={socketIds}
            {busy}
            onselect={select}
            onunload={() => unload(mv.name)}
            ondiscard={() => discard(mv.name)}
            {onputall}
            onmove={onmovesingle}
          />
        {/each}
      {/if}
      {#if !models.length}<div class="muted">No models yet. Pull one above.</div>{/if}
    </aside>

    <section class="content">
      <h2>Machine</h2>
      {#if view && machine}
        <MachineView {view} {machine} {selected} {over} {projected} onselect={select} {onmove} />
        <p class="legend muted">
          <span>Letter and colour = which model.</span>
          <span>Dashed outline = one piece of a split subsystem.</span>
          <span>Hatched fill = a pending edit, not applied yet.</span>
          <span>White outline = the subsystem selected on the left.</span>
          <span>Thin bar = how full, on a linear scale.</span>
        </p>
      {/if}
    </section>
  </main>
</div>

<style>
  .app {
    max-width: 1700px;
    margin: 0 auto;
    height: 100vh;
    display: flex;
    flex-direction: column;
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
  .actions {
    display: flex;
    gap: 10px;
    padding-top: 10px;
  }
  .status {
    padding: 0 24px 4px;
    display: flex;
    flex-wrap: wrap;
    gap: 2px 16px;
    font-size: 13px;
  }
  main {
    display: grid;
    grid-template-columns: minmax(300px, 320px) 1fr;
    gap: 18px;
    padding: 8px 24px 0;
    align-items: stretch;
    flex: 1;
    min-height: 0;
  }
  aside,
  .content {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
    min-height: 0;
    overflow-y: auto;
    padding-bottom: 28px;
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
    .app {
      height: auto;
      display: block;
    }
    main {
      grid-template-columns: 1fr;
    }
    aside,
    .content {
      overflow: visible;
    }
    header,
    main,
    .status {
      padding-left: 16px;
      padding-right: 16px;
    }
  }
</style>
