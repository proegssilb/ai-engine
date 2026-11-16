<script lang="ts">
  // What the mockup lacks, for the selected model: the context and socket it
  // is placed with, whether every weight is where planned, where its host side
  // runs, whether the RAM tier is held, speed, and its memory per device.
  // The memory table adds up the same pieces the chips show.
  import type { ModelEdit } from './edit.svelte'
  import { bytes, type Placement } from './api'
  import { size, type Chip, type ModelView, type View } from './view'

  let {
    mv,
    view,
    p,
    edit,
    sockets,
    busy,
    onunload,
    ondiscard,
  }: {
    mv: ModelView
    view: View
    p: Placement | null
    edit: ModelEdit
    sockets: number[]
    busy: string
    onunload: () => void
    ondiscard: () => void
  } = $props()

  const loaded = $derived(p?.loaded ?? null)
  const touched = () => {
    edit.mode = 'custom'
    edit.dirty = true
  }

  // One row per box that holds something of this model.
  const rows = $derived.by(() => {
    const out: { box: string; name: string; weights: number; kv: number; scratch: number; free: number | null; disk: boolean }[] = []
    for (const [box, chips] of Object.entries(view.chips)) {
      const mine = chips.filter((c: Chip) => c.model === mv.name && c.kind !== 'runs')
      if (!mine.length) continue
      const sum = (k: Chip['kind']) => mine.filter((c) => c.kind === k).reduce((s, c) => s + c.bytes, 0)
      const m = /^gpu(\d+)$/.exec(box)
      const row = m ? loaded?.measured?.find((r) => r.name === `CUDA${m[1]}`) : null
      out.push({ box, name: view.name(box), weights: sum('weights'), kv: sum('kv'), scratch: sum('scratch'), free: row && row.total > 0 ? row.free : null, disk: box.startsWith('disk') })
    }
    const o = (b: string) => (b.startsWith('gpu') ? 0 : b.startsWith('ram') ? 1 : 2)
    return out.sort((a, b) => o(a.box) - o(b.box) || a.box.localeCompare(b.box, undefined, { numeric: true }))
  })

  // Weights outside GPU memory make llama.cpp stage them on GPU 0 for big batches.
  const offloadScratch = $derived(
    !!loaded?.measured?.some((m) => m.name === 'CUDA0' && m.model + m.context + m.compute > 0) &&
      Object.entries(loaded?.planned ?? {}).some(([sub, t]) => !['embeddings', 'output head', 'other'].includes(sub) && (t.ram ?? 0) + (t.disk ?? 0) > 0),
  )
  const host = $derived(loaded?.host ?? null)
  const hostScratch = $derived(loaded?.measured?.find((r) => r.kind === 'host') ?? null)
</script>

<section class="details">
  <header>
    <h3><span class="badge" style="background: {mv.color}">{mv.letter}</span> Details · {mv.title}</h3>
    <div class="muted">
      {#if p}
        {p.anatomy.architecture} · {p.anatomy.layers} layers · {bytes(p.anatomy.bytes)}
        {#if p.anatomy.moe}· {p.anatomy.moe.experts} experts, {p.anatomy.moe.experts_used} active{/if}
        {#if !p.anatomy.supported}· <span class="error">architecture not yet checked</span>{/if}
      {/if}
    </div>
  </header>

  {#if p}
    <div class="grid">
      <div class="box">
        <h5>Placement mode</h5>
        <label><input type="radio" bind:group={edit.mode} value="auto" onchange={() => (edit.dirty = true)} /> Automatic <span class="muted">(llama.cpp decides, by free memory)</span></label>
        <label><input type="radio" bind:group={edit.mode} value="custom" onchange={() => (edit.dirty = true)} /> Custom split <span class="muted">(edit it in the model's card)</span></label>
        <div class="row">
          <label>Context <input type="number" min="512" step="512" bind:value={edit.nCtx} oninput={touched} /> tokens</label>
          {#if p.anatomy.moe}
            <label>Expert cache <input type="number" min="0" step="0.5" bind:value={edit.cacheGiB} oninput={touched} /> GiB on GPU</label>
          {/if}
        </div>
        {#if sockets.length > 1}
          <label>Host socket
            <select bind:value={edit.hostNode} onchange={touched}>
              <option value={null}>Automatic: next to its GPU</option>
              {#each sockets as n (n)}<option value={n}>Socket {n}</option>{/each}
            </select>
          </label>
          <div class="muted note">The model's RAM tier, CPU threads and host-side buffers all go on one socket, never split. Applies to custom splits.</div>
        {/if}
        <div class="row">
          {#if loaded}<button onclick={onunload} disabled={!!busy}>Unload</button>{/if}
          {#if edit.dirty}<button onclick={ondiscard} disabled={!!busy}>Discard edits</button>{/if}
          {#if busy}<span class="muted">{busy}</span>{/if}
        </div>
      </div>

      {#if loaded}
        <div class="box">
          <h5>Verified placement</h5>
          <div>
            {loaded.description} · context {loaded.n_ctx} · loaded in {(loaded.load_ms / 1000).toFixed(1)} s ·
            {#if loaded.misplaced === 0}
              <span class="ok">every weight verified where planned</span>
            {:else}
              <span class="error">{loaded.misplaced} weights not where planned</span>
            {/if}
          </div>
          {#if host && host.sockets > 1}
            <div class="ramtier" class:bad={host.problems.length > 0}>
              <strong>Host socket {host.node}</strong>
              <span class="muted">({host.auto ? host.reason : 'chosen'})</span> ·
              {host.threads} threads on {host.affinity.cpus} CPUs{host.affinity.on_node ? '' : ', not only this socket'} ·
              RAM tier {Object.entries(host.ram_tier_by_node).map(([n, b]) => `socket ${n} ${size(b)}`).join(', ') || 'empty'}
              {#each host.problems as pr (pr)}<div class="error">{pr}</div>{/each}
            </div>
          {/if}
          {#each (loaded.warnings ?? []).filter((w) => !(host?.problems ?? []).includes(w)) as w (w)}
            <div class="error">Warning: {w}</div>
          {/each}
          {#if loaded.stats?.ram_tier && loaded.stats.ram_tier.bytes > 0}
            {@const rt = loaded.stats.ram_tier}
            <div class="ramtier" class:bad={!rt.held_by_os}>
              <div>
                <strong>RAM tier</strong>
                {size(rt.resident)} in RAM{#if rt.swapped > 0}, <span class="error">{size(rt.swapped)} swapped out</span>{/if} ·
                {#if !rt.held}
                  <span class="error">not held: may be swapped out</span>
                {:else if rt.held_by_os}
                  <span class="ok">held</span>
                {:else}
                  <span class="error">held only by the GPU driver, no OS backstop</span>
                {/if}
              </div>
              <div class="mechs">
                {#each rt.mechanisms as m (m.name)}
                  <span class="mech" class:on={m.ok} title={m.detail}>{m.ok ? '✓' : '·'} {m.name}{m.layer !== 'os' ? ` (${m.layer})` : ''}</span>
                {/each}
              </div>
              {#if !rt.held_by_os}
                <ul class="fixes">{#each rt.fixes as f (f)}<li>{f}</li>{/each}</ul>
              {/if}
            </div>
          {/if}
          {#if loaded.stats}
            {@const st = loaded.stats}
            <div class="stats mono">
              <span>{st.requests} request{st.requests === 1 ? '' : 's'}</span>
              {#if st.requests > 0}
                <span>prompt {st.prefill_tps.toFixed(0)} tok/s</span>
                <span>generation {st.decode_tps.toFixed(1)} tok/s</span>
                <span class="muted">last: {st.last_prefill_tps.toFixed(0)} / {st.last_decode_tps.toFixed(1)} tok/s</span>
              {/if}
              {#if st.disk_tier_bytes > 0}
                <span title="Weights left in the mapped file are the kernel's page cache: they may be dropped from RAM and re-read.">
                  disk tier {((100 * st.disk_tier_resident) / st.disk_tier_bytes).toFixed(0)}% in RAM (the kernel's cache, may be dropped)
                </span>
              {/if}
            </div>
          {/if}
        </div>
      {:else}
        <div class="box muted">Not loaded. Applying a placement loads it.</div>
      {/if}
    </div>

    {#if loaded && rows.length}
      <h5>Memory per device</h5>
      <div class="scroll">
        <table class="mem">
          <thead>
            <tr><th>Device</th><th>Weights</th><th>KV cache</th><th>Compute scratch</th><th>Total</th><th>Free before load</th></tr>
          </thead>
          <tbody>
            {#each rows as r (r.box)}
              <tr>
                <td>{r.name}{r.disk ? ' (disk tier, in the model file)' : ''}</td>
                <td class="mono">{size(r.weights)}</td>
                <td class="mono">{size(r.kv)}</td>
                <td class="mono">{size(r.scratch)}</td>
                <td class="mono">{size(r.weights + r.kv + r.scratch)}</td>
                <td class="mono" class:error={r.free !== null && r.weights + r.kv + r.scratch > r.free}>{r.free !== null ? size(r.free) : ''}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
      <ul class="effects">
        <li>The RAM rows are what the worker measured on that socket: the RAM tier's weights, the KV cache and the compute scratch. The disk tier is the kernel's page cache and is not counted here.</li>
        {#if hostScratch && hostScratch.compute > (rows.find((r) => r.box.startsWith('ram'))?.scratch ?? 0) + (1 << 20)}
          <li>
            <strong>Compute scratch</strong>: llama.cpp reserves {size(hostScratch.compute)} on the host; {size(rows.filter((r) => r.box.startsWith('ram')).reduce((s, r) => s + r.scratch, 0))} of it is in RAM so far. The pages are counted once they are first used, so this grows after the first requests.
          </li>
        {/if}
        {#if offloadScratch}
          <li>
            <strong>GPU 0's compute scratch is larger than the others'</strong>: with weights in RAM or on disk, llama.cpp copies them to GPU 0 in pieces for big batches, and the staging space comes out of GPU 0.
          </li>
        {/if}
        {#if loaded.expert_cache.length > 0}
          <li>
            <strong>Expert cache</strong> (busy experts kept on a GPU, you set only its size): {loaded.expert_cache.map((c) => `GPU ${c.gpu} ${size(c.bytes)}`).join(', ')}. It follows the layer split: each GPU gets a share in proportion to the layers it holds.
          </li>
        {:else if (p.intent?.expert_cache_bytes ?? 0) > 0}
          <li><strong>Expert cache</strong> is set but not in use: llama.cpp only uses it when some layer has all its experts in RAM or on disk.</li>
        {/if}
        {#if loaded.stats && loaded.stats.disk_tier_bytes > 0}
          <li>
            <strong>Disk tier</strong>: {size(loaded.stats.disk_tier_bytes)} stays in the model file, and {size(loaded.stats.disk_tier_resident)} of it is in RAM right now, in the kernel's page cache. The system may drop it and read it again when needed.
          </li>
        {/if}
        {#if host && host.ram_tier_exact === false}
          <li>The RAM tier's size is estimated from a sample of its pages.</li>
        {/if}
      </ul>
    {/if}
  {/if}
</section>

<style>
  .details {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  header h3 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
  }
  .badge {
    width: 22px;
    height: 22px;
    border-radius: 6px;
    color: #10131a;
    font-weight: 700;
    display: grid;
    place-items: center;
    font-size: 13px;
  }
  h5 {
    margin: 4px 0 2px;
    font-size: 12px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr));
    gap: 12px;
  }
  .box {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    align-items: center;
  }
  .row input[type='number'] {
    width: 90px;
  }
  .note {
    font-size: 12.5px;
  }
  .ramtier {
    padding: 8px 10px;
    border: 1px solid var(--line);
    border-radius: 8px;
  }
  .ramtier.bad {
    border-color: var(--bad);
  }
  .mechs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 14px;
    margin-top: 6px;
    font-size: 12.5px;
    color: var(--muted);
  }
  .mech.on {
    color: var(--ok);
  }
  .fixes {
    margin: 8px 0 0;
    padding-left: 18px;
    font-size: 12.5px;
  }
  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
  }
  .scroll {
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  td,
  th {
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
  }
  th {
    color: var(--muted);
    font-weight: 500;
  }
  .mem td:not(:first-child),
  .mem th:not(:first-child) {
    text-align: right;
  }
  .effects {
    margin: 4px 0 0;
    padding-left: 18px;
    font-size: 12.5px;
    color: var(--muted);
  }
  .effects li {
    margin: 4px 0;
  }
</style>
