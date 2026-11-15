<script lang="ts">
  // One model's placement: automatic (llama.cpp's fit) or shares per tier
  // for its layers and its biggest per-layer subsystem (experts or FFN),
  // then apply, reload, and show where every weight really is.
  import SplitEditor from './SplitEditor.svelte'
  import { api, bytes, tierFromKey, tierKey, tierLabel, type Intent, type Placement, type Share } from './api'

  let { model, gpus, onchange }: { model: string; gpus: number; onchange: () => void } = $props()

  let p = $state<Placement | null>(null)
  let error = $state('')
  let busy = $state('')
  let mode = $state<'auto' | 'custom'>('auto')
  let layerPct = $state<Record<string, number>>({})
  let bulkPct = $state<Record<string, number>>({})
  let cacheGiB = $state(0)
  let nCtx = $state(4096)

  const gpuKeys = $derived(Array.from({ length: gpus }, (_, i) => `gpu${i}`))
  const layerTiers = $derived([...gpuKeys, 'ram'])
  const bulkTiers = $derived([...gpuKeys, 'ram', 'disk'])
  const bulkKey = $derived(p?.anatomy.moe ? 'RoutedExperts' : 'DenseFfn')
  const bulkLabel = $derived(p?.anatomy.moe ? 'routed experts' : 'dense ffn')
  // what follows a layer's home: everything per-layer except the split subsystem
  const layerBytes = $derived(
    Object.entries(p?.anatomy.subsystems ?? {})
      .filter(([k]) => !['embeddings', 'output head', 'other', bulkLabel].includes(k))
      .reduce((s, [, b]) => s + b, 0),
  )

  function toPercents(shares: Share[], keys: string[], total: number): Record<string, number> {
    const out: Record<string, number> = Object.fromEntries(keys.map((k) => [k, 0]))
    let rest: string | null = null
    for (const s of shares) {
      const k = tierKey(s.tier)
      if (s.amount === 'Rest') rest = k
      else if ('Fraction' in s.amount) out[k] = Math.round(s.amount.Fraction * 100)
      else out[k] = Math.round((s.amount.Bytes / Math.max(total, 1)) * 100)
    }
    if (rest) out[rest] = Math.max(0, 100 - Object.values(out).reduce((a, b) => a + b, 0))
    return out
  }

  function toShares(pct: Record<string, number>, keys: string[]): Share[] {
    const used = keys.filter((k) => (pct[k] ?? 0) > 0)
    return used.map((k, i) => ({
      tier: tierFromKey(k),
      amount: i === used.length - 1 ? ('Rest' as const) : { Fraction: (pct[k] ?? 0) / 100 },
    }))
  }

  function defaults() {
    // everything on the first GPU if there is one, else RAM
    const home = gpus > 0 ? 'gpu0' : 'ram'
    layerPct = Object.fromEntries(layerTiers.map((k) => [k, k === home ? 100 : 0]))
    bulkPct = Object.fromEntries(bulkTiers.map((k) => [k, k === home ? 100 : 0]))
  }

  async function refresh() {
    error = ''
    try {
      p = await api.placement(model)
      const i = p.intent
      if (i) {
        mode = 'custom'
        nCtx = i.n_ctx
        cacheGiB = i.expert_cache_bytes / (1 << 30)
        layerPct = toPercents(i.layers, layerTiers, p.anatomy.layers)
        const bulkTotal = p.anatomy.subsystems[bulkLabel] ?? 0
        bulkPct = i.subsystems[bulkKey] ? toPercents(i.subsystems[bulkKey], bulkTiers, bulkTotal) : { ...layerPct, disk: 0 }
      } else {
        mode = 'auto'
        defaults()
      }
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
    }
  }

  $effect(() => {
    void model
    refresh()
  })

  const pctOk = (pct: Record<string, number>, keys: string[]) => Math.round(keys.reduce((s, k) => s + (pct[k] ?? 0), 0)) === 100

  async function apply() {
    if (!p) return
    error = ''
    let intent: Intent | null = null
    if (mode === 'custom') {
      if (!pctOk(layerPct, layerTiers) || !pctOk(bulkPct, bulkTiers)) {
        error = 'Each split has to add up to 100%.'
        return
      }
      intent = {
        n_ctx: nCtx,
        layers: toShares(layerPct, layerTiers),
        subsystems: { [bulkKey]: toShares(bulkPct, bulkTiers) },
        embeddings: null,
        output: null,
        expert_cache_bytes: Math.round(cacheGiB * (1 << 30)),
        allow_repack: false,
      }
    }
    try {
      busy = 'Saving…'
      await api.savePlacement(model, intent, true)
      busy = 'Loading and checking every weight…'
      await api.load(model)
      await refresh()
      onchange()
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
    } finally {
      busy = ''
    }
  }

  async function unload() {
    busy = 'Unloading…'
    try {
      await api.unload(model)
      await refresh()
      onchange()
    } finally {
      busy = ''
    }
  }
</script>

{#if p}
  <div class="panel editor">
    <div class="title">
      <div>
        <h3>{p.model}</h3>
        <div class="muted">
          {p.anatomy.architecture} · {p.anatomy.layers} layers · {bytes(p.anatomy.bytes)}
          {#if p.anatomy.moe}· {p.anatomy.moe.experts} experts, {p.anatomy.moe.experts_used} active{/if}
          {#if !p.anatomy.supported}· <span class="error">architecture not yet checked</span>{/if}
        </div>
      </div>
      <div class="actions">
        {#if p.loaded}<button onclick={unload} disabled={!!busy}>Unload</button>{/if}
      </div>
    </div>

    <div class="modes">
      <label><input type="radio" bind:group={mode} value="auto" /> Automatic <span class="muted">(llama.cpp decides, by free memory)</span></label>
      <label><input type="radio" bind:group={mode} value="custom" /> Custom split</label>
    </div>

    {#if mode === 'custom'}
      <section>
        <h2>Layers</h2>
        <SplitEditor
          tiers={layerTiers}
          total={layerBytes}
          bind:percents={layerPct}
          note="Attention, norms, router and the KV cache live with their layer. Layers in RAM come first, then each GPU in order."
        />
      </section>
      <section>
        <h2>{p.anatomy.moe ? 'Expert bank' : 'Feed-forward'}</h2>
        <SplitEditor
          tiers={bulkTiers}
          total={p.anatomy.subsystems[bulkLabel] ?? 0}
          bind:percents={bulkPct}
          note="Whole layers at a time, so shares are approximate. GPU shares prefer layers that already live on that GPU."
        />
      </section>
      <section class="row">
        <label>Context <input type="number" min="512" step="512" bind:value={nCtx} /> tokens</label>
        {#if p.anatomy.moe}
          <label>Expert cache <input type="number" min="0" step="0.5" bind:value={cacheGiB} /> GiB on GPU
            <span class="muted">(llama.cpp keeps busy experts there)</span></label>
        {/if}
      </section>
    {/if}

    <div class="apply">
      <button class="primary" onclick={apply} disabled={!!busy}>Apply placement</button>
      {#if busy}<span class="muted">{busy}</span>{/if}
    </div>
    {#if error}<div class="error">{error}</div>{/if}

    {#if p.loaded}
      <section>
        <h2>Loaded: where the weights are</h2>
        <div class="muted">
          {p.loaded.description} · context {p.loaded.n_ctx} · loaded in {(p.loaded.load_ms / 1000).toFixed(1)} s ·
          {#if p.loaded.misplaced === 0}
            <span class="ok">every weight verified where planned</span>
          {:else}
            <span class="error">{p.loaded.misplaced} weights not where planned</span>
          {/if}
        </div>
        {#each p.loaded.warnings ?? [] as w (w)}
          <div class="error">Warning: {w}</div>
        {/each}
        {#if p.loaded.stats}
          {@const st = p.loaded.stats}
          <div class="stats mono">
            <span>{st.requests} request{st.requests === 1 ? '' : 's'}</span>
            {#if st.requests > 0}
              <span>prompt {st.prefill_tps.toFixed(0)} tok/s</span>
              <span>generation {st.decode_tps.toFixed(1)} tok/s</span>
              <span class="muted">last: {st.last_prefill_tps.toFixed(0)} / {st.last_decode_tps.toFixed(1)} tok/s</span>
            {/if}
            {#if st.disk_tier_bytes > 0}
              <span title="Weights left in the mapped file may be dropped from RAM and re-read; this is how much is in RAM right now.">
                disk tier {((100 * st.disk_tier_resident) / st.disk_tier_bytes).toFixed(0)}% in RAM
              </span>
            {/if}
          </div>
        {/if}
        <table>
          <thead><tr><th>Subsystem</th><th>Where</th><th></th></tr></thead>
          <tbody>
            {#each Object.entries(p.loaded.planned) as [sub, tiers] (sub)}
              {@const total = Object.values(tiers).reduce((a, b) => a + b, 0)}
              <tr>
                <td>{sub}</td>
                <td>
                  <div class="bar wide">
                    {#each Object.entries(tiers) as [t, b] (t)}
                      <span style="width: {(100 * b) / Math.max(total, 1)}%; background: var(--{t.startsWith('gpu') ? 'gpu' : t})"></span>
                    {/each}
                  </div>
                </td>
                <td class="mono">{Object.entries(tiers).map(([t, b]) => `${tierLabel(t)} ${bytes(b)}`).join(' · ')}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </section>
    {:else}
      <div class="muted">Not loaded. Applying a placement loads it.</div>
    {/if}
  </div>
{:else if error}
  <div class="error">{error}</div>
{/if}

<style>
  .editor {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .title {
    display: flex;
    justify-content: space-between;
    gap: 12px;
  }
  .modes {
    display: flex;
    gap: 24px;
  }
  section h2 {
    margin-bottom: 8px;
  }
  .row {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
  }
  .row input {
    width: 90px;
  }
  .apply {
    display: flex;
    gap: 12px;
    align-items: center;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
  }
  td,
  th {
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid var(--line);
  }
  th {
    color: var(--muted);
    font-weight: 500;
  }
  .wide {
    width: 180px;
  }
  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
    margin-top: 8px;
  }
</style>
