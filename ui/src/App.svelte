<script lang="ts">
  import MachineView from './lib/MachineView.svelte'
  import PlacementEditor from './lib/PlacementEditor.svelte'
  import { api, bytes, type Machine, type ModelTag, type Running } from './lib/api'

  let models = $state<ModelTag[]>([])
  let running = $state<Running[]>([])
  let machine = $state<Machine | null>(null)
  let selected = $state<string | null>(null)
  let error = $state('')
  let pullName = $state('')
  let pullStatus = $state('')

  const palette = ['#5b9cff', '#f2a43a', '#b48ead', '#57c785', '#ff6b6b', '#4fd1c5']
  let colors = $derived(Object.fromEntries(models.map((m, i) => [m.name, palette[i % palette.length]])))

  async function refresh() {
    try {
      ;[models, running, machine] = await Promise.all([api.tags(), api.ps(), api.machine()])
      error = ''
      if (!selected && models.length) selected = models[0].name
    } catch (e) {
      error = String(e instanceof Error ? e.message : e)
    }
  }

  async function pull() {
    const name = pullName.trim()
    if (!name) return
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
    if (!confirm(`Delete ${name} from the store?`)) return
    await api.remove(name)
    if (selected === name) selected = null
    await refresh()
  }

  $effect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  })

  const isRunning = (name: string) => running.some((r) => r.name === name)
  let gpuCount = $derived(machine ? machine.devices.filter((d) => d.gpu).length : 0)
</script>

<header>
  <div>
    <h1>Placement</h1>
    {#if machine}
      <div class="muted mono">
        {gpuCount} GPU{gpuCount === 1 ? '' : 's'} · {bytes(machine.ram.total)} RAM · {machine.numa.length} NUMA node{machine.numa.length === 1 ? '' : 's'} · {bytes(machine.disk.total)} disk
      </div>
    {/if}
  </div>
  {#if error}<div class="error">{error}</div>{/if}
</header>

<main>
  <aside>
    <h2>Models</h2>
    {#each models as m (m.name)}
      <button class="model" class:selected={selected === m.name} onclick={() => (selected = m.name)}>
        <span class="swatch" style="background: {colors[m.name]}"></span>
        <span class="info">
          <span class="name" title={m.name}>{m.name.replace(/^hf\.co\//, '')}</span>
          <span class="muted">{m.details.family} · {m.details.parameter_size} · {m.details.quantization_level} · {bytes(m.size)}</span>
        </span>
        {#if isRunning(m.name)}<span class="ok dot" title="loaded">●</span>{/if}
      </button>
    {/each}
    {#if !models.length}<div class="muted">No models yet. Pull one below.</div>{/if}

    <div class="pull">
      <input placeholder="hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M" bind:value={pullName} onkeydown={(e) => e.key === 'Enter' && pull()} />
      <button onclick={pull} disabled={!!pullStatus && !pullStatus.startsWith('failed')}>Pull</button>
    </div>
    {#if pullStatus}<div class="muted small">{pullStatus}</div>{/if}
    {#if selected}
      <button class="link" onclick={() => selected && remove(selected)}>Delete selected model</button>
    {/if}
  </aside>

  <section class="content">
    <h2>Machine</h2>
    {#if machine}<MachineView {machine} {colors} />{/if}

    {#if selected}
      <h2>Placement</h2>
      <PlacementEditor model={selected} gpus={gpuCount} onchange={refresh} />
    {/if}
  </section>
</main>

<style>
  header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 20px 24px 8px;
    gap: 16px;
  }
  main {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 20px;
    padding: 12px 24px 32px;
  }
  aside,
  .content {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .model {
    display: flex;
    align-items: center;
    gap: 10px;
    text-align: left;
    background: var(--panel);
    padding: 10px 12px;
  }
  .model.selected {
    border-color: var(--text);
  }
  .info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
    flex: 1;
  }
  .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .info .muted {
    font-size: 12px;
  }
  .swatch {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    flex: none;
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
  @media (max-width: 800px) {
    main {
      grid-template-columns: 1fr;
    }
  }
</style>
