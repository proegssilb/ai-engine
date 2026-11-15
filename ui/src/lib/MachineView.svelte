<script lang="ts">
  // The machine as boxes: GPUs, RAM (per NUMA node when there are several),
  // and the store's disk, each with what loaded models planned there.
  import { bytes, type Machine } from './api'

  let { machine, colors }: { machine: Machine; colors: Record<string, string> } = $props()

  interface Box {
    key: string
    title: string
    detail: string
    total: number
    used: number
    models: { model: string; bytes: number }[]
  }

  let boxes = $derived.by((): Box[] => {
    const usage = (key: string) =>
      machine.models.map((m) => ({ model: m.model, bytes: m.bytes_by_tier[key] ?? 0 })).filter((m) => m.bytes > 0)
    const out: Box[] = []
    machine.devices
      .filter((d) => d.gpu)
      .forEach((d, i) =>
        out.push({ key: `gpu${i}`, title: `GPU ${i}`, detail: d.description, total: d.total, used: d.total - d.free, models: usage(`gpu${i}`) }),
      )
    const nodes = machine.numa.length > 1 ? machine.numa.map((n) => ` · node ${n.node}`) : ['']
    out.push({
      key: 'ram',
      title: 'RAM',
      detail: `${machine.numa.length} NUMA node${machine.numa.length === 1 ? '' : 's'}${nodes.length > 1 ? '' : ''}`,
      total: machine.ram.total,
      used: machine.ram.total - machine.ram.available,
      models: usage('ram'),
    })
    out.push({
      key: 'disk',
      title: 'Disk',
      detail: machine.disk.path,
      total: machine.disk.total,
      used: machine.disk.total - machine.disk.free,
      models: usage('disk'),
    })
    return out
  })

  // Box height grows a step per doubling of capacity, as in the mockup.
  const height = (total: number) => 90 + 18 * Math.max(0, Math.log2(Math.max(total, 1) / (8 << 30)))
</script>

<div class="machine">
  {#each boxes as b (b.key)}
    <div class="box panel" style="min-height: {height(b.total)}px">
      <div class="head">
        <h3>{b.title}</h3>
        <span class="muted mono">{bytes(b.used)} / {bytes(b.total)}</span>
      </div>
      <div class="muted detail" title={b.detail}>{b.detail}</div>
      <div class="bar"><span style="width: {(100 * b.used) / Math.max(b.total, 1)}%; background: var(--accent)"></span></div>
      {#each b.models as m (m.model)}
        <div class="chip" style="border-color: {colors[m.model]}">
          <span class="swatch" style="background: {colors[m.model]}"></span>
          <span class="name" title={m.model}>{m.model.replace(/^hf\.co\//, '')}</span>
          <span class="mono">{bytes(m.bytes)}</span>
        </div>
      {/each}
      {#if b.key === 'disk' && b.models.length}
        <div class="muted small">Disk tier: may be dropped from RAM, never forced out.</div>
      {/if}
    </div>
  {/each}
  {#if !machine.cuda_build}
    <div class="muted small">This build has no CUDA backend, so no GPUs are shown.</div>
  {/if}
</div>

<style>
  .machine {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 12px;
    align-items: start;
  }
  .head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .detail {
    font-size: 12px;
    margin: 4px 0 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .chip {
    display: flex;
    gap: 8px;
    align-items: center;
    border: 1px solid;
    border-radius: 7px;
    padding: 5px 8px;
    margin-top: 8px;
    background: var(--panel-2);
  }
  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    flex: none;
  }
  .name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .small {
    font-size: 12px;
    margin-top: 8px;
  }
</style>
