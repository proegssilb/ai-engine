<script lang="ts">
  // The machine as boxes: GPUs, RAM (per NUMA node when there are several),
  // and the store's disk, each with what loaded models planned there.
  import RamBar from './RamBar.svelte'
  import { bytes, sockets, type Machine, type MemBox } from './api'

  let { machine, colors }: { machine: Machine; colors: Record<string, string> } = $props()

  interface Box {
    key: string
    title: string
    detail: string
    total: number
    used: number
    models: { model: string; bytes: number }[]
    /** shown as "runs" chips instead of byte chips (CPU boxes) */
    runs?: boolean
    /** the fuzzy RAM bar, for RAM boxes once the daemon has samples */
    mem?: MemBox
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
    // One RAM box, or one per NUMA node when there are several.
    const mem = machine.memory
    const socks = sockets(machine.numa)
    if (socks.length > 1) {
      // Per socket, as in the mockup: a CPU box (models whose threads run
      // there) and a RAM box (models' RAM tier, where it measurably is).
      socks.forEach((n) => {
        const mb = mem?.nodes.find((x) => x.node === n.node)
        out.push({
          key: `cpu${n.node}`,
          title: `CPU ${n.node}`,
          detail: `Compute only. Runs what sits in RAM ${n.node}. CPUs ${n.cpus}`,
          total: 0,
          used: 0,
          runs: true,
          models: machine.models
            .filter((m) => m.host?.node === n.node)
            .map((m) => ({ model: m.model, bytes: m.host?.affinity.cpus ?? 0 })),
        })
        out.push({
          key: `ram${n.node}`,
          title: `RAM ${n.node}`,
          detail: `Socket ${n.node}`,
          total: n.total,
          used: n.total - n.free,
          models: machine.models
            .map((m) => ({ model: m.model, bytes: m.host?.ram_tier_by_node?.[String(n.node)] ?? 0 }))
            .filter((m) => m.bytes > 0),
          mem: mb,
        })
      })
    } else {
      out.push({
        key: 'ram',
        title: 'RAM',
        detail: `${machine.numa.length} NUMA node${machine.numa.length === 1 ? '' : 's'}`,
        total: machine.ram.total,
        used: machine.ram.total - machine.ram.available,
        models: usage('ram'),
        mem: mem?.system,
      })
    }
    // one box per store: a model's disk tier is on the disk its store is on
    const disks = machine.disks?.length ? machine.disks : [machine.disk]
    disks.forEach((d, i) =>
      out.push({
        key: `disk:${d.path}`,
        title: disks.length > 1 ? `Disk ${i}` : 'Disk',
        detail: d.path,
        total: d.total,
        used: d.total - d.free,
        models: machine.models
          .filter((m) => (m.store ?? disks[0].path) === d.path)
          .map((m) => ({ model: m.model, bytes: m.bytes_by_tier['disk'] ?? 0 }))
          .filter((m) => m.bytes > 0),
      }),
    )
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
        {#if !b.runs}<span class="muted mono">{bytes(b.used)} / {bytes(b.total)}</span>{/if}
      </div>
      <div class="muted detail" title={b.detail}>{b.detail}</div>
      {#if b.runs}
        <!-- CPU boxes have no bar: they run what sits in their RAM box -->
      {:else if b.mem}
        <RamBar box={b.mem} {colors} />
      {:else}
        <div class="bar"><span style="width: {(100 * b.used) / Math.max(b.total, 1)}%; background: var(--accent)"></span></div>
      {/if}
      {#each b.models as m (m.model)}
        <div class="chip" style="border-color: {colors[m.model]}">
          <span class="swatch" style="background: {colors[m.model]}"></span>
          <span class="name" title={m.model}>{m.model.replace(/^hf\.co\//, '')}</span>
          <span class="mono">{b.runs ? 'runs' : bytes(m.bytes)}</span>
        </div>
      {/each}
      {#if b.runs && !b.models.length}<div class="muted small">No model's threads run here.</div>{/if}
      {#if b.key.startsWith('disk') && b.models.length}
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
