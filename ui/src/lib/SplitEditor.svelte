<script lang="ts">
  // A subsystem split across tiers by share (decision D6), drawn like the
  // mockup's expert bank: one segment per tier. The last tier with a share
  // takes whatever the others leave.
  import { bytes, tierLabel } from './api'

  let {
    tiers,
    total,
    percents = $bindable(),
    note = '',
  }: { tiers: string[]; total: number; percents: Record<string, number>; note?: string } = $props()

  const color = (k: string) => (k.startsWith('gpu') ? 'var(--gpu)' : k === 'ram' ? 'var(--ram)' : 'var(--disk)')
  let sum = $derived(tiers.reduce((s, k) => s + (percents[k] ?? 0), 0))
</script>

<div class="split">
  <div class="segments">
    {#each tiers as k (k)}
      {#if (percents[k] ?? 0) > 0}
        <span style="flex: {percents[k]}; background: {color(k)}" title="{tierLabel(k)} {percents[k]}%">
          {percents[k]}%
        </span>
      {/if}
    {/each}
  </div>
  <div class="inputs">
    {#each tiers as k (k)}
      <label>
        <span class="dot" style="background: {color(k)}"></span>
        {tierLabel(k)}
        <input type="number" min="0" max="100" step="5" bind:value={percents[k]} />%
        <span class="muted mono">{bytes(((percents[k] ?? 0) / 100) * total)}</span>
      </label>
    {/each}
  </div>
  {#if Math.round(sum) !== 100}
    <div class="error">Shares add up to {Math.round(sum)}%; they need to make 100%.</div>
  {/if}
  {#if note}<div class="muted note">{note}</div>{/if}
</div>

<style>
  .segments {
    display: flex;
    height: 26px;
    border-radius: 6px;
    overflow: hidden;
    background: var(--bg);
    margin-bottom: 8px;
  }
  .segments span {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #10131a;
    font-weight: 600;
    font-size: 12px;
    min-width: 28px;
  }
  .inputs {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
  }
  label {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  input {
    width: 64px;
  }
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 3px;
  }
  .note {
    margin-top: 6px;
    font-size: 12.5px;
  }
</style>
