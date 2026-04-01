<script lang="ts">
  type BarDatum = {
    label: string;
    baseline: number;
    current: number;
  };

  export let bars: BarDatum[];
</script>

<div class="col-span-12 rounded-xl border-l border-tertiary/20 bg-surface-container p-6 lg:col-span-8">
  <div class="mb-8 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
    <h2 class="text-sm font-bold uppercase tracking-widest text-primary">Volumetric Delta (cm3)</h2>
    <div class="flex gap-4 text-xs">
      <div class="flex items-center gap-2">
        <span class="h-3 w-3 rounded-full bg-tertiary"></span>
        <span>Baseline Study</span>
      </div>
      <div class="flex items-center gap-2">
        <span class="h-3 w-3 rounded-full bg-primary-container"></span>
        <span>Current Post-OP</span>
      </div>
    </div>
  </div>

  <div class="flex h-64 items-end justify-between gap-4 px-2 md:gap-6 md:px-4">
    {#each bars as bar}
      <div class="group flex flex-1 flex-col items-center justify-end">
        <div class="flex h-full w-full items-end justify-center gap-1">
          <div class="w-full rounded-t-sm bg-tertiary/40 transition-all group-hover:bg-tertiary/60" style={`height: ${bar.baseline}%`}></div>
          <div class="w-full rounded-t-sm bg-tertiary shadow-[0_0_15px_rgba(0,218,243,0.3)]" style={`height: ${bar.current}%`}></div>
        </div>
        <span class="mt-3 text-[10px] font-medium text-on-surface-variant group-hover:text-tertiary">{bar.label}</span>
      </div>
    {/each}
  </div>
</div>
