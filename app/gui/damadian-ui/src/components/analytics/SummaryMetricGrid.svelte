<script lang="ts">
  type SummaryMetric = {
    label: string;
    value: string;
    valueClass?: string;
  };

  export let metrics: SummaryMetric[];
</script>

<div class="col-span-12 flex flex-col justify-center rounded-xl border-r-4 border-tertiary bg-surface-container p-6 lg:col-span-6">
  <div class="grid grid-cols-1 gap-8 sm:grid-cols-2">
    {#each metrics as metric}
      <div>
        <div class="mb-1 text-[10px] font-bold uppercase text-on-surface-variant">{metric.label}</div>
        <div class={`font-headline text-2xl font-bold text-on-background ${metric.valueClass ?? ''}`}>{metric.value}</div>
      </div>
    {/each}
  </div>
</div>
