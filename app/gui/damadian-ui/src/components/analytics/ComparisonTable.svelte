<script lang="ts">
  type ComparisonRow = {
    studyId: string;
    metric: string;
    baseline: string;
    processed: string;
    variance: string;
    varianceClass: string;
    status: string;
    statusClass: string;
  };

  export let rows: ComparisonRow[];
</script>

<div class="col-span-12 overflow-hidden rounded-xl border border-outline-variant/5 bg-surface-container">
  <div class="flex items-center justify-between border-b border-outline-variant/10 px-6 py-5">
    <h2 class="text-sm font-bold tracking-tight text-on-background">Study Comparison Matrix</h2>
    <div class="flex gap-2">
      <button class="rounded px-3 py-1 text-xs font-bold text-primary transition-all hover:bg-surface-container-high" type="button">Export CSV</button>
      <button class="rounded px-3 py-1 text-xs font-bold text-primary transition-all hover:bg-surface-container-high" type="button">Filter</button>
    </div>
  </div>

  <div class="custom-scrollbar overflow-x-auto">
    <table class="w-full border-collapse text-left">
      <thead>
        <tr class="bg-surface-container-low/50">
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Study ID</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Metric</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Baseline Value</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Processed Value</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Variance</th>
          <th class="px-6 py-4 text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Status</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-outline-variant/5">
        {#each rows as row}
          <tr class="transition-colors hover:bg-surface-container-high/40">
            <td class="px-6 py-4 font-mono text-xs text-primary">{row.studyId}</td>
            <td class="px-6 py-4 text-sm font-medium">{row.metric}</td>
            <td class="px-6 py-4 text-sm text-on-surface-variant">{row.baseline}</td>
            <td class="px-6 py-4 text-sm font-bold">{row.processed}</td>
            <td class={`px-6 py-4 text-xs font-bold ${row.varianceClass}`}>{row.variance}</td>
            <td class="px-6 py-4">
              <span class={`rounded px-2 py-1 text-[10px] font-bold uppercase tracking-tighter ${row.statusClass}`}>{row.status}</span>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>
