<script lang="ts">
  type Study = {
    patientId: string;
    scanName: string;
    protocol: string;
    status: string;
    statusIcon: string;
    statusClass: string;
    time: string;
  };

  export let studies: Study[];
</script>

<div class="col-span-12 overflow-hidden rounded-xl bg-surface-container-low lg:col-span-8">
  <div class="flex items-center justify-between bg-surface-container p-6">
    <h2 class="font-headline font-bold text-on-background">Recent MRI Studies</h2>
    <button class="flex items-center gap-1 text-sm font-medium text-primary hover:underline" type="button">
      View All
      <span aria-hidden="true" class="material-symbols-outlined text-sm">arrow_forward</span>
    </button>
  </div>

  <div class="custom-scrollbar overflow-x-auto">
    <table class="w-full text-left">
      <thead>
        <tr class="border-b border-outline-variant/10">
          <th class="px-6 py-4 text-xs font-semibold uppercase tracking-widest text-on-surface-variant/50">Patient ID</th>
          <th class="px-6 py-4 text-xs font-semibold uppercase tracking-widest text-on-surface-variant/50">Protocol</th>
          <th class="px-6 py-4 text-xs font-semibold uppercase tracking-widest text-on-surface-variant/50">Status</th>
          <th class="px-6 py-4 text-xs font-semibold uppercase tracking-widest text-on-surface-variant/50">Time</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-outline-variant/10">
        {#each studies as study}
          <tr class="transition-colors hover:bg-surface-container-high">
            <td class="px-6 py-5">
              <div class="flex flex-col">
                <span class="font-semibold text-on-background">{study.patientId}</span>
                <span class="text-xs text-on-surface-variant">{study.scanName}</span>
              </div>
            </td>
            <td class="px-6 py-5">
              <span class="rounded bg-secondary-container px-2.5 py-1 text-[10px] font-bold tracking-wider text-on-secondary-container">{study.protocol}</span>
            </td>
            <td class="px-6 py-5">
              <div class="flex items-center gap-2">
                {#if study.statusIcon === "pulse"}
                  <div class="h-2 w-2 animate-pulse rounded-full bg-tertiary"></div>
                {:else}
                  <span aria-hidden="true" class={`material-symbols-outlined text-sm ${study.statusClass}`}>{study.statusIcon}</span>
                {/if}
                <span class={`text-sm ${study.statusClass}`}>{study.status}</span>
              </div>
            </td>
            <td class="px-6 py-5 text-sm text-on-surface-variant">{study.time}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>
