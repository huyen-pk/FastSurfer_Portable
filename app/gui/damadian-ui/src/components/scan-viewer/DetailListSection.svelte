<script lang="ts">
  type DetailItem = {
    label: string;
    value: string;
  };

  export let title: string;
  export let items: DetailItem[];
</script>

<section class="space-y-3">
  <h3 class="px-1 text-xs font-bold uppercase tracking-widest text-on-surface-variant">{title}</h3>
  <div class="space-y-2">
    {#each items as item}
      <div class="rounded-lg border border-outline-variant/10 bg-surface-container-low p-3">
        <p class="mb-0.5 text-[9px] text-on-surface/40">{item.label}</p>
        <p class="text-xs font-medium text-on-surface">{item.value}</p>
      </div>
    {/each}
  </div>
</section>
