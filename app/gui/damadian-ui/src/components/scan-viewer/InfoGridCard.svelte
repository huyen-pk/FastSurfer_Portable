<script lang="ts">
  type InfoItem = {
    label: string;
    value: string;
  };

  export let title: string;
  export let items: InfoItem[];
</script>

<section class="rounded-xl border border-outline-variant/10 bg-surface-container-high p-5">
  <h3 class="mb-4 flex items-center gap-2 text-[10px] font-bold text-on-primary-container">
    <span aria-hidden="true" class="material-symbols-outlined text-sm">info</span>
    {title}
  </h3>

  <div class="grid grid-cols-2 gap-x-4 gap-y-6">
    {#each items as item}
      <div>
        <p class="mb-1 text-[9px] font-bold uppercase tracking-wider text-on-surface/40">{item.label}</p>
        <p class="font-mono text-[13px] text-on-surface">{item.value}</p>
      </div>
    {/each}
  </div>
</section>
