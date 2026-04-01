<script lang="ts">
  export let title: string;
  export let subtitle: string;
  export let icon: string;
  export let accentClass: string;
  export let iconClass: string;
  export let iconWrapperClass: string;
  export let badge = "";
  export let badgeClass = "";
  export let highlighted = false;
  export let showConnector = false;
</script>

<div class="group relative">
  <div class={`flex items-center gap-6 rounded-xl border-l-4 bg-surface-container p-6 shadow-xl ${accentClass} ${highlighted ? 'bg-surface-container-high' : ''}`}>
    <div class={`flex h-12 w-12 items-center justify-center rounded-lg ${iconWrapperClass} ${iconClass}`}>
      <span aria-hidden="true" class="material-symbols-outlined text-3xl">{icon}</span>
    </div>
    <div class="flex-1">
      <h2 class="font-bold text-on-background">{title}</h2>
      <p class="text-sm text-primary/60">{subtitle}</p>
    </div>
    {#if badge}
      <span class={`rounded px-2 py-1 text-[10px] font-bold uppercase tracking-widest ${badgeClass}`}>{badge}</span>
    {/if}
  </div>
  {#if showConnector}
    <div class="pipeline-connector"></div>
  {/if}
</div>
