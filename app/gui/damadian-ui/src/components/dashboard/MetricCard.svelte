<script lang="ts">
  export let icon: string;
  export let iconWrapperClass: string;
  export let iconClass: string;
  export let label: string;
  export let value: string;
  export let description: string;
  export let descriptionClass: string;
</script>

<div class="col-span-12 rounded-xl border border-outline-variant/5 bg-surface-container p-6 md:col-span-4">
  <div class="mb-4 flex items-start justify-between">
    <span class={`rounded-lg p-2 ${iconWrapperClass}`}>
      <span aria-hidden="true" class={`material-symbols-outlined ${iconClass}`}>{icon}</span>
    </span>
    <span class="text-xs font-medium uppercase tracking-widest text-on-surface-variant">{label}</span>
  </div>
  <div class="space-y-1">
    <h2 class="font-headline text-4xl font-extrabold text-on-background">{value}</h2>
    <p class={`text-sm ${descriptionClass}`}>{description}</p>
  </div>
</div>
