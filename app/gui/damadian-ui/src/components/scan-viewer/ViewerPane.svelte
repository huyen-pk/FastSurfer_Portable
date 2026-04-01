<script lang="ts">
  export let imageSrc: string;
  export let imageAlt: string;
  export let label: string;
  export let sublabel = "";
  export let paneClass = "bg-surface-container-lowest";
  export let imageClass = "h-full w-full object-contain grayscale opacity-80 transition-opacity group-hover:opacity-100";
  export let accent = false;
</script>

<div class={`group relative overflow-hidden ${paneClass} ${accent ? 'border-2 border-tertiary/20' : ''}`}>
  {#if accent}
    <div class="pointer-events-none absolute inset-0 bg-gradient-to-br from-tertiary/5 to-transparent"></div>
  {/if}
  <img alt={imageAlt} class={imageClass} src={imageSrc} />
  <div class="pointer-events-none absolute left-3 top-3 flex flex-col">
    <span class={`text-[10px] font-bold tracking-tighter ${accent ? 'text-tertiary' : 'text-tertiary/60'}`}>{label}</span>
    {#if sublabel}
      <span class={`text-[9px] ${accent ? 'text-on-surface/60' : 'text-on-surface/40'}`}>{sublabel}</span>
    {/if}
  </div>
  <slot />
</div>
