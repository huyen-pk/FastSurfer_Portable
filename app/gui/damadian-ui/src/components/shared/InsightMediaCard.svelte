<script lang="ts">
  export let imageSrc: string;
  export let imageAlt: string;
  export let badge = "";
  export let title: string;
  export let description: string;
  export let aspectClass = "aspect-video";
  export let imageClass = "h-full w-full object-cover transition-transform duration-500 group-hover:scale-110";
  export let containerClass = "";
  export let overlayClass = "absolute inset-0 flex flex-col justify-end bg-gradient-to-t from-surface-container-lowest to-transparent p-6";
</script>

<div class={`group relative overflow-hidden rounded-xl ${aspectClass} ${containerClass}`}>
  <img alt={imageAlt} class={imageClass} src={imageSrc} />
  <div class={overlayClass}>
    {#if $$slots.default}
      <slot />
    {:else}
      {#if badge}
        <span class="mb-2 w-fit rounded bg-tertiary px-2 py-0.5 text-[10px] font-bold text-on-tertiary">{badge}</span>
      {/if}
      <h3 class="font-headline font-bold text-on-background">{title}</h3>
      <p class="text-xs text-on-surface-variant">{description}</p>
    {/if}
  </div>
</div>
