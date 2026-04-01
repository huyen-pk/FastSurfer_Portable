<script lang="ts">
  export let eyebrow = "";
  export let title: string;
  export let description = "";
</script>

<header class="mb-10">
  {#if eyebrow}
    <p class="mb-2 text-xs font-semibold uppercase tracking-[0.35em] text-primary/60">{eyebrow}</p>
  {/if}
  <h1 class="font-headline text-3xl font-extrabold tracking-tight text-on-background md:text-4xl">{title}</h1>
  {#if description}
    <p class="mt-1 text-on-surface-variant">{description}</p>
  {/if}
</header>
