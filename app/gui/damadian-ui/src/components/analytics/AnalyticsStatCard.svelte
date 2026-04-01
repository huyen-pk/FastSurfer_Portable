<script lang="ts">
  export let label: string;
  export let value: string;
  export let description = "";
  export let valueClass = "text-primary";
  export let cardClass = "bg-surface-container-highest border border-outline-variant/10";
  export let badge = "";
  export let badgeCaption = "";
  export let icon = "";
  export let iconClass = "text-tertiary";
</script>

<div class={`relative overflow-hidden rounded-xl p-6 ${cardClass}`}>
  <div class="relative z-10">
    <h2 class="mb-4 text-xs font-bold uppercase text-on-surface-variant/70">{label}</h2>
    <div class={`font-headline text-4xl font-extrabold ${valueClass}`}>{value}</div>
    {#if description}
      <p class="mt-2 text-[11px] text-on-surface-variant">{description}</p>
    {/if}
    {#if badge}
      <div class="mt-4 flex items-center gap-2">
        <span class="rounded-full bg-tertiary/20 px-2 py-0.5 text-xs font-bold text-tertiary">{badge}</span>
        <span class="text-[10px] text-on-surface-variant">{badgeCaption}</span>
      </div>
    {/if}
  </div>
  {#if icon}
    <div class="absolute -bottom-4 -right-4 opacity-10">
      <span aria-hidden="true" class={`material-symbols-outlined text-[8rem] ${iconClass}`}>{icon}</span>
    </div>
  {/if}
</div>
