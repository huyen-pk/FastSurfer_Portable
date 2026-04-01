<script lang="ts">
  type TimelineEvent = {
    title: string;
    description: string;
    age: string;
    dotClass: string;
  };

  export let title = "Pipeline Events";
  export let icon = "history";
  export let events: TimelineEvent[];
</script>

<div class="glass-panel rounded-xl border border-outline-variant/10 p-6">
  <h2 class="mb-6 flex items-center gap-2 font-headline font-bold text-on-background">
    <span aria-hidden="true" class="material-symbols-outlined text-tertiary">{icon}</span>
    {title}
  </h2>

  <div class="space-y-6">
    {#each events as event}
      <div class="relative border-l border-outline-variant/20 pl-6">
        <div class={`absolute -left-1.5 top-0 h-3 w-3 rounded-full ring-4 ring-background ${event.dotClass}`}></div>
        <div class="space-y-1">
          <p class="text-sm font-semibold text-on-background">{event.title}</p>
          <p class="text-xs text-on-surface-variant">{event.description}</p>
          <span class="text-[10px] text-on-surface-variant/40">{event.age}</span>
        </div>
      </div>
    {/each}
  </div>
</div>
