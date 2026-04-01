<script lang="ts">
  import type { NavigationItem, ViewId } from "./types";

  export let items: NavigationItem[];
  export let activeView: ViewId;
  export let onSelect: (viewId: ViewId) => void;

  function getNavigationClass(viewId: ViewId): string {
    return activeView === viewId
      ? "bg-surface-container-high border-r-2 border-tertiary text-tertiary"
      : "text-primary hover:bg-surface-container-high/50 hover:text-on-background";
  }
</script>

<aside class="fixed left-0 top-0 z-40 hidden h-screen w-64 flex-col border-r border-outline-variant/10 bg-surface-container pt-16 lg:flex">
  <div class="px-6 py-8">
    <h2 class="font-headline text-lg font-bold tracking-tight text-on-background">Project Alpha</h2>
    <p class="text-xs font-medium tracking-wide text-on-surface-variant/60">Clinical Research</p>
  </div>

  <nav aria-label="Primary navigation" class="flex-1 space-y-1 px-4">
    {#each items as item}
      <button
        aria-current={activeView === item.id ? "page" : undefined}
        class={`flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left transition-all duration-300 ${getNavigationClass(item.id)}`}
        type="button"
        on:click={() => onSelect(item.id)}
      >
        <span aria-hidden="true" class="material-symbols-outlined">{item.icon}</span>
        <span>{item.label}</span>
      </button>
    {/each}
  </nav>

  <div class="px-4 py-6">
    <button class="backlit-button flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-bold shadow-lg shadow-tertiary/10 transition-all active:opacity-80" type="button" on:click={() => onSelect("pipelines")}>
      <span aria-hidden="true" class="material-symbols-outlined text-sm">add</span>
      <span>New Processing Task</span>
    </button>
  </div>

  <div class="mt-auto space-y-1 border-t border-outline-variant/10 p-4">
    <button class="flex w-full items-center gap-3 rounded-lg px-4 py-2 text-left text-primary transition-all hover:bg-surface-container-high/50 hover:text-on-background" type="button">
      <span aria-hidden="true" class="material-symbols-outlined text-sm">memory</span>
      <span class="text-xs">System Status</span>
    </button>
    <button class="flex w-full items-center gap-3 rounded-lg px-4 py-2 text-left text-primary transition-all hover:bg-surface-container-high/50 hover:text-on-background" type="button">
      <span aria-hidden="true" class="material-symbols-outlined text-sm">contact_support</span>
      <span class="text-xs">Support</span>
    </button>
  </div>
</aside>
