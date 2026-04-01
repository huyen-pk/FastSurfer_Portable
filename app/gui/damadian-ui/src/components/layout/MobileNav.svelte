<script lang="ts">
  import type { NavigationItem, ViewId } from "./types";

  export let items: NavigationItem[];
  export let activeView: ViewId;
  export let searchPlaceholder: string;
  export let onSelect: (viewId: ViewId) => void;

  function getNavigationClass(viewId: ViewId): string {
    return activeView === viewId
      ? "bg-surface-container-high text-tertiary"
      : "bg-surface-container text-primary";
  }
</script>

<div class="sticky top-16 z-30 border-b border-outline-variant/10 bg-background/95 px-4 py-4 backdrop-blur lg:hidden">
  <div class="mb-4 flex items-center gap-3 rounded-lg bg-surface-container-highest px-3 py-2">
    <span aria-hidden="true" class="material-symbols-outlined text-sm text-primary">search</span>
    <input
      aria-label="Search workspace mobile"
      class="w-full border-none bg-transparent text-sm text-on-surface placeholder:text-on-surface-variant/50 focus:outline-none"
      placeholder={searchPlaceholder}
      type="text"
    />
  </div>

  <nav aria-label="Primary navigation mobile" class="grid grid-cols-2 gap-2 sm:grid-cols-4">
    {#each items as item}
      <button
        class={`flex items-center justify-center gap-2 rounded-lg px-3 py-3 text-sm font-medium transition-colors ${getNavigationClass(item.id)}`}
        type="button"
        on:click={() => onSelect(item.id)}
      >
        <span aria-hidden="true" class="material-symbols-outlined text-base">{item.icon}</span>
        <span>{item.mobileLabel}</span>
      </button>
    {/each}
  </nav>
</div>
