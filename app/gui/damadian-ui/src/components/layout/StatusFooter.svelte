<script lang="ts">
  import type { FooterState, ViewId } from "./types";

  export let activeView: ViewId;
  export let footerState: FooterState;
</script>

<footer class="fixed bottom-0 left-0 z-50 flex h-10 w-full items-center gap-4 overflow-x-auto border-t border-outline-variant/10 bg-surface-container-lowest px-4 shadow-[0_-10px_30px_rgba(0,0,0,0.4)] lg:pl-72">
  <button class={`flex items-center gap-2 text-[11px] font-medium tracking-wide ${activeView === 'pipelines' ? 'text-tertiary' : 'text-primary/60 hover:text-primary'}`} type="button">
    <span aria-hidden="true" class="material-symbols-outlined text-sm">hourglass_empty</span>
    <span>{footerState.queueLabel}</span>
  </button>
  <button class="flex items-center gap-2 text-[11px] font-medium tracking-wide text-tertiary" type="button">
    <span aria-hidden="true" class="material-symbols-outlined text-sm">running_with_errors</span>
    <span>{footerState.activeLabel}</span>
  </button>
  <button class={`flex items-center gap-2 text-[11px] font-medium tracking-wide ${activeView === 'analytics' ? 'text-tertiary' : 'text-primary/60 hover:text-primary'}`} type="button">
    <span aria-hidden="true" class="material-symbols-outlined text-sm">terminal</span>
    <span>{footerState.logsLabel}</span>
  </button>

  <div class="ml-auto hidden items-center gap-4 md:flex">
    {#if activeView === "pipelines"}
      <div class="flex items-center gap-2">
        <div class={`h-2 w-2 animate-pulse rounded-full ${footerState.indicatorClass}`}></div>
        <span class="font-mono text-[10px] text-primary/70">{footerState.rightLabel}</span>
      </div>
    {:else}
      <div class="h-1.5 w-24 overflow-hidden rounded-full bg-surface-container-highest">
        <div class={`h-full ${footerState.indicatorClass}`} style={`width: ${footerState.meterWidth}`}></div>
      </div>
      <span class="font-mono text-[10px] text-primary/70">{footerState.rightLabel}</span>
    {/if}
  </div>
</footer>
