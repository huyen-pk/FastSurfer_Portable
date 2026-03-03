<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { getCurrentWindow } from "@tauri-apps/api/window";
  import TaskManagerPanel from "./components/TaskManagerPanel.svelte";

  let unlistenCloseRequested: (() => void) | null = null;
  let taskManagerPanel: { maybeConfirmClose: (event: { preventDefault: () => void }) => Promise<void> } | null = null;

  function isTauriRuntime(): boolean {
    return typeof window !== "undefined" && "__TAURI__" in window;
  }

  onDestroy(() => {
    if (unlistenCloseRequested) {
      unlistenCloseRequested();
      unlistenCloseRequested = null;
    }
  });

  onMount(async () => {
    if (!isTauriRuntime()) return;

    const window = getCurrentWindow();
    unlistenCloseRequested = await window.onCloseRequested((event) => {
      if (!taskManagerPanel) {
        return;
      }
      void taskManagerPanel.maybeConfirmClose(event);
    });
  });
</script>

<TaskManagerPanel bind:this={taskManagerPanel} />
