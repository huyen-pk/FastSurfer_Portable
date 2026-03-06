<script lang="ts">
  type TaskStatus = "waiting" | "processing" | "completed" | "failed" | "cancelled";

  interface SidebarTaskItem {
    id: string;
    status: TaskStatus;
    selectedPaths: string[];
    total: number;
    completed: number;
    progress: number;
    message: string;
    currentPath: string;
  }

  export let task: SidebarTaskItem;
  export let title: string;
  export let variant = "";
  export let summary: string;
  export let showProgress = false;
  export let showMessage = false;
  export let showCurrentPath = false;
  export let actionLabel: string;
  export let actionDanger = false;
  export let onAction: (taskId: string) => void;
</script>

<li class={`task ${variant}`}>
  <div class="task-title">{title}</div>
  <div class="task-meta">{summary}</div>
  {#if showProgress}
    <progress max="100" value={task.progress}></progress>
  {/if}
  {#if showMessage}
    <div class="task-message">{task.message}</div>
  {/if}
  {#if showCurrentPath && task.currentPath}
    <div class="task-path" title={task.currentPath}>{task.currentPath}</div>
  {/if}
  <details class="task-paths">
    <summary>Added paths ({task.selectedPaths.length})</summary>
    <ul>
      {#each task.selectedPaths as path}
        <li title={path}>{path}</li>
      {/each}
    </ul>
  </details>
  <button
    type="button"
    class="task-action"
    class:danger={actionDanger}
    on:click={() => onAction(task.id)}
  >
    {actionLabel}
  </button>
</li>

<style>
  .task {
    border: 1px solid var(--border-color, #d8d8d8);
    border-radius: 0.5rem;
    padding: 0.6rem;
    background: var(--surface, #fff);
  }

  .task-title {
    font-weight: 600;
  }

  .task-meta,
  .task-message,
  .task-path {
    margin-top: 0.25rem;
    font-size: 0.85rem;
    word-break: break-all;
  }

  .task progress {
    margin-top: 0.4rem;
    width: 100%;
  }

  .task--waiting {
    opacity: 0.9;
  }

  .task--completed {
    opacity: 0.95;
  }

  .task--failed {
    border-color: #b00020;
  }

  .task--cancelled {
    opacity: 0.85;
  }

  .task-action {
    margin-top: 0.4rem;
  }

  .task-paths {
    margin-top: 0.4rem;
    font-size: 0.82rem;
  }

  .task-paths ul {
    margin: 0.3rem 0 0;
    padding-left: 1rem;
  }

  .task-paths li {
    word-break: break-all;
  }

  .task-action.danger {
    color: #b00020;
  }
</style>
