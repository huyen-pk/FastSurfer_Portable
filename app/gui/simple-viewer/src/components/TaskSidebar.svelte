<script lang="ts">
  type TaskStatus = "waiting" | "processing" | "completed" | "failed" | "cancelled";

  export interface SidebarTask {
    id: string;
    status: TaskStatus;
    selectedPaths: string[];
    total: number;
    completed: number;
    progress: number;
    message: string;
    currentPath: string;
  }

  export let isSidebarCollapsed = false;
  export let activeTask: SidebarTask | null = null;
  export let waitingTasks: SidebarTask[] = [];
  export let completedTasks: SidebarTask[] = [];
  export let failedTasks: SidebarTask[] = [];
  export let cancelledTasks: SidebarTask[] = [];
  export let formatTaskSummary: (task: SidebarTask) => string;
  export let onToggleSidebar: () => void;
  export let onCancelTask: (taskId: string) => void;
  export let onRemoveTask: (taskId: string) => void;
</script>

<aside class="dock-menu" aria-label="Task queue" class:collapsed={isSidebarCollapsed}>
  <button class="sidebar-toggle" type="button" on:click={onToggleSidebar}>
    {isSidebarCollapsed ? "Show Tasks" : "Hide Tasks"}
  </button>

  {#if !isSidebarCollapsed}
    <h2>Tasks</h2>
    <ul>
      {#if activeTask}
        <li class="task task--processing">
          <div class="task-title">Processing</div>
          <div class="task-meta">{formatTaskSummary(activeTask)}</div>
          <progress max="100" value={activeTask.progress}></progress>
          <div class="task-message">{activeTask.message}</div>
          {#if activeTask.currentPath}
            <div class="task-path" title={activeTask.currentPath}>{activeTask.currentPath}</div>
          {/if}
          <details class="task-paths">
            <summary>Added paths ({activeTask.selectedPaths.length})</summary>
            <ul>
              {#each activeTask.selectedPaths as path}
                <li title={path}>{path}</li>
              {/each}
            </ul>
          </details>
          <button type="button" class="task-action danger" on:click={() => onCancelTask(activeTask.id)}>Cancel & Remove</button>
        </li>
      {/if}

      {#if waitingTasks.length > 0}
        {#each waitingTasks as waitingTask}
          <li class="task task--waiting">
            <div class="task-title">Waiting</div>
            <div class="task-meta">{formatTaskSummary(waitingTask)}</div>
            <details class="task-paths">
              <summary>Added paths ({waitingTask.selectedPaths.length})</summary>
              <ul>
                {#each waitingTask.selectedPaths as path}
                  <li title={path}>{path}</li>
                {/each}
              </ul>
            </details>
            <button type="button" class="task-action" on:click={() => onCancelTask(waitingTask.id)}>Remove</button>
          </li>
        {/each}
      {:else if !activeTask}
        <li class="task task--empty">No queued task</li>
      {/if}

      {#if failedTasks.length > 0}
        {#each failedTasks as failedTask}
          <li class="task task--failed">
            <div class="task-title">Failed</div>
            <div class="task-meta">{formatTaskSummary(failedTask)}</div>
            <div class="task-message">{failedTask.message}</div>
            <details class="task-paths">
              <summary>Added paths ({failedTask.selectedPaths.length})</summary>
              <ul>
                {#each failedTask.selectedPaths as path}
                  <li title={path}>{path}</li>
                {/each}
              </ul>
            </details>
            <button type="button" class="task-action" on:click={() => onRemoveTask(failedTask.id)}>Remove</button>
          </li>
        {/each}
      {/if}

      {#if cancelledTasks.length > 0}
        {#each cancelledTasks as cancelledTask}
          <li class="task task--cancelled">
            <div class="task-title">Cancelled</div>
            <div class="task-meta">{formatTaskSummary(cancelledTask)}</div>
            <div class="task-message">{cancelledTask.message}</div>
            <details class="task-paths">
              <summary>Added paths ({cancelledTask.selectedPaths.length})</summary>
              <ul>
                {#each cancelledTask.selectedPaths as path}
                  <li title={path}>{path}</li>
                {/each}
              </ul>
            </details>
            <button type="button" class="task-action" on:click={() => onRemoveTask(cancelledTask.id)}>Remove</button>
          </li>
        {/each}
      {/if}

      {#if completedTasks.length > 0}
        {#each completedTasks as completedTask}
          <li class="task task--completed">
            <div class="task-title">Completed</div>
            <div class="task-meta">{formatTaskSummary(completedTask)}</div>
            <div class="task-message">{completedTask.message}</div>
            <details class="task-paths">
              <summary>Added paths ({completedTask.selectedPaths.length})</summary>
              <ul>
                {#each completedTask.selectedPaths as path}
                  <li title={path}>{path}</li>
                {/each}
              </ul>
            </details>
            <button type="button" class="task-action" on:click={() => onRemoveTask(completedTask.id)}>Remove</button>
          </li>
        {/each}
      {/if}
    </ul>
  {/if}
</aside>

<style>
  .dock-menu {
    width: 260px;
    min-width: 260px;
    border-right: 1px solid var(--border-color, #d8d8d8);
    padding: 1rem;
    overflow-y: auto;
    background: var(--surface-muted, #fafafa);
  }

  .dock-menu.collapsed {
    width: 90px;
    min-width: 90px;
    overflow: hidden;
  }

  .sidebar-toggle {
    margin-bottom: 0.75rem;
    width: 100%;
  }

  .dock-menu h2 {
    margin: 0 0 0.75rem;
    font-size: 1rem;
  }

  .dock-menu ul {
    margin: 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

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

  .task--empty {
    text-align: center;
    color: var(--text-muted, #666);
  }

  @media (max-width: 900px) {
    .dock-menu {
      width: 100%;
      min-width: 0;
      border-right: none;
      border-bottom: 1px solid var(--border-color, #d8d8d8);
    }
  }
</style>
