<script lang="ts">
  import TaskSidebarItem from "./TaskSidebarItem.svelte";

  type TaskStatus = "waiting" | "processing" | "completed" | "failed" | "cancelled";

  interface SidebarTask {
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
        <TaskSidebarItem
          task={activeTask}
          title="Processing"
          variant="task--processing"
          summary={formatTaskSummary(activeTask)}
          showProgress={true}
          showMessage={true}
          showCurrentPath={true}
          actionLabel="Cancel & Remove"
          actionDanger={true}
          onAction={onCancelTask}
        />
      {/if}

      {#if waitingTasks.length > 0}
        {#each waitingTasks as waitingTask}
          <TaskSidebarItem
            task={waitingTask}
            title="Waiting"
            variant="task--waiting"
            summary={formatTaskSummary(waitingTask)}
            actionLabel="Remove"
            onAction={onCancelTask}
          />
        {/each}
      {:else if !activeTask}
        <li class="task task--empty">No queued task</li>
      {/if}

      {#if failedTasks.length > 0}
        {#each failedTasks as failedTask}
          <TaskSidebarItem
            task={failedTask}
            title="Failed"
            variant="task--failed"
            summary={formatTaskSummary(failedTask)}
            showMessage={true}
            actionLabel="Remove"
            onAction={onRemoveTask}
          />
        {/each}
      {/if}

      {#if cancelledTasks.length > 0}
        {#each cancelledTasks as cancelledTask}
          <TaskSidebarItem
            task={cancelledTask}
            title="Cancelled"
            variant="task--cancelled"
            summary={formatTaskSummary(cancelledTask)}
            showMessage={true}
            actionLabel="Remove"
            onAction={onRemoveTask}
          />
        {/each}
      {/if}

      {#if completedTasks.length > 0}
        {#each completedTasks as completedTask}
          <TaskSidebarItem
            task={completedTask}
            title="Completed"
            variant="task--completed"
            summary={formatTaskSummary(completedTask)}
            showMessage={true}
            actionLabel="Remove"
            onAction={onRemoveTask}
          />
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
