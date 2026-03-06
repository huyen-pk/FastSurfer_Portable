<script lang="ts">
  export let selectedPaths: string[] = [];
  export let onPickFiles: () => void;
  export let onPickFolders: () => void;
  export let onProcess: () => void;
  export let onClearAll: () => void;
  export let onRemovePath: (path: string) => void;
</script>

<div class="controls">
  <button on:click={onPickFiles}>Pick Files</button>
  <button on:click={onPickFolders}>Pick Folders</button>
</div>

{#if selectedPaths.length > 0}
  <div class="picked-paths">
    <div class="picked-header">
      <h3>Added Paths</h3>
      <button
        type="button"
        class="process-button"
        on:click={onProcess}
        disabled={selectedPaths.length === 0}
      >
        Process
      </button>
      <button type="button" class="task-action" on:click={onClearAll}>Clear All</button>
    </div>
    <ul>
      {#each selectedPaths as path}
        <li title={path}>
          <span>{path}</span>
          <button type="button" class="task-action" on:click={() => onRemovePath(path)}>Remove</button>
        </li>
      {/each}
    </ul>
  </div>
{/if}

<style>
  .task-action {
    margin-top: 0.4rem;
  }

  .controls {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .picked-paths {
    margin-top: 1rem;
    text-align: left;
  }

  .picked-paths ul {
    margin: 0;
    padding-left: 1rem;
  }

  .picked-paths li {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
    word-break: break-all;
  }

  .picked-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.5rem;
  }

  .process-button {
    background: var(--success-color, #198754);
    color: var(--success-contrast-color, #fff);
    border: 1px solid var(--success-color, #198754);
  }

  .process-button:disabled {
    opacity: 0.65;
    cursor: not-allowed;
  }
</style>
