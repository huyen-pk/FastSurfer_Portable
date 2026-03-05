<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { confirm } from "@tauri-apps/plugin-dialog";
  import { exit } from "@tauri-apps/plugin-process";
  import ProcessedResultViewer from "./ProcessedResultViewer.svelte";
  import TaskSidebar from "./TaskSidebar.svelte";
  import FilePickerPanel from "./FilePickerPanel.svelte";
  import { createTransport } from "../transport";
  import type {
    InferenceProgressEvent,
    InferenceResult,
    ProcessInferenceResponse,
    RawInferenceResult
  } from "../types/inference";

  const transport = createTransport();

  type TaskStatus = "waiting" | "processing" | "completed" | "failed" | "cancelled";

  interface ProcessingTask {
    id: string;
    createdAt: number;
    status: TaskStatus;
    selectedPaths: string[];
    total: number;
    completed: number;
    progress: number;
    message: string;
    currentPath: string;
    results: InferenceResult[];
    ackMessage: string;
    errorMessage: string;
  }

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

  let selectedPaths: string[] = [];
  let inferenceResults: InferenceResult[] = [];
  let errorMessage = "";
  let ackMessage = "";
  let isRunning = false;
  let isLoadingScreenVisible = false;
  let isLoadingScreenDismissed = false;
  let tasks: ProcessingTask[] = [];
  let isProcessingQueue = false;
  let isSidebarCollapsed = false;
  let activeResult: InferenceResult | null = null;
  let isViewerOpen = false;
  let filePickerInput: HTMLInputElement | null = null;
  let folderPickerInput: HTMLInputElement | null = null;
  let activeTask: ProcessingTask | null = null;
  let waitingTasks: ProcessingTask[] = [];
  let completedTasks: ProcessingTask[] = [];
  let failedTasks: ProcessingTask[] = [];
  let cancelledTasks: ProcessingTask[] = [];
  let allowClose = false;
  let isAppClosing = false;
  const pendingTaskRemovals = new Set<string>();

  $: activeTask = tasks.find((task) => task.status === "processing") || null;
  $: waitingTasks = tasks.filter((task) => task.status === "waiting");
  $: completedTasks = tasks.filter((task) => task.status === "completed");
  $: failedTasks = tasks.filter((task) => task.status === "failed");
  $: cancelledTasks = tasks.filter((task) => task.status === "cancelled");

  function createTaskId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  }

  function isTauriRuntime(): boolean {
    return typeof window !== "undefined" && "__TAURI__" in window;
  }

  function getFileName(path: string): string {
    if (!path) return "";
    const normalized = path.replaceAll("\\", "/");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || normalized;
  }

  function splitPickedPaths(paths: string[]): { files: string[]; folders: string[] } {
    const files: string[] = [];
    const folders: string[] = [];

    for (const path of paths) {
      const normalized = path.toLowerCase();
      if (
        normalized.endsWith(".nii")
        || normalized.endsWith(".nii.gz")
        || normalized.endsWith(".mgz")
        || normalized.endsWith(".mgh")
      ) {
        files.push(path);
      } else {
        folders.push(path);
      }
    }

    return { files, folders };
  }

  function normalizeOpenResult(result: string | string[] | null): string[] {
    if (!result) return [];
    return Array.isArray(result) ? result : [result];
  }

  function normalizeBrowserInputFiles(input: HTMLInputElement | null): string[] {
    if (!input?.files) return [];
    return Array.from(input.files)
      .map((file) => file.webkitRelativePath || file.name)
      .filter(Boolean);
  }

  function addPickedPaths(paths: string[]): void {
    const dedupedIncoming = [...new Set(paths.filter(Boolean))];
    if (dedupedIncoming.length === 0) return;
    selectedPaths = [...new Set([...selectedPaths, ...dedupedIncoming])];
  }

  function handleFileInputChange(): void {
    const paths = normalizeBrowserInputFiles(filePickerInput);
    addPickedPaths(paths);
  }

  function handleFolderInputChange(): void {
    const paths = normalizeBrowserInputFiles(folderPickerInput);
    addPickedPaths(paths);
  }

  function pickFromBrowserInput(input: HTMLInputElement | null): Promise<string[]> {
    if (!input) return Promise.resolve([]);

    return new Promise((resolve) => {
      const onChange = () => {
        const pickedPaths = normalizeBrowserInputFiles(input);
        input.value = "";
        resolve(pickedPaths);
      };

      input.addEventListener("change", onChange, { once: true });
      input.click();
    });
  }

  function mapInferenceResult(result: RawInferenceResult): InferenceResult {
    const rawArtifacts = result.artifacts;
    const brainmaskPath = rawArtifacts?.brainmaskPath || rawArtifacts?.brainmask_path;
    const asegPath = rawArtifacts?.asegPath || rawArtifacts?.aseg_path;

    const artifacts = brainmaskPath || asegPath
      ? {
        ...(brainmaskPath ? { brainmaskPath: String(brainmaskPath) } : {}),
        ...(asegPath ? { asegPath: String(asegPath) } : {})
      }
      : undefined;

    const hasQcPassed = typeof result.qc?.passed === "boolean";
    const hasQcMessage = typeof result.qc?.message === "string" && result.qc.message.length > 0;
    const qc = hasQcPassed || hasQcMessage
      ? {
        ...(hasQcPassed ? { passed: Boolean(result.qc?.passed) } : {}),
        ...(hasQcMessage ? { message: String(result.qc?.message) } : {})
      }
      : undefined;

    return {
      inputPath: String(result.inputPath || result.input_path || ""),
      outputPath: String(result.outputPath || result.output_path || ""),
      outputFilename: String(result.outputFilename || result.output_filename || ""),
      runResult: String(result.runResult || result.run_result || ""),
      ...(artifacts ? { artifacts } : {}),
      ...(qc ? { qc } : {})
    };
  }

  function openResultViewer(result: InferenceResult): void {
    activeResult = result;
    isViewerOpen = true;
  }

  async function openResultInFileManager(path: string): Promise<void> {
    errorMessage = "";
    try {
      await transport.openInFileManager(path);
    } catch (error: unknown) {
      errorMessage = String(error);
      console.error(error);
    }
  }

  function closeResultViewer(): void {
    isViewerOpen = false;
    activeResult = null;
  }

  function toggleSidebar(): void {
    isSidebarCollapsed = !isSidebarCollapsed;
  }

  function removeSelectedPath(pathToRemove: string): void {
    selectedPaths = selectedPaths.filter((path) => path !== pathToRemove);
  }

  function clearSelectedPaths(): void {
    selectedPaths = [];
  }

  function removeTask(taskId: string): void {
    tasks = tasks.filter((task) => task.id !== taskId);
  }

  function formatTaskSummary(task: ProcessingTask): string {
    if (task.total <= 0) {
      return `${task.selectedPaths.length} path(s)`;
    }
    return `${task.completed}/${task.total} path(s)`;
  }

  function gatherResultDirectories(): string[] {
    const dirs = new Set<string>();
    for (const task of tasks) {
      for (const result of task.results) {
        if (!result.outputPath) continue;
        const normalized = result.outputPath.replaceAll("\\", "/");
        const idx = normalized.lastIndexOf("/");
        if (idx > 0) dirs.add(normalized.slice(0, idx));
      }
    }
    return [...dirs].sort();
  }

  function hasRunningOrQueuedTasks(): boolean {
    return tasks.some((task) => task.status === "processing" || task.status === "waiting");
  }

  async function quitApp() {
    await exit(0);
  }

  export async function maybeConfirmClose(event: { preventDefault: () => void }): Promise<void> {
    if (allowClose) {
      return;
    }

    event.preventDefault();

    if (isAppClosing) {
      return;
    }

    if (hasRunningOrQueuedTasks()) {
      const resultDirs = gatherResultDirectories();
      const resultHint = resultDirs.length > 0
        ? `\n\nCurrent result directories:\n${resultDirs.join("\n")}`
        : "\n\nResults are written to output directories reported in task results.";

      const shouldQuit = await confirm(
        `Tasks are still running or queued. Quit now to stop the app, or keep running to continue processing.${resultHint}`,
        {
          title: "Quit FastSurfer?",
          kind: "warning",
          okLabel: "Quit",
          cancelLabel: "Keep Running"
        }
      );

      if (!shouldQuit) {
        return;
      }
    }

    isAppClosing = true;
    isLoadingScreenVisible = true;
    isLoadingScreenDismissed = false;

    allowClose = true;
    await quitApp();
  }

  async function pickFiles(): Promise<void> {
    errorMessage = "";
    ackMessage = "";

    if (!isTauriRuntime()) {
      const files = await pickFromBrowserInput(filePickerInput);
      addPickedPaths(files);
      return;
    }

    try {
      const fileResult = await open({
        multiple: true,
        directory: false,
        title: "Select medical image files"
      });

      const files = normalizeOpenResult(fileResult);
      addPickedPaths(files);
    } catch (error: unknown) {
      errorMessage = String(error);
      console.error(error);
    }
  }

  async function pickFolders(): Promise<void> {
    errorMessage = "";
    ackMessage = "";

    if (!isTauriRuntime()) {
      const folders = await pickFromBrowserInput(folderPickerInput);
      addPickedPaths(folders);
      return;
    }

    try {
      const folderResult = await open({
        multiple: true,
        directory: true,
        title: "Select folders containing medical image files"
      });

      const folders = normalizeOpenResult(folderResult);
      addPickedPaths(folders);
    } catch (error: unknown) {
      errorMessage = String(error);
      console.error(error);
    }
  }

  function updateTask(taskId: string, updater: (task: ProcessingTask) => ProcessingTask): void {
    tasks = tasks.map((task) => (task.id === taskId ? updater(task) : task));
  }

  function handleTaskProgress(taskId: string, event: InferenceProgressEvent): void {
    updateTask(taskId, (task) => ({
      ...task,
      status: event.status === "failed"
        ? "failed"
        : event.status === "completed"
          ? "completed"
          : event.status === "cancelled"
            ? "cancelled"
          : task.status,
      total: typeof event.total === "number" ? event.total : task.total,
      completed: typeof event.completed === "number" ? event.completed : task.completed,
      progress: Math.max(0, Math.min(100, event.progress ?? task.progress)),
      message: event.message || task.message,
      currentPath: event.currentPath || task.currentPath
    }));

    if (event.status === "cancelled" && pendingTaskRemovals.has(taskId)) {
      pendingTaskRemovals.delete(taskId);
      removeTask(taskId);
    }
  }

  async function cancelTask(taskId: string): Promise<void> {
    const task = tasks.find((candidate) => candidate.id === taskId);
    if (!task) return;

    if (task.status === "waiting") {
      removeTask(taskId);
      return;
    }

    if (task.status === "processing") {
      try {
        pendingTaskRemovals.add(taskId);
        await transport.cancelTask(taskId);
        updateTask(taskId, (current) => ({
          ...current,
          message: "Cancellation requested..."
        }));
      } catch (error: unknown) {
        pendingTaskRemovals.delete(taskId);
        errorMessage = String(error);
      }
      return;
    }

    removeTask(taskId);
  }

  async function runTask(task: ProcessingTask): Promise<void> {
    const { files, folders } = splitPickedPaths(task.selectedPaths);

    updateTask(task.id, (current) => ({
      ...current,
      status: "processing",
      progress: 0,
      message: "Initializing processing...",
      currentPath: "",
      errorMessage: ""
    }));

    isRunning = true;
    isLoadingScreenVisible = true;
    isLoadingScreenDismissed = false;

    try {
      const data: ProcessInferenceResponse = await transport.processInferenceWithProgress(
        {
          filePaths: files,
          folderPaths: folders
        },
        task.id,
        (event) => {
          handleTaskProgress(task.id, event);
        }
      );

      const mappedResults = Array.isArray(data?.results)
        ? data.results.map(mapInferenceResult)
        : [];
      const responseAck = data?.ackMessage || data?.ack_message || "Processing completed.";

      if (pendingTaskRemovals.has(task.id)) {
        pendingTaskRemovals.delete(task.id);
        removeTask(task.id);
        return;
      }

      updateTask(task.id, (current) => ({
        ...current,
        status: "completed",
        progress: 100,
        message: responseAck,
        currentPath: "",
        results: mappedResults,
        ackMessage: responseAck,
        errorMessage: ""
      }));

      inferenceResults = mappedResults;
      ackMessage = responseAck;
      errorMessage = "";
    } catch (error: unknown) {
      const message = String(error);

      if (pendingTaskRemovals.has(task.id)) {
        pendingTaskRemovals.delete(task.id);
        removeTask(task.id);
        return;
      }

      updateTask(task.id, (current) => ({
        ...current,
        status: "failed",
        progress: current.progress > 0 ? current.progress : 0,
        message,
        errorMessage: message
      }));
      errorMessage = message;
      ackMessage = "";
      inferenceResults = [];
      console.error(error);
    } finally {
      isRunning = false;
      isLoadingScreenVisible = false;
      isLoadingScreenDismissed = false;
    }
  }

  function dismissLoadingScreen(): void {
    isLoadingScreenDismissed = true;
  }

  async function processQueue(): Promise<void> {
    if (isProcessingQueue) return;
    isProcessingQueue = true;

    try {
      while (true) {
        const nextTask = tasks.find((task) => task.status === "waiting");
        if (!nextTask) break;
        await runTask(nextTask);
      }
    } finally {
      isProcessingQueue = false;
    }
  }

  async function processImage(): Promise<void> {
    if (selectedPaths.length === 0) {
      errorMessage = "Please pick at least one file or folder first.";
      return;
    }

    errorMessage = "";
    ackMessage = "";
    inferenceResults = [];

    const normalizedSelection = [...new Set(selectedPaths.filter(Boolean))];
    const taskId = createTaskId();
    const newTask: ProcessingTask = {
      id: taskId,
      createdAt: Date.now(),
      status: "waiting",
      selectedPaths: normalizedSelection,
      total: normalizedSelection.length,
      completed: 0,
      progress: 0,
      message: `Waiting (${normalizedSelection.length} path(s))`,
      currentPath: "",
      results: [],
      ackMessage: "",
      errorMessage: ""
    };

    tasks = [...tasks, newTask];
    selectedPaths = [];
    void processQueue();
  }
</script>

<main>
  <TaskSidebar
    isSidebarCollapsed={isSidebarCollapsed}
    activeTask={activeTask as SidebarTask | null}
    waitingTasks={waitingTasks as SidebarTask[]}
    completedTasks={completedTasks as SidebarTask[]}
    failedTasks={failedTasks as SidebarTask[]}
    cancelledTasks={cancelledTasks as SidebarTask[]}
    formatTaskSummary={formatTaskSummary as (task: SidebarTask) => string}
    onToggleSidebar={toggleSidebar}
    onCancelTask={(taskId: string) => {
      void cancelTask(taskId);
    }}
    onRemoveTask={removeTask}
  />

  <section class="main-panel" aria-busy={isRunning}>
    <h1>FastSurfer Processing</h1>

    <input
      bind:this={filePickerInput}
      data-testid="file-picker-input"
      on:change={handleFileInputChange}
      type="file"
      multiple
      accept=".nii,.nii.gz,.mgz,.mgh"
      style="display: none"
    />
    <input
      bind:this={folderPickerInput}
      data-testid="folder-picker-input"
      on:change={handleFolderInputChange}
      type="file"
      multiple
      webkitdirectory
      style="display: none"
    />

    <FilePickerPanel
      selectedPaths={selectedPaths}
      onPickFiles={pickFiles}
      onPickFolders={pickFolders}
      onProcess={processImage}
      onClearAll={clearSelectedPaths}
      onRemovePath={removeSelectedPath}
    />

    {#if ackMessage}
      <p class="ack">{ackMessage}</p>
    {/if}

    {#if activeTask}
      <div class="progress-wrap">
        <progress
          max="100"
          value={activeTask.progress}
        ></progress>
        <span>{activeTask.progress}% ({activeTask.completed}/{activeTask.total})</span>
      </div>
    {/if}

    {#if errorMessage}
      <p class="error">{errorMessage}</p>
    {/if}

    {#if inferenceResults.length > 0}
      <div class="results">
        <h3>Processed Files</h3>
        <ul>
          {#each inferenceResults as result}
            <li>
              <button
                class="result-link"
                on:click={() => openResultViewer(result)}
              >
                {getFileName(result.inputPath)}
              </button>
              <div class="result-path-row">
                <span>Saved result:</span>
                <button
                  class="path-link"
                  on:click={() => openResultInFileManager(result.outputPath)}
                  title={result.outputPath}
                >
                  {result.outputPath}
                </button>
              </div>
            </li>
          {/each}
        </ul>
      </div>
    {/if}

    {#if isViewerOpen && activeResult}
      <ProcessedResultViewer
        result={activeResult}
        on:close={closeResultViewer}
      />
    {/if}
  </section>

  {#if (isLoadingScreenVisible && !isLoadingScreenDismissed) || isAppClosing}
    <div class="loading-overlay" role="status" aria-live="polite" aria-label="Processing in progress">
      <div class="loading-card">
        {#if !isAppClosing}
          <button class="loading-close" type="button" on:click={dismissLoadingScreen} aria-label="Close processing popup">×</button>
        {/if}
        <h3>{isAppClosing ? "Closing FastSurfer" : "Processing in progress"}</h3>
        <p>{isAppClosing ? "Stopping running processes and exiting..." : "Please wait while backend is running."}</p>
      </div>
    </div>
  {/if}
</main>

<style>
  main {
    display: flex;
    min-height: 100vh;
    text-align: left;
  }

  .main-panel {
    flex: 1;
    padding: 1.25rem;
    min-width: 0;
  }

  .progress-wrap {
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .progress-wrap progress {
    width: min(520px, 90vw);
    height: 1rem;
  }

  .results {
    margin-top: 1rem;
    text-align: left;
  }

  .ack {
    margin-top: 0.75rem;
  }

  .results ul {
    padding-left: 1rem;
  }

  .result-link {
    border: none;
    background: transparent;
    color: #0a58ca;
    text-decoration: underline;
    cursor: pointer;
    padding: 0;
    font: inherit;
    text-align: left;
  }

  .result-path-row {
    margin-top: 0.25rem;
    display: flex;
    gap: 0.35rem;
    align-items: baseline;
    flex-wrap: wrap;
    font-size: 0.85rem;
    word-break: break-all;
  }

  .path-link {
    border: none;
    background: transparent;
    text-decoration: underline;
    cursor: pointer;
    padding: 0;
    font: inherit;
    text-align: left;
    word-break: break-all;
  }

  .error {
    margin-top: 1rem;
    color: #b00020;
  }

  .loading-overlay {
    position: fixed;
    inset: 0;
    background: rgb(0 0 0 / 35%);
    display: grid;
    place-items: center;
    z-index: 1000;
    pointer-events: none;
  }

  .loading-card {
    position: relative;
    background: var(--surface, #fff);
    border-radius: 0.6rem;
    padding: 1rem 1.25rem;
    min-width: 260px;
    max-width: 80vw;
    pointer-events: auto;
  }

  .loading-close {
    position: absolute;
    top: 0.35rem;
    right: 0.35rem;
    border: none;
    background: transparent;
    cursor: pointer;
    font-size: 1.1rem;
    line-height: 1;
    padding: 0.2rem 0.35rem;
  }

  @media (max-width: 900px) {
    main {
      flex-direction: column;
    }
  }
</style>
