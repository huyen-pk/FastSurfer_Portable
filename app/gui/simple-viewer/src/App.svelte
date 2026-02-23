<script lang="ts">
  import { onDestroy } from "svelte";
  import { open } from "@tauri-apps/plugin-dialog";
  import ProcessedResultViewer from "./components/ProcessedResultViewer.svelte";
  import { createTransport } from "./transport";
  import type { InferenceResult, ProcessInferenceResponse, RawInferenceResult } from "./types/inference";

  const transport = createTransport();

  let selectedPaths: string[] = [];
  let inferenceResults: InferenceResult[] = [];
  let errorMessage = "";
  let ackMessage = "";
  let isRunning = false;
  let hasProcessingStarted = false;
  let progressValue = 0;
  let progressTimer: ReturnType<typeof setInterval> | null = null;
  let activeResult: InferenceResult | null = null;
  let isViewerOpen = false;
  let filePickerInput: HTMLInputElement | null = null;
  let folderPickerInput: HTMLInputElement | null = null;

  function isTauriRuntime(): boolean {
    return typeof window !== "undefined" && "__TAURI__" in window;
  }

  function getFileName(path: string): string {
    if (!path) return "";
    const normalized = path.replaceAll("\\", "/");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || normalized;
  }

  function startProgress(): void {
    progressValue = 5;
    if (progressTimer) clearInterval(progressTimer);
    progressTimer = setInterval(() => {
      if (progressValue < 90) {
        progressValue += 5;
      }
    }, 350);
  }

  function stopProgress(success: boolean): void {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    progressValue = success ? 100 : 0;
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
    selectedPaths = [...new Set([...selectedPaths, ...paths])];
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
        resolve(pickedPaths);
      };

      input.addEventListener("change", onChange, { once: true });
      input.click();
    });
  }

  function mapInferenceResult(result: RawInferenceResult): InferenceResult {
    return {
      inputPath: String(result.inputPath || result.input_path || ""),
      outputPath: String(result.outputPath || result.output_path || ""),
      outputFilename: String(result.outputFilename || result.output_filename || ""),
      runResult: String(result.runResult || result.run_result || "")
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

  onDestroy(() => {
    if (progressTimer) clearInterval(progressTimer);
  });

  async function pickFilesAndFolders(): Promise<void> {
    errorMessage = "";
    ackMessage = "";

    if (!isTauriRuntime()) {
      const files = await pickFromBrowserInput(filePickerInput);
      const folders = await pickFromBrowserInput(folderPickerInput);
      selectedPaths = [...new Set([...files, ...folders])];
      return;
    }

    try {
      const fileResult = await open({
        multiple: true,
        directory: false,
        title: "Select medical image files"
      });

      const folderResult = await open({
        multiple: true,
        directory: true,
        title: "Select folders containing medical image files"
      });

      const files = normalizeOpenResult(fileResult);
      const folders = normalizeOpenResult(folderResult);

      selectedPaths = [...new Set([...files, ...folders])];
    } catch (error: unknown) {
      errorMessage = String(error);
      console.error(error);
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
    isRunning = true;
    hasProcessingStarted = false;

    const { files, folders } = splitPickedPaths(selectedPaths);

    try {
      hasProcessingStarted = true;
      startProgress();

      const data: ProcessInferenceResponse = await transport.processInference({
        filePaths: files,
        folderPaths: folders
      });

      ackMessage = data?.ackMessage || data?.ack_message || "Processing completed.";

      inferenceResults = Array.isArray(data?.results)
        ? data.results.map(mapInferenceResult)
        : [];
      stopProgress(true);
      console.log(inferenceResults);
    } catch (error: unknown) {
      errorMessage = String(error);
      stopProgress(false);
      console.error(error);
    } finally {
      isRunning = false;
    }
  }
</script>

<main>
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

  <div class="controls">
    <button on:click={pickFilesAndFolders}>Pick Files/Folders</button>
    <button
      on:click={processImage}
      disabled={isRunning || selectedPaths.length === 0}
    >
      {isRunning ? "Processing..." : "Start Processing"}
    </button>
  </div>

  {#if selectedPaths.length > 0}
    <div class="picked-paths">
      <h3>Picked Paths</h3>
      <ul>
        {#each selectedPaths as path}
          <li title={path}>{path}</li>
        {/each}
      </ul>
    </div>
  {/if}

  {#if ackMessage}
    <p class="ack">{ackMessage}</p>
  {/if}

  {#if hasProcessingStarted}
    <div class="progress-wrap">
      <progress
        max="100"
        value={progressValue}
      ></progress>
      <span>{progressValue}%</span>
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
</main>

<style>
  .controls {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
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

  .picked-paths {
    margin-top: 1rem;
    text-align: left;
  }

  .picked-paths ul {
    margin: 0;
    padding-left: 1rem;
  }

  .picked-paths li {
    word-break: break-all;
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
</style>