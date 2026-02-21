<script>
  import { onDestroy } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";
  import ProcessedResultViewer from "./components/ProcessedResultViewer.svelte";

  let selectedFiles = [];
  let selectedFolders = [];
  let inferenceResults = [];
  let errorMessage = "";
  let isRunning = false;
  let progressValue = 0;
  let progressTimer = null;
  let activeResult = null;
  let isViewerOpen = false;

  function getFileName(path) {
    if (!path) return "";
    const normalized = path.replaceAll("\\", "/");
    const parts = normalized.split("/");
    return parts[parts.length - 1] || normalized;
  }

  function startProgress() {
    progressValue = 5;
    if (progressTimer) clearInterval(progressTimer);
    progressTimer = setInterval(() => {
      if (progressValue < 90) {
        progressValue += 5;
      }
    }, 350);
  }

  function stopProgress(success) {
    if (progressTimer) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    progressValue = success ? 100 : 0;
  }

  function openResultViewer(result) {
    activeResult = result;
    isViewerOpen = true;
  }

  async function openResultInFileManager(path) {
    errorMessage = "";
    try {
      await invoke("open_result_in_file_manager", {
        resultPath: path
      });
    } catch (error) {
      errorMessage = String(error);
      console.error(error);
    }
  }

  function closeResultViewer() {
    isViewerOpen = false;
    activeResult = null;
  }

  onDestroy(() => {
    if (progressTimer) clearInterval(progressTimer);
  });

  async function selectFiles() {
    errorMessage = "";
    const result = await open({
      multiple: true,
      directory: false,
      title: "Select medical image files"
    });

    if (!result) return;
    selectedFiles = Array.isArray(result) ? result : [result];
  }

  async function selectFolders() {
    errorMessage = "";
    const result = await open({
      multiple: true,
      directory: true,
      title: "Select folders containing medical image files"
    });

    if (!result) return;
    selectedFolders = Array.isArray(result) ? result : [result];
  }

  async function processImage() {
    errorMessage = "";
    inferenceResults = [];
    isRunning = true;
    startProgress();

    try {
      const data = await invoke("run_fastsurfer_inference", {
        filePaths: selectedFiles,
        folderPaths: selectedFolders
      });

      inferenceResults = Array.isArray(data) ? data : [];
      stopProgress(true);
      console.log(inferenceResults);
    } catch (error) {
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

  <div class="controls">
    <button on:click={selectFiles}>Select Files</button>
    <button on:click={selectFolders}>Select Folders</button>
    <button
      on:click={processImage}
      disabled={isRunning}
    >
      {isRunning ? "Processing..." : "Process Selected Paths"}
    </button>
  </div>

  <div class="progress-wrap">
    <progress
      max="100"
      value={progressValue}
    ></progress>
    <span>{progressValue}%</span>
  </div>

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