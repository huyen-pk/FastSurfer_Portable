<script>
  import { createEventDispatcher, onMount } from "svelte";
  import { Niivue } from "@niivue/niivue";

  export let result;

  const dispatch = createEventDispatcher();

  let canvas;
  let nv;

  function toViewerUrl(path) {
    if (!path) return "";
    if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("file://")) {
      return path;
    }
    return `file://${path}`;
  }

  onMount(() => {
    nv = new Niivue();
    nv.attachToCanvas(canvas);
  });

  $: if (nv && result?.outputPath) {
    nv.loadVolumes([
      {
        url: toViewerUrl(result.outputPath),
        name: result.outputFilename || "Processed Result",
        colormap: "gray"
      }
    ]);
  }
</script>

<div class="viewer-overlay">
  <div class="viewer-card">
    <div class="viewer-header">
      <h2>{result?.outputFilename || "Processed Result"}</h2>
      <button on:click={() => dispatch("close")}>Close</button>
    </div>
    <p class="path">{result?.outputPath}</p>
    <canvas
      bind:this={canvas}
      width="900"
      height="650"
    ></canvas>
  </div>
</div>

<style>
  .viewer-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .viewer-card {
    background: #fff;
    color: #111;
    padding: 1rem;
    border-radius: 8px;
    width: min(95vw, 980px);
  }

  .viewer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .path {
    margin: 0.5rem 0 1rem;
    font-size: 0.85rem;
    word-break: break-all;
  }

  canvas {
    border: 1px solid #444;
    width: 100%;
    height: auto;
    display: block;
  }
</style>
