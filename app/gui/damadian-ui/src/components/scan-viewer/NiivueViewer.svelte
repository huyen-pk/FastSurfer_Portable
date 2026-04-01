<script lang="ts">
  import { Niivue, SHOW_RENDER, SLICE_TYPE } from "@niivue/niivue";
  import { onMount } from "svelte";
  import defaultAnatomicalVolumeUrl from "../../../testing/data/BRATS_001/BRATS_001.nii?url";
  import defaultLabelVolumeUrl from "../../../testing/data/BRATS_001/BRATS_001_labels.nii?url";

  export let label: string;
  export let sublabel = "";
  export let paneClass = "bg-surface-container-lowest";
  export let accent = false;
  export let anatomicalVolumeUrl = defaultAnatomicalVolumeUrl;
  export let labelVolumeUrl = defaultLabelVolumeUrl;
  export let sliceType: SLICE_TYPE = SLICE_TYPE.AXIAL;
  export let showPaneLabel = true;

  let canvasElement: HTMLCanvasElement | null = null;
  let viewerRoot: HTMLDivElement | null = null;
  let niivue: Niivue | null = null;
  let isLoading = true;
  let viewerReady = false;
  let fallbackMessage = "";

  function hasWebGlSupport(): boolean {
    if (typeof window === "undefined" || typeof document === "undefined") {
      return false;
    }

    if (typeof navigator !== "undefined" && /jsdom/i.test(navigator.userAgent)) {
      return false;
    }

    try {
      const probeCanvas = document.createElement("canvas");

      return Boolean(probeCanvas.getContext("webgl2") || probeCanvas.getContext("webgl") || probeCanvas.getContext("experimental-webgl"));
    } catch {
      return false;
    }
  }

  function setFallback(message: string): void {
    fallbackMessage = message;
    viewerReady = false;
    isLoading = false;
  }

  onMount(() => {
    let disposed = false;

    async function initializeViewer(): Promise<void> {
      if (!canvasElement || !viewerRoot) {
        setFallback("Viewer canvas is unavailable.");
        return;
      }

      if (!hasWebGlSupport()) {
        setFallback("Interactive scan rendering requires WebGL support.");
        return;
      }

      try {
        const viewer = new Niivue({
          backColor: [0.02, 0.06, 0.08, 1],
          crosshairColor: [0, 0.85, 0.95, 0.9],
          crosshairWidth: 1,
          show3Dcrosshair: true,
          textHeight: 0.03,
          isColorbar: false,
          isRuler: false,
          multiplanarEqualSize: true
        });

        niivue = viewer;
        viewer.onLocationChange = () => {
          viewerRoot?.setAttribute("data-niivue-ready", "true");
        };
        await viewer.attachToCanvas(canvasElement);
        viewer.opts.multiplanarShowRender = sliceType === SLICE_TYPE.RENDER ? SHOW_RENDER.ALWAYS : SHOW_RENDER.NEVER;
        viewer.setSliceType(sliceType);
        viewer.setInterpolation(false);
        await viewer.loadVolumes([
          {
            url: anatomicalVolumeUrl,
            colormap: "gray",
            opacity: 1,
            trustCalMinMax: true,
            colorbarVisible: false
          },
          {
            url: labelVolumeUrl,
            colormap: "actc",
            opacity: 0.55,
            cal_min: 0.5,
            trustCalMinMax: true,
            alphaThreshold: true,
            ignoreZeroVoxels: true,
            colorbarVisible: false
          }
        ]);

        if (disposed) {
          viewerRoot.removeAttribute("data-niivue-ready");
          return;
        }

        viewerReady = true;
        isLoading = false;
        fallbackMessage = "";
        viewerRoot.setAttribute("data-niivue-ready", "true");
        viewer.resizeListener();
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to load the scan volume.";
        viewerRoot?.setAttribute("data-niivue-ready", "false");
        setFallback(message);
      }
    }

    void initializeViewer();

    return () => {
      disposed = true;
      viewerRoot?.removeAttribute("data-niivue-ready");

      if (niivue?.volumes?.length) {
        const volumes = [...niivue.volumes];

        for (const volume of volumes) {
          niivue.removeVolume(volume);
        }
      }

      niivue = null;
    };
  });
</script>

<div
  bind:this={viewerRoot}
  aria-label="BRATS segmentation viewer"
  class={`group relative overflow-hidden ${paneClass} ${accent ? 'border-2 border-tertiary/20' : ''}`}
  data-niivue-root="true"
  data-niivue-label={label}
  data-niivue-status={viewerReady ? "ready" : fallbackMessage ? "fallback" : "loading"}
>
  {#if accent}
    <div class="pointer-events-none absolute inset-0 bg-gradient-to-br from-tertiary/5 to-transparent"></div>
  {/if}

  <canvas bind:this={canvasElement} class="h-full w-full" data-testid="niivue-canvas"></canvas>

  {#if showPaneLabel && (label || sublabel)}
    <div class="pointer-events-none absolute left-3 top-3 flex flex-col">
      {#if label}
        <span class={`text-[10px] font-bold tracking-tighter ${accent ? 'text-tertiary' : 'text-tertiary/60'}`}>{label}</span>
      {/if}
      {#if sublabel}
        <span class={`text-[9px] ${accent ? 'text-on-surface/60' : 'text-on-surface/40'}`}>{sublabel}</span>
      {/if}
    </div>
  {/if}

  {#if isLoading}
    <div class="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-surface-container-lowest via-surface-container-lowest/80 to-transparent px-4 py-3 text-[10px] uppercase tracking-[0.28em] text-tertiary/80">
      <span>Loading MRI + labels</span>
      <span>BRATS 001</span>
    </div>
  {:else if fallbackMessage}
    <div class="pointer-events-none absolute inset-0 flex items-end bg-gradient-to-t from-surface-container-lowest via-surface-container-lowest/70 to-transparent p-4">
      <div class="max-w-xs rounded-lg bg-surface-container-high/80 px-3 py-2 text-xs text-on-surface-variant shadow-[0_20px_40px_rgba(0,0,0,0.25)] backdrop-blur-md">
        {fallbackMessage}
      </div>
    </div>
  {/if}

  <slot />
</div>