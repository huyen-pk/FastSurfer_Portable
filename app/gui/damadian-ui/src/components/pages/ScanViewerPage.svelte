<script lang="ts">
  import { SLICE_TYPE } from "@niivue/niivue";
  import DetailListSection from "../scan-viewer/DetailListSection.svelte";
  import InfoGridCard from "../scan-viewer/InfoGridCard.svelte";
  import NiivueViewer from "../scan-viewer/NiivueViewer.svelte";

  const scanMetadata = [
    { label: "Field of View", value: "240 x 240" },
    { label: "TR / TE", value: "11 / 4.6ms" },
    { label: "Slice Thickness", value: "1.0mm" },
    { label: "Flip Angle", value: "15deg" },
    { label: "Sequence", value: "GRE-3D" },
    { label: "Magnet", value: "3.0 Tesla" }
  ];

  const studyDetails = [
    { label: "Study Date", value: "Oct 24, 2023 - 14:32" },
    { label: "Referring Physician", value: "Dr. Sarah Chen" },
    { label: "Accession", value: "AX-20398-11" },
    { label: "Clinical Note", value: "Follow-up review after post-op tumor resection." }
  ];

</script>

<section class="bg-background">
  <div class="flex min-h-[calc(100vh-6.5rem)] flex-col lg:h-[calc(100vh-6.5rem)] lg:flex-row">
    <div class="flex flex-1 flex-col overflow-hidden bg-surface-container-lowest">
      <div class="flex flex-col gap-3 border-b border-outline-variant/10 bg-surface-container px-4 py-4 md:flex-row md:items-center md:justify-between md:px-6">
        <div class="flex items-center gap-3">
          <span class="rounded bg-tertiary-container px-2 py-0.5 text-xs font-bold tracking-widest text-tertiary">PATIENT: JX-9802</span>
          <span class="text-xs font-medium text-on-surface-variant">Series: T1 Weighted Contrast Enhancement</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="flex rounded bg-surface-container-highest p-1 text-xs">
            <button class="rounded bg-tertiary px-2 py-1 text-on-tertiary shadow-sm" type="button">Multi-View</button>
            <button class="px-2 py-1 text-on-surface-variant hover:text-on-surface" type="button">Cinema</button>
            <button class="px-2 py-1 text-on-surface-variant hover:text-on-surface" type="button">Comparison</button>
          </div>
          <button aria-label="Fullscreen viewer" class="text-on-surface-variant transition-colors hover:text-primary" type="button">
            <span aria-hidden="true" class="material-symbols-outlined">fullscreen</span>
          </button>
        </div>
      </div>

      <div class="flex-1 grid grid-cols-2 grid-rows-2 gap-0.5 overflow-hidden bg-outline-variant/20">
        <NiivueViewer label="AXIAL" sliceType={SLICE_TYPE.AXIAL} sublabel="Z: 142mm">
          <div class="pointer-events-none absolute bottom-3 right-3 text-[10px] text-on-surface/30">R: 45.2 L: 22.1</div>
        </NiivueViewer>

        <NiivueViewer label="SAGITTAL" sliceType={SLICE_TYPE.SAGITTAL} sublabel="X: 12.5mm" />

        <NiivueViewer label="CORONAL" sliceType={SLICE_TYPE.CORONAL} sublabel="Y: -84.2mm" />

        <NiivueViewer accent={true} label="3D VOLUME" sliceType={SLICE_TYPE.RENDER} sublabel="Voxel: 0.5mm ISO">
          <div class="absolute bottom-4 left-1/2 flex -translate-x-1/2 gap-4 rounded-full border border-outline-variant/20 bg-surface-container-highest/80 px-4 py-2 backdrop-blur-md">
            {#each ["rotate_right", "zoom_in", "layers"] as viewerAction}
              <button aria-label={viewerAction} class="text-lg text-on-surface-variant hover:text-tertiary" type="button">
                <span aria-hidden="true" class="material-symbols-outlined">{viewerAction}</span>
              </button>
            {/each}
          </div>
        </NiivueViewer>
      </div>
    </div>

    <aside class="flex w-full flex-col overflow-hidden border-t border-outline-variant/10 bg-surface-container lg:w-80 lg:border-l lg:border-t-0">
      <div class="flex items-center justify-between border-b border-outline-variant/10 p-4">
        <h2 class="font-headline text-sm font-bold text-on-surface">Scan Information</h2>
        <button aria-label="More scan actions" class="text-sm text-on-surface-variant" type="button">
          <span aria-hidden="true" class="material-symbols-outlined">more_vert</span>
        </button>
      </div>

      <div class="custom-scrollbar flex-1 space-y-8 overflow-y-auto p-6">
        <InfoGridCard items={scanMetadata} title="SCAN METADATA" />
        <DetailListSection items={studyDetails} title="Study Details" />
      </div>

      <div class="border-t border-outline-variant/10 bg-surface-container-high/50 p-6">
        <button class="mb-3 w-full rounded-lg border border-outline-variant/30 bg-surface-container-highest px-4 py-2.5 text-sm font-semibold text-primary transition-all hover:bg-surface-container-highest/80" type="button">
          Export DICOM
        </button>
        <button class="w-full rounded-lg bg-tertiary px-4 py-2.5 text-sm font-bold text-on-tertiary shadow-lg shadow-tertiary/10" type="button">
          Finalize Report
        </button>
      </div>
    </aside>
  </div>
</section>
