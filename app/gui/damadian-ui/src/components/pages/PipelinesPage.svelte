<script lang="ts">
  import PipelineStepCard from "../pipelines/PipelineStepCard.svelte";

  const pipelineSteps = [
    {
      title: "DICOM Input",
      subtitle: "Source: Research_Repository_A1",
      icon: "upload_file",
      accentClass: "border-primary-fixed",
      iconClass: "text-primary",
      iconWrapperClass: "bg-primary/10",
      badge: "Input Data",
      badgeClass: "bg-secondary-container text-on-secondary-container"
    },
    {
      title: "Motion Correction",
      subtitle: "FSL McFlirt Algorithm • 6 DOF (Standard)",
      icon: "unfold_more",
      accentClass: "border-tertiary",
      iconClass: "text-tertiary",
      iconWrapperClass: "bg-tertiary/10",
      badge: "Optimized",
      badgeClass: "bg-tertiary/10 text-tertiary"
    },
    {
      title: "Skull Stripping",
      subtitle: "BET (Brain Extraction Tool) • Robust Estimation",
      icon: "psychology",
      accentClass: "border-outline-variant/30",
      iconClass: "text-primary",
      iconWrapperClass: "bg-surface-variant",
      badge: "",
      badgeClass: ""
    },
    {
      title: "Volume Quantification",
      subtitle: "Segmentation-based T1 measurement",
      icon: "bar_chart_4_bars",
      accentClass: "border-outline-variant/30",
      iconClass: "text-primary",
      iconWrapperClass: "bg-surface-variant",
      badge: "",
      badgeClass: ""
    },
    {
      title: "Report Generation",
      subtitle: "Clinical PDF and raw CSV output",
      icon: "description",
      accentClass: "border-secondary",
      iconClass: "text-secondary",
      iconWrapperClass: "bg-secondary/10",
      badge: "Endpoint",
      badgeClass: "bg-secondary/10 text-secondary"
    }
  ];
</script>

<section class="min-h-[calc(100vh-6.5rem)] bg-background px-4 py-8 md:px-8">
  <header class="mb-10 flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
    <div>
      <p class="mb-2 text-xs font-semibold uppercase tracking-[0.35em] text-primary/60">Pipelines</p>
      <h1 class="font-headline text-3xl font-extrabold text-on-background">Automated Pipeline Overview</h1>
      <p class="mt-2 text-primary">Sequence #4429 • Optimized Research Template</p>
    </div>

    <div class="flex flex-wrap gap-4">
      <button class="rounded-lg border border-outline-variant/30 px-6 py-2.5 text-on-surface transition-all hover:bg-surface-container-high" type="button">
        Duplicate Template
      </button>
      <button class="flex items-center gap-2 rounded-lg bg-tertiary px-6 py-2.5 font-bold text-on-tertiary shadow-lg shadow-tertiary/20 transition-all hover:brightness-110" type="button">
        <span aria-hidden="true" class="material-symbols-outlined text-sm">play_arrow</span>
        Execute Sequence
      </button>
    </div>
  </header>

  <div class="mx-auto max-w-4xl">
    <div class="space-y-0">
      {#each pipelineSteps as step, index}
        <PipelineStepCard {...step} highlighted={index === 1} showConnector={index < pipelineSteps.length - 1} />
      {/each}

      <button class="group mt-12 flex w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-outline-variant/20 p-8 transition-all hover:bg-surface-container-low" type="button">
        <div class="mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-surface-container-highest transition-colors group-hover:bg-tertiary">
          <span aria-hidden="true" class="material-symbols-outlined text-tertiary transition-colors group-hover:text-on-tertiary">add</span>
        </div>
        <p class="font-bold text-primary">Add Step to Sequence</p>
        <p class="text-xs text-primary/50">Choose from pre-configured clinical modules</p>
      </button>
    </div>
  </div>
</section>
