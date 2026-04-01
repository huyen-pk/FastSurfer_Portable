<script lang="ts">
  import AnalyticsStatCard from "../analytics/AnalyticsStatCard.svelte";
  import BarComparisonChart from "../analytics/BarComparisonChart.svelte";
  import ComparisonTable from "../analytics/ComparisonTable.svelte";
  import SummaryMetricGrid from "../analytics/SummaryMetricGrid.svelte";
  import InsightMediaCard from "../shared/InsightMediaCard.svelte";

  const barData = [
    { label: "Hippocampus", baseline: 75, current: 82 },
    { label: "Amygdala", baseline: 60, current: 58 },
    { label: "Thalamus", baseline: 90, current: 94 },
    { label: "Caudate", baseline: 45, current: 42 },
    { label: "Putamen", baseline: 68, current: 72 }
  ];

  const comparisonRows = [
    {
      studyId: "MRI-2024-X01",
      metric: "Whole Brain Volume",
      baseline: "1,240 cm3",
      processed: "1,215 cm3",
      variance: "-2.01%",
      varianceClass: "text-error",
      status: "Critical",
      statusClass: "bg-secondary-container text-on-secondary-container"
    },
    {
      studyId: "MRI-2024-X04",
      metric: "Ventricular Width",
      baseline: "12.4 mm",
      processed: "12.6 mm",
      variance: "+1.61%",
      varianceClass: "text-tertiary",
      status: "Stable",
      statusClass: "bg-surface-container-highest text-on-surface-variant"
    },
    {
      studyId: "MRI-2024-X09",
      metric: "Cortical Thickness",
      baseline: "2.54 mm",
      processed: "2.51 mm",
      variance: "-1.18%",
      varianceClass: "text-error",
      status: "Review",
      statusClass: "bg-secondary-container text-on-secondary-container"
    },
    {
      studyId: "MRI-2024-Y12",
      metric: "White Matter Integrity",
      baseline: "0.84 FA",
      processed: "0.88 FA",
      variance: "+4.76%",
      varianceClass: "text-tertiary",
      status: "Improved",
      statusClass: "bg-surface-container-highest text-on-surface-variant"
    }
  ];

  const summaryMetrics = [
    { label: "Standard Deviation", value: "sigma = 0.042" },
    { label: "P-Value Significance", value: "p < 0.001" },
    { label: "Model Version", value: "v4.2.8-stable" },
    { label: "Verification Hash", value: "8f2g-9k1L-pZ92", valueClass: "font-mono text-sm text-tertiary" }
  ];
</script>

<section class="min-h-[calc(100vh-6.5rem)] bg-background px-4 py-8 md:px-8">
  <div class="mb-10 flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
    <div>
      <nav class="mb-2 flex items-center gap-2 text-xs text-on-surface-variant">
        <span>Analytics</span>
        <span aria-hidden="true" class="material-symbols-outlined text-[10px]">chevron_right</span>
        <span class="text-primary">Comparative Results</span>
      </nav>
      <h1 class="font-headline text-4xl font-extrabold tracking-tight text-on-background">Comparative Study Analysis</h1>
      <p class="mt-2 max-w-2xl text-on-surface-variant">Visualizing volumetric neural data across Project Alpha Cohort B (N=142). Quantitative metrics verified against standardized atlas.</p>
    </div>

    <button class="flex items-center gap-2 rounded-lg bg-gradient-to-tr from-primary to-primary-fixed-dim px-6 py-3 font-bold text-on-primary shadow-lg transition-all hover:shadow-primary/20" type="button">
      <span aria-hidden="true" class="material-symbols-outlined">picture_as_pdf</span>
      Generate PDF Report
    </button>
  </div>

  <div class="grid grid-cols-12 gap-6">
    <BarComparisonChart bars={barData} />

    <div class="col-span-12 flex flex-col gap-6 lg:col-span-4">
      <AnalyticsStatCard
        cardClass="bg-surface-container-high"
        description="Iterative refinement active. 0.2ms latency in volumetric mapping."
        icon="verified"
        label="Processing Confidence"
        value="98.4%"
        valueClass="text-tertiary"
      />
      <AnalyticsStatCard
        badge="+12.3%"
        badgeCaption="vs previous month"
        label="Total Studies Analyzed"
        value="1,248"
      />
    </div>

    <ComparisonTable rows={comparisonRows} />

    <InsightMediaCard
      containerClass="col-span-12 bg-surface-container lg:col-span-6"
      description="Global cohort average baseline vs Subject Alpha-01"
      imageAlt="Brain MRI heatmap"
      imageClass="h-full w-full object-cover opacity-60 transition-transform duration-700 group-hover:scale-105"
      imageSrc="https://lh3.googleusercontent.com/aida-public/AB6AXuBP5TY2ckWzqPqw0lltnSXTOSlY-KWI0eQ3DDIO7NtEtP3uCkj1F7EnRdLKGSRYIS5DMebxWmFnpaMdkIlOY7NjESmFKY398oai1yl46ZXFBPtDaKw2iVRG62ddfCMQToFzpHkZgpzw_iPEcLvbCnac3h6m1rlytjOzm5_aUxKak44YbP5Ie1iY-p2ui2gjhvt8G0haHFzaRt8hLT_DqRelzEi4ml6Fq-fujDbqIKP9WCgZruny2-HkP34EnJQTj9k2ZyiqfEpT4EBH"
      overlayClass="absolute inset-0 bg-gradient-to-t from-surface-container via-transparent to-transparent"
      title="Neural Density Mapping"
    >
      <div class="absolute bottom-6 left-6">
        <h3 class="font-headline text-xl font-bold text-on-background">Neural Density Mapping</h3>
        <p class="text-xs text-on-surface-variant">Global cohort average baseline vs Subject Alpha-01</p>
      </div>
    </InsightMediaCard>

    <SummaryMetricGrid metrics={summaryMetrics} />
  </div>
</section>
