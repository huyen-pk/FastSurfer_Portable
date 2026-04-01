<script lang="ts">
  import ActivityTimeline from "../dashboard/ActivityTimeline.svelte";
  import MetricCard from "../dashboard/MetricCard.svelte";
  import StudiesTable from "../dashboard/StudiesTable.svelte";
  import InsightMediaCard from "../shared/InsightMediaCard.svelte";
  import PageSectionHeader from "../shared/PageSectionHeader.svelte";

  const metrics = [
    {
      icon: "layers",
      iconWrapperClass: "bg-primary-container",
      iconClass: "text-primary",
      label: "MTD Scans",
      value: "1,284",
      description: "+12.5% from last month",
      descriptionClass: "text-tertiary"
    },
    {
      icon: "check_circle",
      iconWrapperClass: "bg-tertiary-container",
      iconClass: "text-tertiary",
      label: "Avg Efficiency",
      value: "99.4%",
      description: "Across 14 active compute nodes",
      descriptionClass: "text-on-surface-variant/60"
    },
    {
      icon: "hourglass_empty",
      iconWrapperClass: "bg-secondary-container",
      iconClass: "text-secondary",
      label: "Queue Status",
      value: "18",
      description: "Tasks awaiting prioritization",
      descriptionClass: "text-secondary"
    }
  ];

  const recentStudies = [
    {
      patientId: "PX-8821",
      scanName: "T1-Weighted Brain",
      protocol: "NEURO_3D",
      status: "Processing...",
      statusIcon: "pulse",
      statusClass: "text-tertiary",
      time: "09:42 AM"
    },
    {
      patientId: "PX-8819",
      scanName: "FLAIR Scan",
      protocol: "STROKE_FAST",
      status: "Completed",
      statusIcon: "check_circle",
      statusClass: "text-emerald-400",
      time: "08:15 AM"
    },
    {
      patientId: "PX-8755",
      scanName: "Cardiac CINE",
      protocol: "CARD_V1",
      status: "Error",
      statusIcon: "error",
      statusClass: "text-error",
      time: "Yesterday"
    }
  ];

  const timelineEvents = [
    {
      title: "Report Generated",
      description: "Diagnostic report for study #PX-8819 has been finalized and signed by Dr. Miller.",
      age: "24m ago",
      dotClass: "bg-tertiary"
    },
    {
      title: "Model Update",
      description: "Segmentation engine v2.4.1 deployed to Node Cluster Alpha.",
      age: "1h ago",
      dotClass: "bg-primary"
    },
    {
      title: "Node Failure",
      description: "Compute Node 04 unresponsive. Tasks rerouted to standby servers.",
      age: "3h ago",
      dotClass: "bg-error"
    }
  ];
</script>

<section class="min-h-[calc(100vh-6.5rem)] bg-surface px-4 py-8 md:px-8">
  <PageSectionHeader
    description="Real-time monitoring of neuroimaging computational pipelines."
    eyebrow="Dashboard"
    title="Clinical Overview"
  />

  <div class="grid grid-cols-12 gap-6">
    {#each metrics as metric}
      <MetricCard {...metric} />
    {/each}

    <StudiesTable studies={recentStudies} />

    <div class="col-span-12 space-y-6 lg:col-span-4">
      <ActivityTimeline events={timelineEvents} />
      <InsightMediaCard
        badge="LIVE MONITOR"
        description="Study #PX-8821 • Slice 128/256"
        imageAlt="MRI scan preview"
        imageSrc="https://lh3.googleusercontent.com/aida-public/AB6AXuCcRNh1L4lhoJXBZJ3tmVA_v14ZAoKAv7phvCAgnYs3arSmjQwx3sz5thSAbHOSnI-tXsSOcibEG8O-hnAxTddDn45K6t9aIdzMGNfz0ffMNsEX3wdNuO--9LDsIMjOuklrvg6Hh1ITEI6u-4J8jqvWM1ygqMB12bpBx2ln4Kw3MAFqNFccVcIKZA3ygNGFHzx4WU71jHHvtYScbP-lz7RYiSUQNj0rkNupEjqxvqwehBQI3LAJAn_oJb9J0UGSdHVvq5k2fKiJZA3W"
        title="Segmental Analysis In Progress"
      />
    </div>
  </div>
</section>
