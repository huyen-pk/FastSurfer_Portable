<script lang="ts">
  import AppHeader from "./components/layout/AppHeader.svelte";
  import AppSidebar from "./components/layout/AppSidebar.svelte";
  import MobileNav from "./components/layout/MobileNav.svelte";
  import StatusFooter from "./components/layout/StatusFooter.svelte";
  import AnalyticsPage from "./components/pages/AnalyticsPage.svelte";
  import DashboardPage from "./components/pages/DashboardPage.svelte";
  import PipelinesPage from "./components/pages/PipelinesPage.svelte";
  import ScanViewerPage from "./components/pages/ScanViewerPage.svelte";
  import type { FooterState, NavigationItem, ViewId } from "./components/layout/types";

  const navigationItems: NavigationItem[] = [
    { id: "dashboard", label: "Dashboard", icon: "dashboard", mobileLabel: "Dashboard" },
    { id: "scanViewer", label: "Scan Viewer", icon: "biotech", mobileLabel: "Viewer" },
    { id: "pipelines", label: "Pipelines", icon: "account_tree", mobileLabel: "Pipelines" },
    { id: "analytics", label: "Analytics", icon: "analytics", mobileLabel: "Analytics" }
  ];

  const searchPlaceholders: Record<ViewId, string> = {
    dashboard: "Search clinical records...",
    scanViewer: "Search Patient ID...",
    pipelines: "Search pipelines...",
    analytics: "Search studies..."
  };

  const footerStates: Record<ViewId, FooterState> = {
    dashboard: {
      queueLabel: "Queue (18)",
      activeLabel: "Active (4)",
      logsLabel: "Logs",
      rightLabel: "LATENCY: 14MS",
      meterWidth: "66%",
      indicatorClass: "bg-tertiary"
    },
    scanViewer: {
      queueLabel: "Queue (2)",
      activeLabel: "Active",
      logsLabel: "Logs",
      rightLabel: "GPU LOAD: 42%",
      meterWidth: "42%",
      indicatorClass: "bg-tertiary"
    },
    pipelines: {
      queueLabel: "Queue: 12 Tasks",
      activeLabel: "Active: 02",
      logsLabel: "Logs",
      rightLabel: "GPU-Cluster: Optimal",
      meterWidth: "100%",
      indicatorClass: "bg-emerald-500"
    },
    analytics: {
      queueLabel: "Queue: 4",
      activeLabel: "Active: 12",
      logsLabel: "Logs: Real-time",
      rightLabel: "75% CPU LOAD",
      meterWidth: "75%",
      indicatorClass: "bg-tertiary"
    }
  };

  let activeView: ViewId = "dashboard";

  $: activeSearchPlaceholder = searchPlaceholders[activeView];
  $: activeFooter = footerStates[activeView];

  function activateView(viewId: ViewId): void {
    activeView = viewId;
  }
</script>

<svelte:head>
  <title>Damadian Imaging</title>
</svelte:head>

<div class="min-h-screen bg-background text-on-background font-body selection:bg-tertiary/30">
  <AppHeader onHome={() => activateView("dashboard")} searchPlaceholder={activeSearchPlaceholder} />
  <AppSidebar activeView={activeView} items={navigationItems} onSelect={activateView} />

  <main class="pb-20 pt-16 lg:ml-64">
    <MobileNav activeView={activeView} items={navigationItems} onSelect={activateView} searchPlaceholder={activeSearchPlaceholder} />

    {#if activeView === "dashboard"}
      <DashboardPage />
    {:else if activeView === "scanViewer"}
      <ScanViewerPage />
    {:else if activeView === "pipelines"}
      <PipelinesPage />
    {:else}
      <AnalyticsPage />
    {/if}
  </main>

  <StatusFooter activeView={activeView} footerState={activeFooter} />
</div>
