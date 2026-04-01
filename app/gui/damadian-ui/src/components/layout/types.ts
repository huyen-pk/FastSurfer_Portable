export type ViewId = "dashboard" | "scanViewer" | "pipelines" | "analytics";

export type NavigationItem = {
  id: ViewId;
  label: string;
  icon: string;
  mobileLabel: string;
};

export type FooterState = {
  queueLabel: string;
  activeLabel: string;
  logsLabel: string;
  rightLabel: string;
  meterWidth: string;
  indicatorClass: string;
};
