import { fireEvent, render, screen } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import App from "../../src/App.svelte";
describe("Damadian_mri_navigation", () => {
  it("render_should_show_dashboard_by_default", () => {
    render(App);

    expect(screen.getByRole("heading", { name: "Clinical Overview" })).toBeInTheDocument();
    expect(screen.getByText("Recent MRI Studies")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /Dashboard$/ })[0]).toHaveAttribute("aria-current", "page");
  });

  it("analytics_button_click_should_switch_to_analytics_view", async () => {
    render(App);

    await fireEvent.click(screen.getAllByRole("button", { name: /Analytics$/ })[0]);

    expect(screen.getByRole("heading", { name: "Comparative Study Analysis" })).toBeInTheDocument();
    expect(screen.getByText("Study Comparison Matrix")).toBeInTheDocument();
  });

  it("scan_viewer_button_click_should_render_niivue_container", async () => {
    render(App);

    await fireEvent.click(screen.getAllByRole("button", { name: /Scan Viewer$/ })[0]);

    expect(screen.getAllByLabelText("BRATS segmentation viewer")).toHaveLength(4);
    expect(screen.getAllByTestId("niivue-canvas")).toHaveLength(4);
    expect(screen.getByText("AXIAL")).toBeInTheDocument();
    expect(screen.getByText("SAGITTAL")).toBeInTheDocument();
    expect(screen.getByText("CORONAL")).toBeInTheDocument();
    expect(screen.getByText("3D VOLUME")).toBeInTheDocument();
  });
});
