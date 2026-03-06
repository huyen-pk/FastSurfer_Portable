import { fireEvent, render, screen } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ProcessedResultViewer from "../../src/components/ProcessedResultViewer.svelte";
import ProcessedResultViewerHarness from "./ProcessedResultViewerHarness.svelte";

const { attachToCanvasMock, loadVolumesMock } = vi.hoisted(() => ({
  attachToCanvasMock: vi.fn(),
  loadVolumesMock: vi.fn()
}));

vi.mock("@niivue/niivue", () => ({
  Niivue: vi.fn().mockImplementation(() => ({
    attachToCanvas: attachToCanvasMock,
    loadVolumes: loadVolumesMock
  }))
}));

describe("processed_result_viewer_behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("mount_should_render_output_metadata", () => {
    render(ProcessedResultViewer, {
      result: {
        inputPath: "/input/sub01.nii.gz",
        outputPath: "/output/sub01.mgz",
        outputFilename: "sub01.mgz",
        runResult: "ok"
      }
    });

    expect(screen.getByText("sub01.mgz")).toBeInTheDocument();
    expect(screen.getByText("/output/sub01.mgz")).toBeInTheDocument();
  });

  it("mount_should_attach_canvas_and_load_volume", () => {
    render(ProcessedResultViewer, {
      result: {
        inputPath: "/input/sub01.nii.gz",
        outputPath: "/output/sub01.mgz",
        outputFilename: "sub01.mgz",
        runResult: "ok"
      }
    });

    expect(attachToCanvasMock).toHaveBeenCalledTimes(1);
    expect(loadVolumesMock).toHaveBeenCalledTimes(1);
  });

  it("close_button_click_should_emit_close_event", async () => {
    render(ProcessedResultViewerHarness);

    await fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(screen.getByTestId("did-close")).toBeInTheDocument();
  });
});