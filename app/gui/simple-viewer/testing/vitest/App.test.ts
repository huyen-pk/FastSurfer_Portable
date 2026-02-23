import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../../src/App.svelte";

const { openMock, processInferenceMock, openInFileManagerMock } = vi.hoisted(() => ({
  openMock: vi.fn(),
  processInferenceMock: vi.fn(),
  openInFileManagerMock: vi.fn()
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: openMock
}));

vi.mock("../../src/transport", () => ({
  createTransport: () => ({
    processInference: processInferenceMock,
    openInFileManager: openInFileManagerMock
  })
}));

describe("app_behavior", () => {
  afterEach(() => {
    vi.clearAllMocks();
    delete (window as Window & { __TAURI__?: unknown }).__TAURI__;
  });

  it("pick_files_and_folders_should_render_selected_paths", async () => {
    (window as Window & { __TAURI__?: unknown }).__TAURI__ = {};
    openMock
      .mockResolvedValueOnce(["/data/sub01.nii.gz"])
      .mockResolvedValueOnce(["/data/folderA"]);

    render(App);

    await fireEvent.click(screen.getByRole("button", { name: "Pick Files/Folders" }));

    expect(openMock).toHaveBeenCalledTimes(2);
    expect(await screen.findByTitle("/data/sub01.nii.gz")).toBeInTheDocument();
    expect(await screen.findByTitle("/data/folderA")).toBeInTheDocument();
  });

  it("start_processing_should_render_acknowledgement_and_processed_result", async () => {
    (window as Window & { __TAURI__?: unknown }).__TAURI__ = {};
    openMock
      .mockResolvedValueOnce(["/input/sub01.nii.gz"])
      .mockResolvedValueOnce(["/input/folderA"]);
    processInferenceMock.mockResolvedValue({
      ackMessage: "Done",
      results: [
        {
          inputPath: "/input/sub01.nii.gz",
          outputPath: "/output/sub01.mgz",
          outputFilename: "sub01.mgz",
          runResult: "ok"
        }
      ]
    });

    render(App);

    await fireEvent.click(screen.getByRole("button", { name: "Pick Files/Folders" }));
    await fireEvent.click(screen.getByRole("button", { name: "Start Processing" }));

    await waitFor(() => expect(processInferenceMock).toHaveBeenCalledTimes(1));
    expect(processInferenceMock).toHaveBeenCalledWith({
      filePaths: ["/input/sub01.nii.gz"],
      folderPaths: ["/input/folderA"]
    });
    expect(await screen.findByText("Done")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "sub01.nii.gz" })).toBeInTheDocument();
  });
});