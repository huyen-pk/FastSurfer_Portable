import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../../src/App.svelte";

const mocked = vi.hoisted(() => {
  let closeRequestedHandler: ((event: { preventDefault: () => void }) => void) | null = null;

  const openMock = vi.fn();
  const confirmMock = vi.fn();
  const processInferenceMock = vi.fn();
  const processInferenceWithProgressMock = vi.fn();
  const cancelTaskMock = vi.fn();
  const shutdownForExitMock = vi.fn();
  const openInFileManagerMock = vi.fn();

  const closeMock = vi.fn();
  const destroyMock = vi.fn();
  const onCloseRequestedMock = vi.fn((handler: (event: { preventDefault: () => void }) => void) => {
    closeRequestedHandler = handler;
    return Promise.resolve(() => {
      closeRequestedHandler = null;
    });
  });

  return {
    openMock,
    confirmMock,
    processInferenceMock,
    processInferenceWithProgressMock,
    cancelTaskMock,
    shutdownForExitMock,
    openInFileManagerMock,
    closeMock,
    destroyMock,
    onCloseRequestedMock,
    triggerCloseRequested: (event: { preventDefault: () => void }) => {
      closeRequestedHandler?.(event);
    }
  };
});

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: mocked.openMock,
  confirm: mocked.confirmMock
}));

vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({
    onCloseRequested: mocked.onCloseRequestedMock,
    close: mocked.closeMock,
    destroy: mocked.destroyMock
  })
}));

vi.mock("../../src/transport", () => ({
  createTransport: () => ({
    processInference: mocked.processInferenceMock,
    processInferenceWithProgress: mocked.processInferenceWithProgressMock,
    cancelTask: mocked.cancelTaskMock,
    shutdownForExit: mocked.shutdownForExitMock,
    openInFileManager: mocked.openInFileManagerMock
  })
}));

describe("app_close_behavior", () => {
  afterEach(() => {
    vi.clearAllMocks();
    delete (window as Window & { __TAURI__?: unknown }).__TAURI__;
  });

  it("close_request_without_running_tasks_should_shutdown_and_close", async () => {
    (window as Window & { __TAURI__?: unknown }).__TAURI__ = {};
    mocked.shutdownForExitMock.mockResolvedValue(undefined);

    render(App);

    const preventDefault = vi.fn();
    mocked.triggerCloseRequested({ preventDefault });

    await waitFor(() => expect(preventDefault).toHaveBeenCalledTimes(1));
    expect(mocked.confirmMock).not.toHaveBeenCalled();
    await waitFor(() => expect(mocked.shutdownForExitMock).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mocked.closeMock).toHaveBeenCalledTimes(1));
  });

  it("close_request_with_running_task_should_shutdown_and_close", async () => {
    (window as Window & { __TAURI__?: unknown }).__TAURI__ = {};

    mocked.openMock.mockResolvedValueOnce(["/input/sub01.nii.gz"]);
    mocked.confirmMock.mockResolvedValueOnce(true);
    mocked.shutdownForExitMock.mockResolvedValue(undefined);

    let resolveProcess: ((value: unknown) => void) | null = null;
    mocked.processInferenceWithProgressMock.mockImplementation(() => new Promise((resolve) => {
      resolveProcess = resolve;
    }));

    render(App);

    await fireEvent.click(screen.getByRole("button", { name: "Pick Files" }));
    await fireEvent.click(screen.getByRole("button", { name: "Process" }));

    await waitFor(() => expect(mocked.processInferenceWithProgressMock).toHaveBeenCalledTimes(1));

    const preventDefault = vi.fn();
    mocked.triggerCloseRequested({ preventDefault });

    await waitFor(() => expect(preventDefault).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mocked.confirmMock).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mocked.shutdownForExitMock).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(mocked.closeMock).toHaveBeenCalledTimes(1));

    resolveProcess?.({ ackMessage: "done", results: [] });
  });
});
