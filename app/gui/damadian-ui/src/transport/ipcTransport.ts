import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { InferenceProgressEvent } from "../types/inference";
import { INFERENCE_PROGRESS_EVENT } from "./events";
import type { ProcessInferenceRequest, ProcessInferenceResponse, Transport } from "./types";

export function createIpcTransport(): Transport {
  return {
    async processInference({ filePaths, folderPaths }: ProcessInferenceRequest): Promise<ProcessInferenceResponse> {
      console.debug("[trace][frontend-ipc] invoke run_fastsurfer_inference_with_progress (legacy)", {
        filePaths: filePaths.length,
        folderPaths: folderPaths.length
      });
      // Legacy callers expect a simple invoke; reuse the progress path with a temporary taskId
      const taskId = `legacy-${Date.now()}`;
      return invoke<ProcessInferenceResponse>("run_fastsurfer_inference_with_progress", {
        taskId,
        filePaths,
        folderPaths
      });
    },
    async processInferenceWithProgress(
      { filePaths, folderPaths }: ProcessInferenceRequest,
      taskId: string,
      observer?: (event: InferenceProgressEvent) => void
    ): Promise<ProcessInferenceResponse> {
      console.debug("[trace][frontend-ipc] subscribe progress + invoke", {
        taskId,
        filePaths: filePaths.length,
        folderPaths: folderPaths.length
      });
      const unlisten = await listen<InferenceProgressEvent>(INFERENCE_PROGRESS_EVENT, (event) => {
        if (event.payload?.taskId === taskId && observer) {
          console.debug("[trace][frontend-ipc] progress event", event.payload);
          observer(event.payload);
        }
      });

      try {
        const response = await invoke<ProcessInferenceResponse>("run_fastsurfer_inference_with_progress", {
          taskId,
          filePaths,
          folderPaths
        });
        console.debug("[trace][frontend-ipc] invoke completed", { taskId });
        return response;
      } finally {
        unlisten();
      }
    },
    async cancelTask(taskId: string): Promise<void> {
      await invoke<void>("cancel_fastsurfer_task", {
        taskId
      });
    },
    async shutdownForExit(): Promise<void> {
      await invoke<void>("shutdown_backend_for_exit");
    },
    async openInFileManager(path: string): Promise<void> {
      await invoke<void>("open_result_in_file_manager", {
        resultPath: path
      });
    }
  };
}