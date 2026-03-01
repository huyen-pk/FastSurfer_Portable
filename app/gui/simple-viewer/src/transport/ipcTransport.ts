import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { InferenceProgressEvent } from "../types/inference";
import type { ProcessInferenceRequest, ProcessInferenceResponse, Transport } from "./types";

export function createIpcTransport(): Transport {
  return {
    async processInference({ filePaths, folderPaths }: ProcessInferenceRequest): Promise<ProcessInferenceResponse> {
      return invoke<ProcessInferenceResponse>("run_fastsurfer_inference", {
        filePaths,
        folderPaths
      });
    },
    async processInferenceWithProgress(
      { filePaths, folderPaths }: ProcessInferenceRequest,
      taskId: string,
      observer?: (event: InferenceProgressEvent) => void
    ): Promise<ProcessInferenceResponse> {
      const unlisten = await listen<InferenceProgressEvent>("fastsurfer://inference-progress", (event) => {
        if (event.payload?.taskId === taskId && observer) {
          observer(event.payload);
        }
      });

      try {
        return await invoke<ProcessInferenceResponse>("run_fastsurfer_inference_with_progress", {
          taskId,
          filePaths,
          folderPaths
        });
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