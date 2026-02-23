import { invoke } from "@tauri-apps/api/core";
import type { ProcessInferenceRequest, ProcessInferenceResponse, Transport } from "./types";

export function createIpcTransport(): Transport {
  return {
    async processInference({ filePaths, folderPaths }: ProcessInferenceRequest): Promise<ProcessInferenceResponse> {
      return invoke<ProcessInferenceResponse>("run_fastsurfer_inference", {
        filePaths,
        folderPaths
      });
    },
    async openInFileManager(path: string): Promise<void> {
      await invoke<void>("open_result_in_file_manager", {
        resultPath: path
      });
    }
  };
}