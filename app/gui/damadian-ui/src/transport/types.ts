import type { InferenceProgressEvent, ProcessInferenceResponse } from "../types/inference";
export type { InferenceProgressEvent, ProcessInferenceResponse };

export interface ProcessInferenceRequest {
  filePaths: string[];
  folderPaths: string[];
}

export interface Transport {
  processInference(request: ProcessInferenceRequest): Promise<ProcessInferenceResponse>;
  processInferenceWithProgress(
    request: ProcessInferenceRequest,
    taskId: string,
    observer?: (event: InferenceProgressEvent) => void
  ): Promise<ProcessInferenceResponse>;
  cancelTask(taskId: string): Promise<void>;
  shutdownForExit(): Promise<void>;
  openInFileManager(path: string): Promise<void>;
}