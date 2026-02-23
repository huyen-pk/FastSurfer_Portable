import type { ProcessInferenceResponse } from "../types/inference";

export interface ProcessInferenceRequest {
  filePaths: string[];
  folderPaths: string[];
}

export interface Transport {
  processInference(request: ProcessInferenceRequest): Promise<ProcessInferenceResponse>;
  openInFileManager(path: string): Promise<void>;
}