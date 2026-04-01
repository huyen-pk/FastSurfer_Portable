import type { InferenceProgressEvent } from "../types/inference";
import type { ProcessInferenceRequest, ProcessInferenceResponse, Transport } from "./types";

function toErrorMessage(error: unknown, fallback: string): string {
  if (!error) return fallback;
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  return fallback;
}

async function postJson(url: string, payload: Record<string, unknown>): Promise<ProcessInferenceResponse> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  let body: ProcessInferenceResponse | null = null;
  try {
    body = await response.json() as ProcessInferenceResponse;
  } catch {
    body = null;
  }

  if (!response.ok) {
    const detail = body?.ackMessage || body?.ack_message || `HTTP ${response.status}`;
    throw new Error(String(detail));
  }

  return body ?? {};
}

export function createHttpsTransport(baseUrl = "http://127.0.0.1:8000"): Transport {
  const normalizedBaseUrl = baseUrl.replace(/\/$/, "");

  return {
    async processInference({ filePaths, folderPaths }: ProcessInferenceRequest): Promise<ProcessInferenceResponse> {
      try {
        return await postJson(`${normalizedBaseUrl}/process/run`, {
          file_paths: filePaths,
          folder_paths: folderPaths
        });
      } catch (error: unknown) {
        throw new Error(toErrorMessage(error, "Failed to run HTTPS processing."));
      }
    },
    async processInferenceWithProgress(
      { filePaths, folderPaths }: ProcessInferenceRequest,
      taskId: string,
      observer?: (event: InferenceProgressEvent) => void
    ): Promise<ProcessInferenceResponse> {
      observer?.({
        taskId,
        status: "started",
        message: "Processing started.",
        total: 0,
        completed: 0,
        progress: 5,
        currentPath: null,
        outputPath: null
      });

      try {
        const response = await postJson(`${normalizedBaseUrl}/process/run`, {
          file_paths: filePaths,
          folder_paths: folderPaths
        });
        observer?.({
          taskId,
          status: "completed",
          message: response?.ackMessage || response?.ack_message || "Processing completed.",
          total: Array.isArray(response?.results) ? response.results.length : 0,
          completed: Array.isArray(response?.results) ? response.results.length : 0,
          progress: 100,
          currentPath: null,
          outputPath: null
        });
        return response;
      } catch (error: unknown) {
        observer?.({
          taskId,
          status: "failed",
          message: toErrorMessage(error, "Failed to run HTTPS processing."),
          total: 0,
          completed: 0,
          progress: 0,
          currentPath: null,
          outputPath: null
        });
        throw new Error(toErrorMessage(error, "Failed to run HTTPS processing."));
      }
    },
    async cancelTask(_taskId: string): Promise<void> {
      throw new Error("Task cancellation is only supported in desktop IPC runtime.");
    },
    async shutdownForExit(): Promise<void> {
      return;
    },
    async openInFileManager(_path: string): Promise<void> {
      throw new Error("Opening files in the system file manager is only supported in desktop IPC runtime.");
    }
  };
}