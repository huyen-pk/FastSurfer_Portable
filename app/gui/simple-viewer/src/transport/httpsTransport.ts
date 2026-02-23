import { invoke } from "@tauri-apps/api/core";
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
    async openInFileManager(path: string): Promise<void> {
      await invoke<void>("open_result_in_file_manager", {
        resultPath: path
      });
    }
  };
}