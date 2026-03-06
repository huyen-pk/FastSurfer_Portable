import { createIpcTransport } from "./ipcTransport";
import { createHttpsTransport } from "./httpsTransport";
import type { Transport } from "./types";

function getRuntimePlatform(): "desktop" | "web" {
  return (typeof window !== "undefined" && "__TAURI__" in window) ? "desktop" : "web";
}

export function createTransport(): Transport {
  const runtimeEnvironment = getRuntimePlatform();
  const selectedProtocol = runtimeEnvironment === "desktop" ? "ipc" : "https";

  if (selectedProtocol === "https") {
    const baseUrl = import.meta.env.VITE_FASTSURFER_BACKEND_URL || "http://127.0.0.1:8000";
    return createHttpsTransport(baseUrl);
  }

  return createIpcTransport();
}