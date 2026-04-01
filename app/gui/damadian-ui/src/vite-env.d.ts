/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_FASTSURFER_TARGET?: string;
  readonly VITE_FASTSURFER_BACKEND_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}