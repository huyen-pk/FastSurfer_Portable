export interface InferenceArtifacts {
  brainmaskPath?: string;
  asegPath?: string;
}

export interface InferenceQc {
  passed?: boolean;
  message?: string;
}

export interface InferenceResult {
  inputPath: string;
  outputPath: string;
  outputFilename: string;
  runResult: string;
  artifacts?: InferenceArtifacts;
  qc?: InferenceQc;
}

export interface RawInferenceArtifacts {
  brainmaskPath?: string;
  brainmask_path?: string;
  asegPath?: string;
  aseg_path?: string;
}

export interface RawInferenceQc {
  passed?: boolean;
  message?: string;
}

export interface RawInferenceResult {
  inputPath?: string;
  input_path?: string;
  outputPath?: string;
  output_path?: string;
  outputFilename?: string;
  output_filename?: string;
  runResult?: string;
  run_result?: string;
  artifacts?: RawInferenceArtifacts;
  qc?: RawInferenceQc;
}

export interface ProcessInferenceResponse {
  ackMessage?: string;
  ack_message?: string;
  requestedPaths?: string[];
  requested_paths?: string[];
  resultDirectories?: string[];
  result_directories?: string[];
  qcSummary?: string;
  qc_summary?: string;
  results?: RawInferenceResult[];
}

export interface InferenceProgressEvent {
  taskId: string;
  status: "started" | "item_progress" | "item_completed" | "completed" | "failed" | "cancelled";
  message: string;
  total: number;
  completed: number;
  progress: number;
  currentPath?: string | null;
  outputPath?: string | null;
}