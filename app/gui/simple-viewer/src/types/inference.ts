export interface InferenceResult {
  inputPath: string;
  outputPath: string;
  outputFilename: string;
  runResult: string;
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
}

export interface ProcessInferenceResponse {
  ackMessage?: string;
  ack_message?: string;
  results?: RawInferenceResult[];
}