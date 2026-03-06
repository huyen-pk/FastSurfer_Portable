from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

from inference_service import FastSurferInferenceService


class FastSurferIPCServer:
	def __init__(self):
		self.inference_service = FastSurferInferenceService()
		self._running = True

	def _predict_from_path(self, params: dict[str, Any]) -> dict[str, Any]:
		input_path_raw = params.get("input_path")
		if not input_path_raw:
			raise ValueError("Missing required param: input_path")

		input_path = Path(str(input_path_raw)).expanduser().resolve()
		if not input_path.exists():
			raise FileNotFoundError(f"input_path does not exist: {input_path}")

		output_dir_raw = params.get("output_dir")
		if output_dir_raw:
			output_dir = Path(str(output_dir_raw)).expanduser().resolve()
			output_dir.mkdir(parents=True, exist_ok=True)
		else:
			output_dir = Path(tempfile.mkdtemp(prefix="fastsurfer_ipc_out_"))

		task_id = str(params.get("task_id") or "")

		def progress_callback(progress: int, message: str) -> None:
			payload: dict[str, Any] = {
				"event": "progress",
				"progress": max(0, min(100, int(progress))),
				"message": str(message or "Processing MRI"),
			}
			if task_id:
				payload["task_id"] = task_id
			sys.stdout.write(json.dumps(payload) + "\n")
			sys.stdout.flush()

		return self.inference_service.predict_from_path(
			input_path=input_path,
			output_dir=output_dir,
			return_base64=bool(params.get("return_base64", False)),
			progress_callback=progress_callback,
		)

	def _predict_from_bytes(self, params: dict[str, Any]) -> dict[str, Any]:
		file_b64 = params.get("file_base64")
		if not file_b64:
			raise ValueError("Missing required param: file_base64")

		file_name = str(params.get("file_name") or "input.nii.gz")
		return self.inference_service.predict_from_bytes(file_b64, file_name=file_name)

	def _start_predict_batch(self, params: dict[str, Any]) -> dict[str, Any]:
		file_paths = params.get("file_paths") or []
		folder_paths = params.get("folder_paths") or []
		if not isinstance(file_paths, list) or not isinstance(folder_paths, list):
			raise ValueError("file_paths and folder_paths must be arrays")

		requested_paths = self.inference_service.resolve_input_paths(
			file_paths=[str(p) for p in file_paths],
			folder_paths=[str(p) for p in folder_paths],
		)
		if not requested_paths:
			raise ValueError("No valid input image files selected (.nii, .nii.gz, .mgz, .mgh).")

		requested_as_str = [str(path) for path in requested_paths]
		return {
			"ack_message": f"Processing started for {len(requested_as_str)} path(s).",
			"requested_paths": requested_as_str,
		}

	def _predict_batch(self, params: dict[str, Any]) -> dict[str, Any]:
		file_paths = params.get("file_paths") or []
		folder_paths = params.get("folder_paths") or []
		if not isinstance(file_paths, list) or not isinstance(folder_paths, list):
			raise ValueError("file_paths and folder_paths must be arrays")

		return self.inference_service.predict_batch_from_paths(
			file_paths=[str(p) for p in file_paths],
			folder_paths=[str(p) for p in folder_paths],
		)

	def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
		request_id = request.get("id")
		method = request.get("method")
		params = request.get("params") or {}

		if not isinstance(params, dict):
			raise ValueError("params must be an object")

		if method == "health":
			return {"id": request_id, "ok": True, "result": {"status": "ok"}}

		if method == "predict":
			result = self._predict_from_path(params)
			return {"id": request_id, "ok": True, "result": result}

		if method == "predict_bytes":
			result = self._predict_from_bytes(params)
			return {"id": request_id, "ok": True, "result": result}

		if method == "start_predict_batch":
			result = self._start_predict_batch(params)
			return {"id": request_id, "ok": True, "result": result}

		if method == "predict_batch":
			result = self._predict_batch(params)
			return {"id": request_id, "ok": True, "result": result}

		if method == "shutdown":
			self._running = False
			return {"id": request_id, "ok": True, "result": {"status": "shutting_down"}}

		raise ValueError(f"Unknown method: {method}")

	def serve_stdio(self):
		for line in sys.stdin:
			line = line.strip()
			if not line:
				continue

			try:
				request = json.loads(line)
				if not isinstance(request, dict):
					raise ValueError("Request must be a JSON object")
				response = self.handle_request(request)
			except Exception as exc:
				response = {
					"id": None,
					"ok": False,
					"error": {
						"message": str(exc),
						"traceback": traceback.format_exc(),
					},
				}

			sys.stdout.write(json.dumps(response) + "\n")
			sys.stdout.flush()

			if not self._running:
				break


def main():
	server = FastSurferIPCServer()
	server.serve_stdio()


if __name__ == "__main__":
	main()
