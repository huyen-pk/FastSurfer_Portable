from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import shutil
import tempfile
from pathlib import Path
import traceback
from pydantic import BaseModel, Field

from inference_service import FastSurferInferenceService

app = FastAPI(title="FastSurferCNN ONNX Inference")
inference_service = FastSurferInferenceService()


class BatchProcessRequest(BaseModel):
    file_paths: list[str] = Field(default_factory=list)
    folder_paths: list[str] = Field(default_factory=list)



@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a T1 image (nifti/mgz/mgh) and run FastSurfer prediction.

    Returns the generated segmentation file as the response (first .mgz/.mgh/.nii found).
    """
    # save upload to temp dir
    tmpdir = Path(tempfile.mkdtemp(prefix="fastsurfer_"))
    try:
        orig_path = tmpdir / file.filename
        with orig_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # set output dir
        out_dir = tmpdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        prediction = inference_service.predict_from_path(orig_path, output_dir=out_dir)
        seg_path = Path(prediction["output_path"])
        return FileResponse(path=str(seg_path), filename=seg_path.name)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/process/start")
async def process_start(request: BatchProcessRequest):
    try:
        requested = inference_service.resolve_input_paths(
            file_paths=request.file_paths,
            folder_paths=request.folder_paths,
        )
        if not requested:
            raise ValueError("No valid input image files selected (.nii, .nii.gz, .mgz, .mgh).")

        requested_as_str = [str(path) for path in requested]
        return {
            "ack_message": f"Processing started for {len(requested_as_str)} path(s).",
            "requested_paths": requested_as_str,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/process/run")
async def process_run(request: BatchProcessRequest):
    try:
        return inference_service.predict_batch_from_paths(
            file_paths=request.file_paths,
            folder_paths=request.folder_paths,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
