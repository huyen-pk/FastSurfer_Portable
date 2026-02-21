from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import shutil
import tempfile
from pathlib import Path
import traceback

from inference_service import FastSurferInferenceService

app = FastAPI(title="FastSurferCNN ONNX Inference")
inference_service = FastSurferInferenceService()



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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
