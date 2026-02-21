FastSurfer FastAPI backend

This folder contains a minimal FastAPI server to run FastSurfer prediction on uploaded T1 images.

Usage:

1. Install dependencies (preferably in a virtualenv):

```
pip install -r app/backend/requirements.txt
```

2. Run the server:

```
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

3. POST a file to `/predict` (multipart form `file`) to run a prediction. The server returns the produced segmentation file.

Notes:
- The backend uses the repository's default checkpoint/config locations. The first request may trigger download of model checkpoints.
- This is a minimal example; production usage should add input validation, authentication, async handling, and resource limits.
