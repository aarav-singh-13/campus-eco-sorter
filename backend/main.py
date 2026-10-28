"""
Campus Eco & Waste Sorting Identifier - FastAPI backend.

Run (from the backend folder):
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000  (API docs at http://localhost:8000/docs)
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from bins import bins_for
from model import WasteClassifier

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
FEEDBACK_DIR = BASE_DIR / "feedback"
CATEGORIES_FILE = BASE_DIR / "data" / "categories.json"

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB

with open(CATEGORIES_FILE, encoding="utf-8") as f:
    CATEGORIES = json.load(f)["categories"]

classifier = WasteClassifier()

app = FastAPI(title="Campus Eco & Waste Sorting Identifier", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this when you deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 8 MB.")
    return data


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "demo_mode": classifier.demo,
        "model_file": classifier.model_file,
        "message": classifier.load_error,
        "classes": classifier.classes,
        "predictable_categories": classifier.categories,
        "confidence_threshold": CONF_THRESHOLD,
    }


@app.get("/api/categories")
def categories():
    return CATEGORIES


@app.get("/api/bins")
def get_bins(category: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None):
    if category and category not in CATEGORIES:
        raise HTTPException(404, f"Unknown category '{category}'.")
    return bins_for(category, lat, lon)


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
):
    data = await read_upload(file)
    try:
        ranked = classifier.predict(data)
    except Exception:
        raise HTTPException(400, "Could not read that file as an image.")

    top_id, top_p = ranked[0]
    return {
        "category": top_id,
        "confidence": round(top_p, 4),
        "low_confidence": top_p < CONF_THRESHOLD,
        "top3": [{"category": c, "confidence": round(p, 4)} for c, p in ranked[:3]],
        "info": CATEGORIES[top_id],
        "bins": bins_for(top_id, lat, lon),
        "demo_mode": classifier.demo,
    }


@app.post("/api/feedback")
async def feedback(
    file: UploadFile = File(...),
    predicted: str = Form(...),
    correct: str = Form(...),
):
    """Save the photo under feedback/<correct label>/ so it can be added to the training set."""
    if correct not in CATEGORIES:
        raise HTTPException(400, f"Unknown category '{correct}'.")
    data = await read_upload(file)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = FEEDBACK_DIR / correct
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    img_path = folder / f"{stamp}{suffix}"
    img_path.write_bytes(data)

    log = FEEDBACK_DIR / "feedback_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "image", "predicted", "correct", "was_correct"])
        w.writerow([stamp, img_path.relative_to(FEEDBACK_DIR).as_posix(), predicted, correct, predicted == correct])

    return {"saved": True}


# Serve the website itself. Mounted last so /api/* routes take priority.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
