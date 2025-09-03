# -*- coding: utf-8 -*-
"""
FastAPI for Skin Lesion Classifier
Run:
  pip install fastapi uvicorn tensorflow pillow numpy python-multipart
  python api_skin_classifier.py  # hoặc: uvicorn api_skin_classifier:app --host 0.0.0.0 --port 8000

ENV / Query params:
  - MODEL_PATH: đường dẫn .h5/.keras/.hdf5 hoặc thư mục SavedModel
  - LABELS_PATH: (tùy chọn) đường dẫn .txt (mỗi dòng 1 lớp). Nếu không có -> dùng DEFAULT_LABELS
  - PREPROCESS: "rescale" (mặc định) hoặc "resnet50"

Endpoints:
  GET  /health
  GET  /meta
  POST /predict        (multipart/form-data: file=ảnh; query: preprocess, top_k)
  POST /predict_base64 (JSON: {"image_base64": "...", "preprocess":"rescale", "top_k":5})
"""

import os
import io
import time
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
try:
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
except Exception:
    resnet50_preprocess = None

APP_TITLE = "Skin Lesion Classifier API"

# ---------- Default labels (alphabetical by flow_from_directory) ----------
DEFAULT_LABELS = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

# ---------- GPU memory growth ----------
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# ---------- Helpers ----------
def _load_labels_from_file(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        raise ValueError("Labels file is empty.")
    return lines

def _detect_input_shape(model: tf.keras.Model) -> Tuple[int, int, int]:
    """Return (H, W, C) with fallbacks."""
    try:
        ishape = model.inputs[0].shape  # (None,H,W,C)
        H = int(ishape[1]) if ishape[1] is not None else 224
        W = int(ishape[2]) if ishape[2] is not None else 224
        C = int(ishape[3]) if ishape[3] is not None else 3
        return H, W, C
    except Exception:
        return 224, 224, 3

def _preprocess(np_img: np.ndarray, mode: str, channels: int) -> np.ndarray:
    """
    np_img: HxWx3 float32 (0..255 or raw)
    mode: 'rescale' | 'resnet50'
    """
    if mode == "resnet50":
        if resnet50_preprocess is None:
            arr = np_img / 255.0
        else:
            arr = resnet50_preprocess(np_img)
    else:
        arr = np_img / 255.0

    if channels == 1:
        arr = arr.mean(axis=-1, keepdims=True)
    return arr

def _load_image_bytes_to_batch(img_bytes: bytes,
                               target_size: Tuple[int, int],
                               mode: str = "rescale",
                               channels: int = 3) -> np.ndarray:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im = im.resize(target_size, Image.LANCZOS)
    arr = np.array(im).astype("float32")
    arr = _preprocess(arr, mode, channels)
    return np.expand_dims(arr, axis=0)

# ---------- App ----------
app = FastAPI(title=APP_TITLE)

# CORS (tùy chỉnh domain nếu cần)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # đổi về domain của bạn cho chặt chẽ hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load model & labels at startup ----------
MODEL_PATH = "model.hdf5"
LABELS_PATH = "labels.txt"

MODEL: Optional[tf.keras.Model] = None
LABELS: List[str] = list(DEFAULT_LABELS)
INPUT_HWC: Tuple[int, int, int] = (224, 224, 3)

@app.on_event("startup")
def _startup():
    global MODEL, LABELS, INPUT_HWC, MODEL_PATH, LABELS_PATH

    if not MODEL_PATH:
        # Cho phép khởi động API chưa có model (để /health, /meta vẫn chạy)
        print("[WARN] MODEL_PATH is empty. Set it via env or pass ?model_path=... to /meta (PUT coming soon).")
    else:
        print(f"[INFO] Loading model from: {MODEL_PATH}")
        MODEL = load_model(MODEL_PATH, compile=False)
        INPUT_HWC = _detect_input_shape(MODEL)
        print(f"[INFO] Detected input shape: {INPUT_HWC}")

    if LABELS_PATH and os.path.isfile(LABELS_PATH):
        print(f"[INFO] Loading labels from: {LABELS_PATH}")
        LABELS = _load_labels_from_file(LABELS_PATH)
    else:
        print("[INFO] Using DEFAULT_LABELS")

# ---------- Schemas ----------
class PredictBase64Request(BaseModel):
    image_base64: str
    preprocess: Optional[str] = "rescale"
    top_k: Optional[int] = 3

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.get("/meta")
def meta():
    return {
        "app": APP_TITLE,
        "model_path": MODEL_PATH or None,
        "labels_path": LABELS_PATH or None,
        "labels_count": len(LABELS) if LABELS else 0,
        "labels": LABELS,
        "input_shape": INPUT_HWC,
        "preprocess_supported": ["rescale", "resnet50"]
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    preprocess: str = Query("rescale", pattern="^(rescale|resnet50)$"),
    top_k: int = Query(3, ge=1, le=20),
):
    global MODEL, LABELS, INPUT_HWC
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH and restart server.")

    try:
        img_bytes = await file.read()
        H, W, C = INPUT_HWC
        batch = _load_image_bytes_to_batch(img_bytes, (W, H), preprocess, C)

        t0 = time.time()
        preds = MODEL.predict(batch, verbose=0)  # shape (1, num_classes)
        elapsed_ms = (time.time() - t0) * 1000.0

        probs = preds[0].astype(float)
        if probs.ndim != 1:
            raise ValueError("Model output is not 1D vector.")
        if len(LABELS) != len(probs):
            # Không khớp số lớp -> vẫn trả về chỉ số
            effective_labels = [f"class_{i}" for i in range(len(probs))]
        else:
            effective_labels = LABELS

        # Top-K
        idx_sorted = np.argsort(probs)[::-1]
        top_k = min(top_k, len(idx_sorted))
        top_items = []
        for i in range(top_k):
            idx = int(idx_sorted[i])
            top_items.append({
                "rank": i + 1,
                "index": idx,
                "label": effective_labels[idx],
                "prob": float(probs[idx])
            })

        return JSONResponse({
            "ok": True,
            "elapsed_ms": round(elapsed_ms, 2),
            "input_shape": {"H": H, "W": W, "C": C},
            "preprocess": preprocess,
            "top": top_items
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")

@app.post("/predict_base64")
def predict_base64(payload: PredictBase64Request):
    import base64
    global MODEL, LABELS, INPUT_HWC
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set MODEL_PATH and restart server.")

    try:
        img_bytes = base64.b64decode(payload.image_base64)
        H, W, C = INPUT_HWC
        preprocess = payload.preprocess or "rescale"
        if preprocess not in ("rescale", "resnet50"):
            preprocess = "rescale"
        top_k = max(1, min(20, payload.top_k or 3))

        batch = _load_image_bytes_to_batch(img_bytes, (W, H), preprocess, C)

        t0 = time.time()
        preds = MODEL.predict(batch, verbose=0)
        elapsed_ms = (time.time() - t0) * 1000.0

        probs = preds[0].astype(float)
        effective_labels = LABELS if len(LABELS) == len(probs) else [f"class_{i}" for i in range(len(probs))]
        idx_sorted = np.argsort(probs)[::-1]
        top_k = min(top_k, len(idx_sorted))

        top_items = []
        for i in range(top_k):
            idx = int(idx_sorted[i])
            top_items.append({
                "rank": i + 1,
                "index": idx,
                "label": effective_labels[idx],
                "prob": float(probs[idx])
            })

        return {
            "ok": True,
            "elapsed_ms": round(elapsed_ms, 2),
            "input_shape": {"H": H, "W": W, "C": C},
            "preprocess": preprocess,
            "top": top_items
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")
