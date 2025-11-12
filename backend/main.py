"""FastAPI backend service for MNIST handwriting recognition."""

from __future__ import annotations

import base64
import io
import logging
import pickle
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps

from . import model_defs
from .model_defs import Convolution, MaxPool, Fully_Connected

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="MNIST Handwriting Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    prediction: int


MODEL_PATH = Path(__file__).resolve().parents[1] / "cnn_model.pkl"
_model_components: Dict[str, Any] | None = None
_model_lock = Lock()


class NotebookModelUnpickler(pickle.Unpickler):
    """Custom unpickler that maps notebook-defined classes to packaged modules."""

    def find_class(self, module, name):
        if module == "__main__":
            module = f"{__package__}.model_defs"
        return super().find_class(module, name)


def _load_model_from_disk() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Ensure notebook module alias resolves for pickle
    sys.modules.setdefault("__main__", model_defs)

    with MODEL_PATH.open("rb") as file:
        components = NotebookModelUnpickler(file).load()

    required_keys = {"conv_layer", "pool_layer", "fc_layer"}
    missing = required_keys - components.keys()
    if missing:
        raise ValueError(f"Loaded model is missing components: {', '.join(sorted(missing))}")

    return components


def _preprocess_image(image_base64: str) -> np.ndarray:
    if "," in image_base64:
        _, image_base64 = image_base64.split(",", 1)

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("L")
            # img = ImageOps.invert(img)
            img = img.resize((28, 28))
            image_array = np.array(img, dtype=np.float32) / 255.0
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Unable to decode image bytes") from exc

    if image_array.shape != (28, 28):
        raise HTTPException(status_code=400, detail="Processed image has invalid shape")

    return image_array


def _run_inference(image_array: np.ndarray) -> int:
    if _model_components is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    with _model_lock:
        conv: Convolution = _model_components["conv_layer"]
        pool: MaxPool = _model_components["pool_layer"]
        fc: Fully_Connected = _model_components["fc_layer"]

        conv_output = conv.forward(image_array)
        pool_output = pool.forward(conv_output)
        fc_output = fc.forward(pool_output)

    prediction = int(np.argmax(fc_output))
    return prediction


@app.on_event("startup")
def _startup_event() -> None:
    global _model_components
    try:
        _model_components = _load_model_from_disk()
        LOGGER.info("CNN model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load CNN model: %s", exc)
        raise


@app.get("/health")
def health_check() -> Dict[str, str]:
    status = "ready" if _model_components is not None else "loading"
    return {"status": status}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    image_array = _preprocess_image(request.image_base64)
    prediction = _run_inference(image_array)
    return PredictResponse(prediction=prediction)


