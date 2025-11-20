"""FastAPI backend service for handwriting/shape recognition."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .inference import (
    ensure_model_available,
    get_available_models,
    init_models,
    is_model_ready,
    run_inference,
)
from .preprocessing import preprocess_image
from .schemas import PredictRequest, PredictResponse

LOGGER = logging.getLogger(__name__)


app = FastAPI(title="Handwriting & Shape Recognition API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_event() -> None:
    """Load all models into memory when the server starts."""
    init_models()


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Endpoint kiểm tra trạng thái server.
    Trả về "ready" nếu model bắt buộc đã load, "loading" nếu chưa.
    """
    status = "ready" if is_model_ready() else "loading"
    return {"status": status}


@app.get("/models")
def models_endpoint() -> Dict[str, Any]:
    """
    Endpoint trả về danh sách các model có sẵn và trạng thái của chúng.
    Frontend sử dụng endpoint này để hiển thị dropdown chọn model.
    """
    return {"models": get_available_models()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Endpoint chính để nhận diện từ ảnh (số, hình dạng, hoặc chữ cái).
    Quy trình:
    1. Kiểm tra model có sẵn không
    2. Tiền xử lý ảnh (decode, resize, normalize)
    3. Chạy inference với model đã chọn
    4. Trả về kết quả dự đoán
    """
    model_type = request.model_type

    # Kiểm tra model có sẵn không
    ensure_model_available(model_type)

    # Tiền xử lý ảnh: decode base64 -> resize 28x28 -> normalize [0,1]
    image_array, steps = preprocess_image(request.image_base64)

    # Chạy inference với model đã chọn
    prediction = run_inference(image_array, model_type)

    # Trả về kết quả + các bước tiền xử lý (nếu có)
    return PredictResponse(
        prediction=prediction,
        model_type=model_type,
        preprocessing_steps=steps or None,
    )



