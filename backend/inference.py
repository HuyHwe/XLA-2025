from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import HTTPException
import tensorflow as tf

from . import model_defs
from .model_defs import Convolution, MaxPool, Fully_Connected

LOGGER = logging.getLogger(__name__)

# Đường dẫn đến các file model
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH_CUSTOM = MODELS_DIR / "cnn_model.pkl"
MODEL_PATH_KERAS = MODELS_DIR / "cnn_mnist_keras.h5"
MODEL_PATH_SHAPE = MODELS_DIR / "cnn_shape.h5"
MODEL_PATH_CHAR = MODELS_DIR / "cnn_char.h5"

# Class mappings cho các model
SHAPE_CLASSES = ["Circle", "Diamond", "Square", "Triangle"]
CHAR_CLASSES = [chr(ord("A") + i) for i in range(26)]  # A-Z

# Biến global lưu trữ các model đã load
_models: Dict[str, Any] = {}
_model_lock = Lock()


class NotebookModelUnpickler(pickle.Unpickler):
    """Unpickler đặc biệt để load model được train trong notebook (__main__)."""

    def find_class(self, module, name):
        if module == "__main__":
            module = f"{__package__}.model_defs"
        return super().find_class(module, name)


def _load_custom_model_from_disk() -> Dict[str, Any]:
    """
    Load model CNN tự xây dựng từ file pickle.
    Model này chứa 3 layer: conv_layer, pool_layer, fc_layer
    """
    if not MODEL_PATH_CUSTOM.exists():
        raise FileNotFoundError(f"Custom model file not found at {MODEL_PATH_CUSTOM}")

    # Đảm bảo module __main__ được map đúng để pickle có thể load class
    sys.modules.setdefault("__main__", model_defs)

    # Load model từ file pickle
    with MODEL_PATH_CUSTOM.open("rb") as file:
        components = NotebookModelUnpickler(file).load()

    # Kiểm tra model có đủ các thành phần cần thiết không
    required_keys = {"conv_layer", "pool_layer", "fc_layer"}
    missing = required_keys - components.keys()
    if missing:
        raise ValueError(f"Loaded model is missing components: {', '.join(sorted(missing))}")

    return components


def _load_keras_model_from_disk(model_path: Path):
    """
    Load model CNN Keras/TensorFlow từ file H5.
    Model này được train bằng TensorFlow và lưu dạng HDF5.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Keras model file not found at {model_path}")

    model = tf.keras.models.load_model(model_path)
    return model


def init_models() -> None:
    """
    Khởi tạo và load các model vào memory.
    Model number_custom là bắt buộc, các model khác là tùy chọn.
    """
    global _models

    # Load model custom cho số (bắt buộc) - nếu fail thì dừng server
    try:
        _models["number_custom"] = _load_custom_model_from_disk()
        LOGGER.info("Custom CNN number model loaded successfully from %s", MODEL_PATH_CUSTOM)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load custom CNN number model: %s", exc)
        raise

    # Load model Keras cho số (tùy chọn)
    try:
        _models["number_keras"] = _load_keras_model_from_disk(MODEL_PATH_KERAS)
        LOGGER.info("Keras CNN number model loaded successfully from %s", MODEL_PATH_KERAS)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load Keras CNN number model: %s", exc)

    # Load model nhận diện hình dạng (tùy chọn)
    try:
        _models["shape"] = _load_keras_model_from_disk(MODEL_PATH_SHAPE)
        LOGGER.info("Shape CNN model loaded successfully from %s", MODEL_PATH_SHAPE)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load Shape CNN model: %s", exc)

    # Load model nhận diện chữ cái (tùy chọn)
    try:
        _models["char"] = _load_keras_model_from_disk(MODEL_PATH_CHAR)
        LOGGER.info("Character CNN model loaded successfully from %s", MODEL_PATH_CHAR)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load Character CNN model: %s", exc)


def get_available_models() -> Dict[str, Any]:
    """Trả về thông tin các model đang có trong memory."""
    return {
        "number_custom": {
            "name": "Nhận diện số - Custom CNN",
            "file": "cnn_model.pkl",
            "available": "number_custom" in _models,
            "category": "number",
        },
        "number_keras": {
            "name": "Nhận diện số - Keras/TensorFlow",
            "file": "cnn_mnist_keras.h5",
            "available": "number_keras" in _models,
            "category": "number",
        },
        "shape": {
            "name": "Nhận diện hình dạng",
            "file": "cnn_shape.h5",
            "available": "shape" in _models,
            "category": "shape",
        },
        "char": {
            "name": "Nhận diện chữ cái (A-Z)",
            "file": "cnn_char.h5",
            "available": "char" in _models,
            "category": "char",
        },
    }


def is_model_ready(key: str = "number_custom") -> bool:
    """Kiểm tra model bắt buộc đã được load chưa."""
    return key in _models


def run_inference(image_array: np.ndarray, model_type: str) -> Union[int, str]:
    """
    Router function: chọn model phù hợp để chạy inference dựa trên model_type.
    Trả về int cho số, string cho hình dạng/chữ cái.
    """
    if model_type == "number_custom":
        return _run_inference_custom(image_array)
    if model_type == "number_keras":
        predictions = _run_inference_keras_model(image_array, "number_keras")
        return int(np.argmax(predictions))
    if model_type == "shape":
        predictions = _run_inference_keras_model(image_array, "shape")
        class_id = int(np.argmax(predictions))
        return SHAPE_CLASSES[class_id]
    if model_type == "char":
        predictions = _run_inference_keras_model(image_array, "char")
        class_id = int(np.argmax(predictions))
        return CHAR_CLASSES[class_id]

    raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")


def ensure_model_available(model_type: str, available: List[str] | None = None) -> None:
    """Throw HTTPException nếu model không có trong _models."""
    available_keys = available if available is not None else list(_models.keys())
    if model_type not in _models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_type}' is not available. Available models: {available_keys}",
        )


def _run_inference_custom(image_array: np.ndarray) -> int:
    """
    Chạy inference với model CNN tự xây dựng cho nhận diện số.
    Quy trình: Conv -> MaxPool -> FullyConnected -> Softmax -> Lấy class có xác suất cao nhất.
    """
    if "number_custom" not in _models:
        raise HTTPException(status_code=503, detail="Custom number model is not loaded yet")

    model_components = _models["number_custom"]

    # Sử dụng lock để đảm bảo thread-safe khi truy cập model
    with _model_lock:
        conv: Convolution = model_components["conv_layer"]
        pool: MaxPool = model_components["pool_layer"]
        fc: Fully_Connected = model_components["fc_layer"]

        # Forward pass qua các layer
        conv_output = conv.forward(image_array)  # Convolution layer
        pool_output = pool.forward(conv_output)  # MaxPooling layer
        fc_output = fc.forward(pool_output)  # FullyConnected + Softmax

    # Lấy class có xác suất cao nhất (argmax)
    prediction = int(np.argmax(fc_output))
    return prediction


def _run_inference_keras_model(image_array: np.ndarray, model_key: str) -> np.ndarray:
    """
    Chạy inference với model Keras/TensorFlow.
    Model Keras cần input shape (batch_size, height, width, channels).
    """
    if model_key not in _models:
        raise HTTPException(status_code=503, detail=f"{model_key} model is not loaded yet")

    model = _models[model_key]

    # Keras model cần input shape (1, 28, 28, 1) - thêm batch dimension và channel dimension
    image_batch = np.expand_dims(image_array, axis=0)

    # Sử dụng lock để đảm bảo thread-safe
    with _model_lock:
        predictions = model.predict(image_batch, verbose=0)

    return predictions[0]



