"""FastAPI backend service for MNIST handwriting recognition."""

from __future__ import annotations

import base64
import io
import logging
import pickle
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageFilter
import tensorflow as tf

from . import model_defs
from .model_defs import Convolution, MaxPool, Fully_Connected

LOGGER = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(title="MNIST Handwriting Recognition API", version="1.0.0")

# Cấu hình CORS để cho phép frontend gọi API từ domain khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Định nghĩa request model: nhận ảnh base64 và loại model muốn sử dụng
class PredictRequest(BaseModel):
    image_base64: str  
    model_type: Literal["custom", "keras"] = "custom"  


# Định nghĩa response model: trả về kết quả dự đoán và model đã dùng
class PredictResponse(BaseModel):
    prediction: int  
    model_type: str  


# Đường dẫn đến các file model
MODEL_PATH_CUSTOM = Path(__file__).resolve().parents[1] / "models" / "cnn_model.pkl" 
MODEL_PATH_KERAS = Path(__file__).resolve().parents[1] / "models" / "cnn_mnist_keras.h5"  

# Biến global lưu trữ các model đã load
_model_components: Dict[str, Any] | None = None  # Lưu các layer của model custom (conv, pool, fc)
_keras_model: Any | None = None  # Lưu model Keras đã load
_model_lock = Lock()  # Lock để đảm bảo thread-safe khi sử dụng model


# Class đặc biệt để load model từ notebook (vì notebook dùng __main__ module)
class NotebookModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Chuyển đổi __main__ thành module thực tế để load được các class
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


def _load_keras_model_from_disk():
    """
    Load model CNN Keras/TensorFlow từ file H5.
    Model này được train bằng TensorFlow và lưu dạng HDF5
    """
    if not MODEL_PATH_KERAS.exists():
        raise FileNotFoundError(f"Keras model file not found at {MODEL_PATH_KERAS}")
    
    # Load model Keras từ file H5
    model = tf.keras.models.load_model(MODEL_PATH_KERAS)
    return model


def _preprocess_image(image_base64: str) -> np.ndarray:
    """
    Tiền xử lý ảnh từ base64 string thành numpy array (28x28) chuẩn hóa [0, 1].
    Bước xử lý:
    1. Decode base64
    2. Chuyển sang grayscale
    3. Căn giữa và normalize chữ số
    4. Resize về 28x28 (kích thước chuẩn của MNIST)
    5. Làm mịn ảnh
    6. Chuẩn hóa pixel về [0, 1]
    """
    # Loại bỏ prefix "data:image/png;base64," nếu có
    if "," in image_base64:
        _, image_base64 = image_base64.split(",", 1)

    # Decode base64 thành bytes
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (base64.binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Chuyển sang grayscale (ảnh đen trắng)
            img = img.convert("L")
        
            # Căn giữa chữ số và normalize kích thước
            img = _center_and_normalize_digit(img)
            
            # Resize về 28x28 (kích thước chuẩn MNIST) với thuật toán LANCZOS để chất lượng tốt
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Làm mịn ảnh để giảm artifacts
            img = img.filter(ImageFilter.SMOOTH_MORE)
            
            # Chuyển sang numpy array và chuẩn hóa pixel về [0, 1]
            image_array = np.array(img, dtype=np.float32) / 255.0
            
            # Tăng độ tương phản nhẹ để dễ nhận diện hơn
            image_array = np.clip((image_array - 0.5) * 1.2 + 0.5, 0.0, 1.0)
            
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Unable to decode image bytes") from exc

    # Kiểm tra kích thước ảnh sau xử lý
    if image_array.shape != (28, 28):
        raise HTTPException(status_code=400, detail="Processed image has invalid shape")

    return image_array


def _center_and_normalize_digit(img: Image.Image) -> Image.Image:
    """
    Căn giữa chữ số trong ảnh và normalize kích thước.
    Tìm bounding box của chữ số, cắt ra, thêm margin, và đặt vào giữa ảnh.
    """
    img_array = np.array(img)
    
    # Tìm bounding box: tìm các hàng và cột có pixel không phải background (khác 0)
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)
    
    # Nếu không có pixel nào (ảnh trống), trả về ảnh gốc
    if not np.any(rows) or not np.any(cols):
        return img
    
    # Lấy tọa độ bounding box
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    # Cắt vùng chứa chữ số
    digit_region = img_array[top:bottom+1, left:right+1]
    
    # Tính toán padding để căn giữa và thêm margin
    height, width = digit_region.shape
    max_dim = max(height, width)
    
    # Thêm margin 10% mỗi bên
    margin = int(max_dim * 0.1)
    new_size = max_dim + 2 * margin
    
    # Tạo ảnh mới với nền đen
    centered = np.zeros((new_size, new_size), dtype=img_array.dtype)
    
    # Tính offset để đặt chữ số vào giữa
    y_offset = (new_size - height) // 2
    x_offset = (new_size - width) // 2
    
    # Đặt chữ số vào giữa ảnh
    centered[y_offset:y_offset+height, x_offset:x_offset+width] = digit_region
    
    # Resize về kích thước gốc để normalize kích thước chữ số
    centered_img = Image.fromarray(centered)
    original_size = img.size
    centered_img = centered_img.resize(original_size, Image.Resampling.LANCZOS)
    
    return centered_img


def _run_inference_custom(image_array: np.ndarray) -> int:
    """
    Chạy inference với model CNN tự xây dựng.
    Quy trình: Conv -> MaxPool -> FullyConnected -> Softmax -> Lấy class có xác suất cao nhất
    """
    if _model_components is None:
        raise HTTPException(status_code=503, detail="Custom model is not loaded yet")

    # Sử dụng lock để đảm bảo thread-safe khi truy cập model
    with _model_lock:
        conv: Convolution = _model_components["conv_layer"]
        pool: MaxPool = _model_components["pool_layer"]
        fc: Fully_Connected = _model_components["fc_layer"]

        # Forward pass qua các layer
        conv_output = conv.forward(image_array)  # Convolution layer
        pool_output = pool.forward(conv_output)   # MaxPooling layer
        fc_output = fc.forward(pool_output)       # FullyConnected + Softmax

    # Lấy class có xác suất cao nhất (argmax)
    prediction = int(np.argmax(fc_output))
    return prediction


def _run_inference_keras(image_array: np.ndarray) -> int:
    """
    Chạy inference với model Keras/TensorFlow.
    Model Keras cần input shape (batch_size, height, width, channels)
    """
    if _keras_model is None:
        raise HTTPException(status_code=503, detail="Keras model is not loaded yet")

    # Keras model cần input shape (1, 28, 28, 1) - thêm batch dimension và channel dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Sử dụng lock để đảm bảo thread-safe
    with _model_lock:
        # Dự đoán với model Keras
        predictions = _keras_model.predict(image_batch, verbose=0)
    
    # Lấy class có xác suất cao nhất
    prediction = int(np.argmax(predictions[0]))
    return prediction


def _run_inference(image_array: np.ndarray, model_type: str) -> int:
    """
    Router function: chọn model phù hợp để chạy inference dựa trên model_type.
    """
    if model_type == "keras":
        return _run_inference_keras(image_array)
    else:  # default to custom
        return _run_inference_custom(image_array)


@app.on_event("startup")
def _startup_event() -> None:
    """
    Event handler khi server khởi động: load cả 2 model vào memory.
    Model custom là bắt buộc, model Keras là tùy chọn.
    """
    global _model_components, _keras_model
    
    # Load model custom (bắt buộc) - nếu fail thì dừng server
    try:
        _model_components = _load_custom_model_from_disk()
        LOGGER.info("Custom CNN model loaded successfully from %s", MODEL_PATH_CUSTOM)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to load custom CNN model: %s", exc)
        raise  # Dừng server nếu không load được model custom
    
    # Load model Keras (tùy chọn) - nếu fail chỉ warning, không dừng server
    try:
        _keras_model = _load_keras_model_from_disk()
        LOGGER.info("Keras CNN model loaded successfully from %s", MODEL_PATH_KERAS)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to load Keras CNN model: %s", exc)
        # Không raise - cho phép server chạy chỉ với model custom


@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Endpoint kiểm tra trạng thái server.
    Trả về "ready" nếu model đã load, "loading" nếu chưa.
    """
    status = "ready" if _model_components is not None else "loading"
    return {"status": status}


@app.get("/models")
def get_available_models() -> Dict[str, Any]:
    """
    Endpoint trả về danh sách các model có sẵn và trạng thái của chúng.
    Frontend sử dụng endpoint này để hiển thị dropdown chọn model.
    """
    models = {
        "custom": {
            "name": "Custom CNN (Tự xây dựng)",
            "file": "cnn_model.pkl",
            "available": _model_components is not None,  # Kiểm tra model đã load chưa
        },
        "keras": {
            "name": "Keras/TensorFlow CNN",
            "file": "cnn_mnist_keras.h5",
            "available": _keras_model is not None,  # Kiểm tra model đã load chưa
        },
    }
    return {"models": models}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Endpoint chính để nhận diện chữ số từ ảnh.
    Quy trình:
    1. Kiểm tra model có sẵn không
    2. Tiền xử lý ảnh (decode, resize, normalize)
    3. Chạy inference với model đã chọn
    4. Trả về kết quả dự đoán
    """
    model_type = request.model_type
    
    # Kiểm tra model có sẵn không
    if model_type == "keras" and _keras_model is None:
        raise HTTPException(status_code=503, detail="Keras model is not available")
    if model_type == "custom" and _model_components is None:
        raise HTTPException(status_code=503, detail="Custom model is not available")
    
    # Tiền xử lý ảnh: decode base64 -> resize 28x28 -> normalize [0,1]
    image_array = _preprocess_image(request.image_base64)
    
    # Chạy inference với model đã chọn
    prediction = _run_inference(image_array, model_type)
    
    # Trả về kết quả
    return PredictResponse(prediction=prediction, model_type=model_type)


