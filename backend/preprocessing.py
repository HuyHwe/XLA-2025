from __future__ import annotations

import base64
import io
from typing import Dict, List, Tuple

import numpy as np
from fastapi import HTTPException
from PIL import Image, ImageFilter


def preprocess_image(image_base64: str) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    Tiền xử lý ảnh từ base64 string thành numpy array (28x28) chuẩn hóa [0, 1].
    Bước xử lý:
    1. Decode base64
    2. Chuyển sang grayscale
    3. Căn giữa và normalize chữ số
    4. Resize về 28x28 (kích thước chuẩn của MNIST)
    5. Chuẩn hóa pixel về [0, 1] và tăng tương phản nhẹ
    """
    # Danh sách lưu lại các bước tiền xử lý để trả về cho FE
    steps: List[Dict[str, str]] = []

    def _pil_to_base64_dict(img: Image.Image, name: str) -> None:
        """Convert PIL Image -> data:image/png;base64,... và append vào steps."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        steps.append(
            {
                "name": name,
                "image_base64": f"data:image/png;base64,{encoded}",
            }
        )

    # Loại bỏ prefix "data:image/png;base64," nếu có
    if "," in image_base64:
        _, image_base64 = image_base64.split(",", 1)

    # Decode base64 thành bytes
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (base64.binascii.Error, ValueError) as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Chuyển sang grayscale (ảnh đen trắng)
            img = img.convert("L")
            _pil_to_base64_dict(img, "Grayscale ban đầu")

            # Căn giữa chữ số và normalize kích thước
            img = _center_and_normalize_digit(img)
            _pil_to_base64_dict(img, "Căn giữa & chuẩn hóa chữ số")
            
            img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
            _pil_to_base64_dict(img, "Gaussian blur nhẹ")

            # Resize về 28x28 (kích thước chuẩn MNIST) với thuật toán LANCZOS để chất lượng tốt
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            _pil_to_base64_dict(img, "Resize về 28x28")
            
            # Chuyển sang numpy array và chuẩn hóa pixel về [0, 1]
            image_array = np.array(img, dtype=np.float32) / 255.0

            # Tăng độ tương phản nhẹ để dễ nhận diện hơn
            image_array = np.clip((image_array - 0.5) * 1.2 + 0.5, 0.0, 1.0)

    except OSError as exc:
        raise HTTPException(status_code=400, detail="Unable to decode image bytes") from exc

    # Kiểm tra kích thước ảnh sau xử lý
    if image_array.shape != (28, 28):
        raise HTTPException(status_code=400, detail="Processed image has invalid shape")

    return image_array, steps


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
    digit_region = img_array[top : bottom + 1, left : right + 1]

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
    centered[y_offset : y_offset + height, x_offset : x_offset + width] = digit_region

    # Resize về kích thước gốc để normalize kích thước chữ số
    centered_img = Image.fromarray(centered)
    original_size = img.size
    centered_img = centered_img.resize(original_size, Image.Resampling.LANCZOS)

    return centered_img



