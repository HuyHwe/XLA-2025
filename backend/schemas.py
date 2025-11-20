from __future__ import annotations

from typing import Dict, List, Literal, Union

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request model: nhận ảnh base64 và loại model muốn sử dụng."""

    image_base64: str
    model_type: Literal["number_custom", "number_keras", "shape", "char"] = "number_custom"


class PredictResponse(BaseModel):
    """Response model: trả về kết quả dự đoán, loại model và các bước tiền xử lý (nếu có)."""

    prediction: Union[int, str]
    model_type: str
    preprocessing_steps: List[Dict[str, str]] | None = None



