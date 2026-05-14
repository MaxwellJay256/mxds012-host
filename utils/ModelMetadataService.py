"""Model metadata loading utilities for online analysis UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json


@dataclass
class ModelMetadata:
    model_path: str
    json_path: str
    sampling_depth: int
    classes: List[str]
    num_classes: int
    model_type: str


def find_sidecar_json_path(model_path: str) -> Path:
    """Return sidecar JSON path with the same stem as the ONNX file."""
    onnx_path = Path(model_path)
    return onnx_path.with_suffix(".json")


def load_model_metadata(model_path: str) -> Tuple[ModelMetadata, List[str]]:
    """Load model metadata from sidecar JSON and return warnings if any."""
    onnx_path = Path(model_path)
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError("请选择 .onnx 模型文件")

    json_path = find_sidecar_json_path(model_path)
    if not json_path.exists():
        raise FileNotFoundError(f"未找到同名 JSON: {json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    sampling_depth = int(payload.get("sampling_depth", 0) or 0)
    if sampling_depth <= 0:
        raise ValueError("JSON 中 sampling_depth 无效")

    classes_raw = payload.get("classes", [])
    if not isinstance(classes_raw, list):
        raise ValueError("JSON 中 classes 必须是数组")

    classes: List[str] = []
    for item in classes_raw:
        name = str(item).strip()
        if name:
            classes.append(name)

    if not classes:
        raise ValueError("JSON 中 classes 为空")

    num_classes = int(payload.get("num_classes", len(classes)) or len(classes))
    model_type = str(payload.get("model_type", "cbamlstm") or "cbamlstm").strip().lower()

    warnings: List[str] = []
    if num_classes != len(classes):
        warnings.append(
            f"num_classes={num_classes} 与 classes 数量={len(classes)} 不一致，将以 classes 实际内容为准"
        )

    if model_type not in {"cbamlstm", "cnn2d"}:
        warnings.append(f"未知 model_type={model_type}，将回退为 cbamlstm")
        model_type = "cbamlstm"

    metadata = ModelMetadata(
        model_path=str(onnx_path),
        json_path=str(json_path),
        sampling_depth=sampling_depth,
        classes=classes,
        num_classes=num_classes,
        model_type=model_type,
    )
    return metadata, warnings
