"""Reusable training helpers for the MXDS012 Host ML workflow."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency during GUI startup
    torch = None
    nn = None
    DataLoader = object  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

torch = cast(Any, torch)
nn = cast(Any, nn)
DataLoader = cast(Any, DataLoader)

try:
    from .CBAMLSTMModel import CBAMLSTMNet
    from .CNN2DStaticModel import CNN2DStaticModel
    from .flat_us_dataset import DEFAULT_GESTURE_ORDER, DataFormatError, FlatUltrasoundDataset, FlatUltrasoundParser, SplitConfig
    from .plot_utils import collect_labels_and_predictions, plot_confusion_matrix, plot_learning_curves
except ImportError:  # pragma: no cover - allows running ml/main.py as a script
    from CBAMLSTMModel import CBAMLSTMNet
    from CNN2DStaticModel import CNN2DStaticModel
    from flat_us_dataset import DEFAULT_GESTURE_ORDER, DataFormatError, FlatUltrasoundDataset, FlatUltrasoundParser, SplitConfig
    from plot_utils import collect_labels_and_predictions, plot_confusion_matrix, plot_learning_curves


SUPPORTED_MODEL_TYPES = ("cbamlstm", "cnn2d")


@dataclass
class DatasetAnalysis:
    root_dir: str
    valid_files: int
    ignored_files: int
    issues: List[str]
    classes: List[str]
    sampling_depth: int
    num_classes: int


@dataclass
class TrainingArtifacts:
    best_model_path: str
    onnx_path: str
    metadata_path: str
    curves_path: str
    confusion_matrix_path: Optional[str]


def set_seed(seed: int) -> None:
    if not _TORCH_AVAILABLE:
        return
    random.seed(seed)
    np.random.seed(seed)
    getattr(torch, 'manual_seed')(seed)
    getattr(getattr(torch, 'cuda'), 'manual_seed_all')(seed)


def prepare_input(x: Any, model_type: str = "cbamlstm") -> Any:
    """Adapt dataset tensor shape to the selected model input contract."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for model input preparation")
    if x.ndim != 4:
        raise ValueError(f"Expected input ndim=4, got shape={tuple(x.shape)}")
    normalized_type = str(model_type).strip().lower()
    if normalized_type == "cbamlstm":
        return x.unsqueeze(1).float()
    if normalized_type == "cnn2d":
        return x.float()
    raise ValueError(f"Unknown model_type={model_type}, supported={SUPPORTED_MODEL_TYPES}")


def build_label_mapper(
    src_class_to_idx: Dict[str, int],
    dst_class_to_idx: Dict[str, int],
    device: Any,
) -> Any:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for label mapping")
    max_src_idx = max(src_class_to_idx.values()) if src_class_to_idx else -1
    mapper = getattr(torch, 'full')((max_src_idx + 1,), -1, dtype=getattr(torch, 'long'))

    for gesture, src_idx in src_class_to_idx.items():
        if gesture in dst_class_to_idx:
            mapper[src_idx] = dst_class_to_idx[gesture]

    return mapper.to(device)


def remap_labels(labels: Any, mapper: Any) -> Any:
    labels = labels.long()
    if labels.numel() == 0:
        return labels
    mapped = mapper[labels]
    if (mapped < 0).any():
        bad = labels[mapped < 0].unique().tolist()
        raise ValueError(f"Found labels that cannot map to train classes: {bad}")
    return mapped


def run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    optimizer: Optional[Any] = None,
    label_mapper: Optional[Any] = None,
    model_type: str = "cbamlstm",
):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for training")
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = prepare_input(x, model_type=model_type).to(device)
        y = y.to(device).long()

        if label_mapper is not None:
            y = remap_labels(y, label_mapper)

        if train_mode:
            optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)

        if train_mode:
            loss.backward()
            optimizer.step()

        pred = out.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        total_correct += (pred == y).sum().item()
        total_count += y.size(0)

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def analyze_dataset(root_dir: str, strict: bool = False, known_gesture_order: Optional[Sequence[str]] = None) -> DatasetAnalysis:
    parser = FlatUltrasoundParser(strict=strict)
    metas = parser.discover(root_dir)
    records = []
    for meta in metas:
        parsed = parser.parse_file(meta)
        if parsed is not None:
            records.append(parsed)

    appeared = {record.meta.gesture for record in records}
    allowed_order = list(known_gesture_order) if known_gesture_order is not None else list(DEFAULT_GESTURE_ORDER)
    classes = [gesture for gesture in allowed_order if gesture in appeared]

    issues = [f"{issue.path}: {issue.message}" for issue in parser.issues]
    return DatasetAnalysis(
        root_dir=str(root_dir),
        valid_files=len(records),
        ignored_files=len(parser.ignored_files),
        issues=issues,
        classes=classes,
        sampling_depth=int(parser.global_depth or 0),
        num_classes=len(classes),
    )


def make_dataset(
    root_dir: str,
    split: str,
    cfg: SplitConfig,
    strict: bool,
) -> Optional[FlatUltrasoundDataset]:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required to build datasets")
    try:
        return FlatUltrasoundDataset(
            root_dir=root_dir,
            split=split,
            split_config=cfg,
            strict=strict,
            preprocess_mode="standardize_only",
            flatten_for_lstm=False,
            label_dtype=getattr(torch, 'float32'),
        )
    except DataFormatError:
        return None


def export_onnx_model(
    model: Any,
    onnx_path: str,
    device: Any,
    depth: int,
    model_type: str = "cbamlstm",
) -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required to export ONNX models")
    model.eval()
    normalized_type = str(model_type).strip().lower()
    if normalized_type == "cbamlstm":
        dummy_input = getattr(torch, 'randn')(1, 1, 1, 4, depth, device=device)
    elif normalized_type == "cnn2d":
        dummy_input = getattr(torch, 'randn')(1, 1, 4, depth, device=device)
    else:
        raise ValueError(f"Unknown model_type={model_type}, supported={SUPPORTED_MODEL_TYPES}")
    with getattr(torch, 'no_grad')():
        _ = model(dummy_input)

    getattr(getattr(torch, 'onnx'), 'export')(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
    )


def export_onnx_metadata(
    onnx_path: str,
    depth: int,
    classes: Sequence[str],
    model_type: str = "cbamlstm",
) -> str:
    root, _ = os.path.splitext(onnx_path)
    json_path = f"{root}.json"

    metadata = {
        "sampling_depth": int(depth),
        "classes": list(classes),
        "num_classes": len(classes),
        "model_type": str(model_type).strip().lower(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return json_path


def with_depth_suffix(path: str, depth: int) -> str:
    """Append _d{depth} before extension, unless already present."""
    root, ext = os.path.splitext(path)
    if re.search(r"_d\d+$", root):
        return path
    return f"{root}_d{depth}{ext}"


def save_training_curves(history: Dict[str, List[float]], save_path: str) -> None:
    plot_learning_curves(history, save_path)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    plot_confusion_matrix(y_true=y_true, y_pred=y_pred, class_names=class_names, save_path=save_path, title=title)


def collect_predictions(
    model: Any,
    loader: Any,
    device: Any,
    label_mapper: Optional[Any] = None,
    model_type: str = "cbamlstm",
) -> tuple[np.ndarray, np.ndarray]:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required to collect predictions")
    return collect_labels_and_predictions(
        model=model,
        loader=loader,
        device=device,
        input_transform=lambda batch_x: prepare_input(batch_x, model_type=model_type),
        label_mapper=label_mapper,
    )


def build_model(num_classes: int, device: Any, model_type: str = "cbamlstm") -> Any:
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required to build the model")
    normalized_type = str(model_type).strip().lower()
    if normalized_type == "cbamlstm":
        return CBAMLSTMNet(num_classes=num_classes).to(device)
    if normalized_type == "cnn2d":
        return CNN2DStaticModel(num_classes=num_classes).to(device)
    raise ValueError(f"Unknown model_type={model_type}, supported={SUPPORTED_MODEL_TYPES}")
