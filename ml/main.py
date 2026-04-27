import argparse
import json
import os
import random
import re
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from CBAMLSTMModel import CBAMLSTMNet
from flat_us_dataset import DataFormatError, FlatUltrasoundDataset, SplitConfig
from plot_utils import collect_labels_and_predictions, plot_confusion_matrix, plot_learning_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_input(x: torch.Tensor) -> torch.Tensor:
    # Dataset output shape is (B, 1, 4, depth); model expects (B, T, C, H, W).
    if x.ndim != 4:
        raise ValueError(f"Expected input ndim=4, got shape={tuple(x.shape)}")
    return x.unsqueeze(1).float()


def build_label_mapper(
    src_class_to_idx: Dict[str, int],
    dst_class_to_idx: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    max_src_idx = max(src_class_to_idx.values()) if src_class_to_idx else -1
    mapper = torch.full((max_src_idx + 1,), -1, dtype=torch.long)

    for gesture, src_idx in src_class_to_idx.items():
        if gesture in dst_class_to_idx:
            mapper[src_idx] = dst_class_to_idx[gesture]

    return mapper.to(device)


def remap_labels(labels: torch.Tensor, mapper: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    if labels.numel() == 0:
        return labels
    mapped = mapper[labels]
    if (mapped < 0).any():
        bad = labels[mapped < 0].unique().tolist()
        raise ValueError(f"Found labels that cannot map to train classes: {bad}")
    return mapped


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
    label_mapper: Optional[torch.Tensor] = None,
):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = prepare_input(x).to(device)
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


def print_dataset_info(name: str, ds: FlatUltrasoundDataset) -> None:
    print(f"\n[{name}] report:")
    for k, v in ds.report.items():
        print(f"  {k}: {v}")

    if ds.issues:
        print(f"  first issues ({min(5, len(ds.issues))}):")
        for issue in ds.issues[:5]:
            print(
                f"    path={issue.path}, frame={issue.frame_index}, "
                f"line={issue.line_number}, msg={issue.message}"
            )


def make_dataset(
    root_dir: str,
    split: str,
    cfg: SplitConfig,
    strict: bool,
) -> Optional[FlatUltrasoundDataset]:
    try:
        return FlatUltrasoundDataset(
            root_dir=root_dir,
            split=split,
            split_config=cfg,
            strict=strict,
            preprocess_mode="standardize_only",
            flatten_for_lstm=False,
            label_dtype=torch.float32,
        )
    except DataFormatError as e:
        print(f"[warn] split={split} unavailable: {e}")
        return None


def export_onnx_model(
    model: nn.Module,
    onnx_path: str,
    device: torch.device,
    depth: int,
) -> None:
    model.eval()
    dummy_input = torch.randn(1, 1, 1, 4, depth, device=device)
    # Initialize lazy parameters (e.g., LazyLinear) before ONNX tracing/export.
    with torch.no_grad():
        _ = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
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
    classes: list[str],
) -> str:
    root, _ = os.path.splitext(onnx_path)
    json_path = f"{root}.json"

    metadata = {
        "sampling_depth": int(depth),
        "classes": list(classes),
        "num_classes": len(classes),
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


def main() -> None:
    parser = argparse.ArgumentParser("Train CBAMLSTMNet on MXDS012 ultrasound dataset")
    parser.add_argument("--data-dir", type=str, default="./20260416")
    parser.add_argument("--epochs", type=int, default=50)  # 训练轮数
    parser.add_argument("--batch-size", type=int, default=64)  # 批量大小，过大可能导致内存不足，过小可能训练不稳定
    parser.add_argument("--lr", type=float, default=1e-3)  # 学习率
    parser.add_argument("--split-strategy", type=str, default="ratio", choices=["ratio", "trial"])  # 数据集划分策略，包括按比例划分和按试验划分
    parser.add_argument("--seed", type=int, default=42)  # 随机划分数据集的种子
    parser.add_argument("--num-workers", type=int, default=0)  # DataLoader 的子进程数，0 表示在主进程加载数据
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export-onnx", action="store_true", help="Export best model to ONNX after training")
    parser.add_argument("--onnx-path", type=str, default="cbam_lstm_best.onnx", help="Output path for ONNX model")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save training plots")
    args = parser.parse_args()

    device = torch.device(args.device)
    # 打印设备信息
    print(f"Using device: {device}")
    set_seed(args.seed)

    cfg = SplitConfig(strategy=args.split_strategy, seed=args.seed)

    train_ds = make_dataset(args.data_dir, "train", cfg, args.strict)
    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError("Train split is empty. Please check data files or split strategy.")
    print_dataset_info("train", train_ds)

    val_ds = make_dataset(args.data_dir, "val", cfg, args.strict)
    if val_ds is not None:
        print_dataset_info("val", val_ds)

    test_ds = make_dataset(args.data_dir, "test", cfg, args.strict)
    if test_ds is not None:
        print_dataset_info("test", test_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_loader = None
    test_loader = None

    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )

    num_classes = len(train_ds.classes)
    depth = train_ds.report.get("sampling_depth")
    if depth is None:
        depth = int(train_ds[0][0].shape[-1])
    depth = int(depth)

    model = CBAMLSTMNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_metric = -1.0
    best_path = with_depth_suffix("cbam_lstm_best.pth", depth)  # 保存最佳模型的路径

    val_mapper = None
    test_mapper = None
    if val_ds is not None:
        val_mapper = build_label_mapper(val_ds.class_to_idx, train_ds.class_to_idx, device)
    if test_ds is not None:
        test_mapper = build_label_mapper(test_ds.class_to_idx, train_ds.class_to_idx, device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            label_mapper=None,
        )

        if val_loader is not None:
            val_loss, val_acc = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                label_mapper=val_mapper,
            )
            metric = val_acc
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        else:
            metric = train_acc
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if metric > best_metric:
            best_metric = metric
            torch.save(model.state_dict(), best_path)
            print(f"  saved best model -> {best_path}")

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    os.makedirs(args.plot_dir, exist_ok=True)
    curves_path = os.path.join(args.plot_dir, with_depth_suffix("learning_curves.png", depth))
    plot_learning_curves(history, curves_path)
    print(f"[plot] learning curves saved -> {curves_path}")

    if val_loader is not None:
        y_true_val, y_pred_val = collect_labels_and_predictions(
            model=model,
            loader=val_loader,
            device=device,
            input_transform=prepare_input,
            label_mapper=val_mapper,
        )
        if y_true_val.size > 0:
            cm_val_path = os.path.join(args.plot_dir, with_depth_suffix("confusion_matrix_val.png", depth))
            plot_confusion_matrix(
                y_true=y_true_val,
                y_pred=y_pred_val,
                class_names=train_ds.classes,
                save_path=cm_val_path,
                title="Validation Confusion Matrix",
            )
            print(f"[plot] val confusion matrix saved -> {cm_val_path}")

    if args.export_onnx:
        onnx_path = with_depth_suffix(args.onnx_path, depth)
        onnx_dir = os.path.dirname(onnx_path)
        if onnx_dir:
            os.makedirs(onnx_dir, exist_ok=True)
        export_onnx_model(model, onnx_path, device, depth)
        print(f"\n[export] onnx model saved -> {onnx_path}")

        metadata_path = export_onnx_metadata(
            onnx_path=onnx_path,
            depth=depth,
            classes=train_ds.classes,
        )
        print(f"[export] metadata json saved -> {metadata_path}")

    if test_loader is not None:
        test_loss, test_acc = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            label_mapper=test_mapper,
        )
        print(f"\n[test] loss={test_loss:.4f}, acc={test_acc:.4f}")

        y_true_test, y_pred_test = collect_labels_and_predictions(
            model=model,
            loader=test_loader,
            device=device,
            input_transform=prepare_input,
            label_mapper=test_mapper,
        )
        if y_true_test.size > 0:
            cm_test_path = os.path.join(args.plot_dir, with_depth_suffix("confusion_matrix_test.png", depth))
            plot_confusion_matrix(
                y_true=y_true_test,
                y_pred=y_pred_test,
                class_names=train_ds.classes,
                save_path=cm_test_path,
                title="Test Confusion Matrix",
            )
            print(f"[plot] test confusion matrix saved -> {cm_test_path}")
    else:
        print("\n[test] skipped (empty split).")


if __name__ == "__main__":
    main()