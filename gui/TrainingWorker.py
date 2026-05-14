"""Background training worker for the GUI training page."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtCore import QThread, pyqtSignal

from ml.training_pipeline import (
    analyze_dataset,
    build_label_mapper,
    build_model,
    collect_predictions,
    export_onnx_metadata,
    export_onnx_model,
    make_dataset,
    save_confusion_matrix,
    save_training_curves,
    set_seed,
    with_depth_suffix,
    run_epoch,
)
from ml.flat_us_dataset import SplitConfig


class TrainingWorker(QThread):
    logMessage = pyqtSignal(str)
    progressChanged = pyqtSignal(int)
    finishedSuccessfully = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._config = dict(config)

    def _log(self, message: str):
        self.logMessage.emit(str(message))

    def run(self):
        try:
            dataset_dir = str(self._config.get("dataset_dir", "")).strip()
            if not dataset_dir:
                raise ValueError("dataset_dir 不能为空")

            seed = int(self._config.get("seed", 42))
            set_seed(seed)

            train_ratio = float(self._config.get("train_ratio", 0.8))
            val_ratio = float(self._config.get("val_ratio", 0.1))
            test_ratio = float(self._config.get("test_ratio", 0.1))
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                raise ValueError("训练/验证/测试比例之和必须为 1.0")

            self._log(f"分析数据集: {dataset_dir}")
            summary = analyze_dataset(dataset_dir, strict=False)
            self._log(
                f"有效文件={summary.valid_files}, 类别={summary.num_classes}, 采样深度={summary.sampling_depth}, 忽略文件={summary.ignored_files}"
            )
            for issue in summary.issues[:10]:
                self._log(f"[issue] {issue}")

            cfg = SplitConfig(
                strategy="ratio",
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

            train_ds = make_dataset(dataset_dir, "train", cfg, strict=False)
            if train_ds is None or len(train_ds) == 0:
                raise RuntimeError("训练集为空")

            val_ds = make_dataset(dataset_dir, "val", cfg, strict=False)
            test_ds = make_dataset(dataset_dir, "test", cfg, strict=False)

            depth = int(train_ds.report.get("sampling_depth") or 0)
            if depth <= 0:
                depth = int(train_ds[0][0].shape[-1])

            num_classes = len(train_ds.classes)
            if num_classes <= 0:
                raise RuntimeError("未找到可训练类别")

            device_name = self._config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
            device = torch.device(str(device_name))
            self._log(f"使用设备: {device}")
            model_type = str(self._config.get("model_type", "cnn2d")).strip().lower()
            self._log(f"模型类型: {model_type}")

            model = build_model(num_classes, device, model_type=model_type)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=float(self._config.get("lr", 1e-3)))

            val_mapper = None
            test_mapper = None
            if val_ds is not None:
                val_mapper = build_label_mapper(val_ds.class_to_idx, train_ds.class_to_idx, device)
            if test_ds is not None:
                test_mapper = build_label_mapper(test_ds.class_to_idx, train_ds.class_to_idx, device)

            epochs = int(self._config.get("epochs", 1))
            batch_size = int(self._config.get("batch_size", 64))
            num_workers = int(self._config.get("num_workers", 0))
            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
            )
            val_loader = None
            test_loader = None
            if val_ds is not None and len(val_ds) > 0:
                val_loader = torch.utils.data.DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                )
            if test_ds is not None and len(test_ds) > 0:
                test_loader = torch.utils.data.DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    drop_last=False,
                )

            output_path = str(self._config.get("output_path", "cbam_lstm_best.onnx")).strip()
            onnx_path = with_depth_suffix(output_path, depth)
            onnx_dir = Path(onnx_path).parent
            if str(onnx_dir):
                onnx_dir.mkdir(parents=True, exist_ok=True)

            best_model_path = with_depth_suffix(str(Path(onnx_path).with_suffix(".pth")), depth)
            curves_path = str(Path(onnx_path).with_name(f"{Path(onnx_path).stem}_learning_curves.png"))
            val_cm_path = str(Path(onnx_path).with_name(f"{Path(onnx_path).stem}_confusion_matrix_val.png"))
            test_cm_path = str(Path(onnx_path).with_name(f"{Path(onnx_path).stem}_confusion_matrix_test.png"))

            history = {
                "train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": [],
            }

            best_metric = -1.0
            self._log(f"开始训练: epochs={epochs}, batch_size={batch_size}, lr={float(self._config.get('lr', 1e-3))}")

            for epoch in range(1, epochs + 1):
                train_loss, train_acc = run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=optimizer,
                    label_mapper=None,
                    model_type=model_type,
                )
                history["train_loss"].append(float(train_loss))
                history["train_acc"].append(float(train_acc))

                if val_loader is not None:
                    val_loss, val_acc = run_epoch(
                        model=model,
                        loader=val_loader,
                        criterion=criterion,
                        device=device,
                        optimizer=None,
                        label_mapper=val_mapper,
                        model_type=model_type,
                    )
                    history["val_loss"].append(float(val_loss))
                    history["val_acc"].append(float(val_acc))
                    metric = float(val_acc)
                    self._log(
                        f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )
                else:
                    metric = float(train_acc)
                    self._log(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

                self.progressChanged.emit(int(round(epoch / max(1, epochs) * 100)))

                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), best_model_path)
                    self._log(f"保存最佳模型: {best_model_path}")

            if Path(best_model_path).exists():
                model.load_state_dict(torch.load(best_model_path, map_location=device))

            save_training_curves(history, curves_path)
            self._log(f"保存训练曲线: {curves_path}")

            if val_loader is not None:
                y_true_val, y_pred_val = collect_predictions(
                    model=model,
                    loader=val_loader,
                    device=device,
                    label_mapper=val_mapper,
                    model_type=model_type,
                )
                if y_true_val.size > 0:
                    save_confusion_matrix(
                        y_true=y_true_val,
                        y_pred=y_pred_val,
                        class_names=train_ds.classes,
                        save_path=val_cm_path,
                        title="Validation Confusion Matrix",
                    )
                    self._log(f"保存验证集混淆矩阵: {val_cm_path}")

            export_onnx_model(model, onnx_path, device, depth, model_type=model_type)
            metadata_path = export_onnx_metadata(
                onnx_path=onnx_path,
                depth=depth,
                classes=train_ds.classes,
                model_type=model_type,
            )
            self._log(f"导出 ONNX: {onnx_path}")
            self._log(f"导出元数据: {metadata_path}")

            test_loss = None
            test_acc = None
            if test_loader is not None:
                test_loss, test_acc = run_epoch(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                    label_mapper=test_mapper,
                    model_type=model_type,
                )
                self._log(f"测试集: loss={test_loss:.4f}, acc={test_acc:.4f}")

                y_true_test, y_pred_test = collect_predictions(
                    model=model,
                    loader=test_loader,
                    device=device,
                    label_mapper=test_mapper,
                    model_type=model_type,
                )
                if y_true_test.size > 0:
                    save_confusion_matrix(
                        y_true=y_true_test,
                        y_pred=y_pred_test,
                        class_names=train_ds.classes,
                        save_path=test_cm_path,
                        title="Test Confusion Matrix",
                    )
                    self._log(f"保存测试集混淆矩阵: {test_cm_path}")

            self.finishedSuccessfully.emit(
                {
                    "best_model_path": best_model_path,
                    "onnx_path": onnx_path,
                    "metadata_path": metadata_path,
                    "curves_path": curves_path,
                    "val_confusion_matrix_path": val_cm_path if Path(val_cm_path).exists() else "",
                    "test_confusion_matrix_path": test_cm_path if Path(test_cm_path).exists() else "",
                    "depth": depth,
                    "classes": list(train_ds.classes),
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
        except Exception as error:
            self.failed.emit(str(error))
