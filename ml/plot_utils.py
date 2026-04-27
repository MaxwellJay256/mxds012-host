import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


_MATPLOTLIB_AVAILABLE = False
_plt = None


def _get_plt():
    global _MATPLOTLIB_AVAILABLE, _plt
    if _plt is not None:
        return _plt

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _MATPLOTLIB_AVAILABLE = True
        _plt = plt
        return _plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to export training plots. Install matplotlib or skip plot export."
        ) from error


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def collect_labels_and_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_transform: Callable[[torch.Tensor], torch.Tensor],
    label_mapper: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect true labels and predicted labels from a loader."""
    model.eval()

    y_true_list: List[torch.Tensor] = []
    y_pred_list: List[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            x = input_transform(x).to(device)
            y = y.to(device).long()

            if label_mapper is not None:
                mapped = label_mapper[y]
                if (mapped < 0).any():
                    bad = y[mapped < 0].unique().tolist()
                    raise ValueError(f"Found labels that cannot map to train classes: {bad}")
                y = mapped

            logits = model(x)
            pred = logits.argmax(dim=1)

            y_true_list.append(y.cpu())
            y_pred_list.append(pred.cpu())

    if not y_true_list:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    y_true = torch.cat(y_true_list, dim=0).numpy().astype(np.int64)
    y_pred = torch.cat(y_pred_list, dim=0).numpy().astype(np.int64)
    return y_true, y_pred


def plot_learning_curves(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot and save loss/accuracy curves."""
    plt = _get_plt()
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    epochs_train = np.arange(1, len(train_loss) + 1)
    epochs_val_loss = np.arange(1, len(val_loss) + 1)
    epochs_val_acc = np.arange(1, len(val_acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if len(train_loss) > 0:
        plt.plot(epochs_train, train_loss, label="train_loss")
    if len(val_loss) > 0:
        plt.plot(epochs_val_loss, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    if len(train_acc) > 0:
        plt.plot(epochs_train, train_acc, label="train_acc")
    if len(val_acc) > 0:
        plt.plot(epochs_val_acc, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save normalized confusion matrix."""
    plt = _get_plt()
    num_classes = len(class_names)

    cm_counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm_counts[t, p] += 1.0

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts), where=row_sums > 0)

    plt.figure(figsize=(10, 8), dpi=120)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            if val > 1e-3:
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=200)
    plt.close()
