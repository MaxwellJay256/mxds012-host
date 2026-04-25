"""Gesture panel card with flow layout for model classes."""

from __future__ import annotations

from typing import Dict, Iterable, List

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, FlowLayout, HeaderCardWidget

from .GestureCard import GestureCard


class AdaptiveFlowLayout(FlowLayout):
    """Compatibility alias for requested adaptive flow layout behavior."""


class GesturePanelCard(HeaderCardWidget):
    PRESET_GESTURES: List[str] = [
        "WF", "WE", "WRD", "WUD", "WS", "WP", "FS", "IP", "IMP", "TIF",
        "TF", "FA", "TU", "IF", "MF", "IMF", "MRF", "RLF", "PH", "RS",
    ]

    def __init__(self, parent=None):  # type: ignore[override]
        super().__init__(parent)
        self.setTitle("手势类别")

        self._cards: Dict[str, GestureCard] = {}
        self._last_model_summary = "请先选择 onnx 模型"

        self.vBoxLayout = QVBoxLayout()
        self.vBoxLayout.setSpacing(10)

        self.summaryLabel = BodyLabel(self._last_model_summary)
        self.vBoxLayout.addWidget(self.summaryLabel)

        self.flowContainer = QWidget(self)
        self.flowLayout = AdaptiveFlowLayout(self.flowContainer, needAni=False)
        self.flowLayout.setContentsMargins(0, 0, 0, 0)
        self.flowLayout.setHorizontalSpacing(10)
        self.flowLayout.setVerticalSpacing(10)

        self.emptyLabel = QLabel("暂无类别")
        self.vBoxLayout.addWidget(self.flowContainer)
        self.vBoxLayout.addWidget(self.emptyLabel)
        self.vBoxLayout.addStretch(1)

        self.viewLayout.addLayout(self.vBoxLayout)
        self._setEmptyVisible(True)

    def _setEmptyVisible(self, visible: bool):
        self.emptyLabel.setVisible(bool(visible))

    def _clearCards(self):
        while self.flowLayout.count() > 0:
            item = self.flowLayout.takeAt(0)
            if item is None:
                continue
            # qfluentwidgets FlowLayout 在不同版本下 takeAt 可能返回 QWidget 或 QLayoutItem。
            if hasattr(item, "widget"):
                widget = item.widget()
            else:
                widget = item
            if widget is not None:
                widget.deleteLater()
        self._cards.clear()

    def populateByClasses(self, classes: Iterable[str], num_classes: int):
        class_list = [str(name).strip() for name in classes if str(name).strip()]
        valid_set = set(self.PRESET_GESTURES)
        filtered = [name for name in class_list if name in valid_set]

        self._clearCards()

        for name in filtered:
            card = GestureCard(name, self.flowContainer)
            card.setProbability(0.0)
            self.flowLayout.addWidget(card)
            self._cards[name] = card

        self._setEmptyVisible(len(filtered) == 0)
        self._last_model_summary = f"模型类别: classes={len(class_list)} | num_classes={int(num_classes)} | 已显示={len(filtered)}"
        self.summaryLabel.setText(self._last_model_summary)

    def updateProbabilities(self, probabilities: Dict[str, float], top_name: str = "", threshold: float = 0.5):
        probs = probabilities or {}
        best_name = ""
        best_prob = -1.0

        for name, card in self._cards.items():
            value = float(probs.get(name, 0.0) or 0.0)
            card.setProbability(value)
            if value > best_prob:
                best_prob = value
                best_name = name

        chosen_top = str(top_name or "").strip() or best_name
        active_name = ""
        if chosen_top in self._cards:
            chosen_prob = float(probs.get(chosen_top, 0.0) or 0.0)
            if chosen_prob >= float(threshold):
                active_name = chosen_top

        self.setActiveGesture(active_name)

        if best_name:
            self.summaryLabel.setText(
                f"{self._last_model_summary} | Top1={best_name} {max(0.0, best_prob) * 100.0:.1f}%"
            )
        else:
            self.summaryLabel.setText(self._last_model_summary)

    def setActiveGesture(self, gesture_name: str):
        target = str(gesture_name).strip()
        for name, card in self._cards.items():
            card.setSelected(name == target)
