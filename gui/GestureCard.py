"""Gesture presentation card for online analysis view."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from qfluentwidgets import BodyLabel, ElevatedCardWidget


class GestureCard(ElevatedCardWidget):
    def __init__(self, gesture_name: str, parent=None):
        super().__init__(parent)
        self._gesture_name = gesture_name
        self._selected = False
        self._probability = 0.0

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(6)

        self.imagePlaceholder = QLabel("图片待添加")
        self.imagePlaceholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imagePlaceholder.setFixedSize(80, 80)
        self._applyPlaceholderStyle()

        self.nameLabel = BodyLabel(gesture_name)
        self.nameLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.probabilityLabel = BodyLabel("0.0%")
        self.probabilityLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        root_layout.addWidget(self.imagePlaceholder, 0, Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.nameLabel, 0, Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.probabilityLabel, 0, Qt.AlignmentFlag.AlignCenter)

        self.setMinimumWidth(120)
        self.setMaximumWidth(140)
        self._loadGestureImage()
        self.setSelected(False)

    def _gestureImagePath(self) -> Path:
        return Path(__file__).resolve().parents[1] / "assets" / "gesture_figures" / f"{self._gesture_name}.png"

    def _applyPlaceholderStyle(self):
        self.imagePlaceholder.setStyleSheet(
            "QLabel {"
            " border: 1px dashed #999999;"
            " border-radius: 8px;"
            " color: #777777;"
            " background-color: rgba(0, 0, 0, 0.02);"
            "}"
        )

    def _loadGestureImage(self):
        image_path = self._gestureImagePath()
        if not image_path.exists():
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return

        scaled = pixmap.scaled(
            self.imagePlaceholder.width(),
            self.imagePlaceholder.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.imagePlaceholder.setPixmap(scaled)
        self.imagePlaceholder.setText("")
        self.imagePlaceholder.setStyleSheet("QLabel { border: none; background: transparent; }")

    def gestureName(self) -> str:
        return self._gesture_name

    def setProbability(self, probability: float):
        value = max(0.0, min(1.0, float(probability)))
        self._probability = value
        self.probabilityLabel.setText(f"{value * 100.0:.1f}%")

    def setSelected(self, selected: bool):
        self._selected = bool(selected)
        if self._selected:
            self.nameLabel.setStyleSheet("font-weight: 700;")
            self.setStyleSheet(
                "GestureCard {"
                " border: 1px solid #e4360f;"
                " border-radius: 10px;"
                " background-color: rgba(228, 54, 15, 0.08);"
                "}"
            )
        else:
            self.nameLabel.setStyleSheet("font-weight: 500;")
            self.setStyleSheet(
                "GestureCard {"
                " border: 1px solid rgba(0, 0, 0, 0.08);"
                " border-radius: 10px;"
                "}"
            )
