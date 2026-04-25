"""Parameter card for online analysis model and acquisition settings."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout
from qfluentwidgets import FluentIcon, HeaderCardWidget, LineEdit, PushButton, SpinBox


class ClickableLineEdit(LineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, a0):
        self.clicked.emit()
        super().mousePressEvent(a0)


class AnalyzeParamCard(HeaderCardWidget):
    modelSelectRequested = pyqtSignal()
    applyRequested = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("识别参数")

        self._sampling_depth = 0
        self.vBoxLayout = QVBoxLayout()
        self.vBoxLayout.setSpacing(10)

        model_row = QHBoxLayout()
        model_row.setSpacing(10)

        self.modelPathLabel = QLabel("模型文件")
        model_row.addWidget(self.modelPathLabel)

        self.modelPathEdit = ClickableLineEdit(self)
        self.modelPathEdit.setReadOnly(True)
        self.modelPathEdit.setPlaceholderText("点击选择 onnx 模型")
        model_row.addWidget(self.modelPathEdit, 1)

        self.modelBrowseButton = PushButton(FluentIcon.FOLDER, "选择模型")
        model_row.addWidget(self.modelBrowseButton)

        self.vBoxLayout.addLayout(model_row)

        config_row = QHBoxLayout()
        config_row.setSpacing(10)

        self.scanFreqLabel = QLabel("扫描频率 (Hz)")
        config_row.addWidget(self.scanFreqLabel)

        self.scanFreqSpin = SpinBox(self)
        self.scanFreqSpin.setRange(1, 60)
        self.scanFreqSpin.setValue(20)
        self.scanFreqSpin.setFixedWidth(120)
        config_row.addWidget(self.scanFreqSpin)

        self.sampleDepthLabel = QLabel("采样深度")
        config_row.addWidget(self.sampleDepthLabel)

        self.sampleDepthEdit = LineEdit(self)
        self.sampleDepthEdit.setReadOnly(True)
        self.sampleDepthEdit.setPlaceholderText("自动读取")
        self.sampleDepthEdit.setFixedWidth(140)
        config_row.addWidget(self.sampleDepthEdit)

        self.applyButton = PushButton(FluentIcon.SEND, "应用")
        config_row.addWidget(self.applyButton)
        config_row.addStretch()

        self.vBoxLayout.addLayout(config_row)
        self.viewLayout.addLayout(self.vBoxLayout)

        self.modelPathEdit.clicked.connect(self.modelSelectRequested.emit)
        self.modelBrowseButton.clicked.connect(self.modelSelectRequested.emit)
        self.applyButton.clicked.connect(self._onApplyClicked)

    def setModelPath(self, path: str):
        self.modelPathEdit.setText(path or "")

    def setScanFrequency(self, scan_freq_hz: int):
        self.scanFreqSpin.setValue(max(1, min(60, int(scan_freq_hz))))

    def setSamplingDepth(self, depth: Optional[int]):
        if depth is None or int(depth) <= 0:
            self._sampling_depth = 0
            self.sampleDepthEdit.setText("")
            return

        self._sampling_depth = int(depth)
        self.sampleDepthEdit.setText(str(self._sampling_depth))

    def getSamplingDepth(self) -> int:
        text = (self.sampleDepthEdit.text() or "").strip()
        try:
            depth = int(text)
        except ValueError:
            depth = self._sampling_depth
        return max(0, int(depth))

    def _onApplyClicked(self):
        scan_freq_hz = int(self.scanFreqSpin.value())
        depth_text = (self.sampleDepthEdit.text() or "").strip()
        try:
            sample_depth = int(depth_text)
        except ValueError:
            sample_depth = 0

        self._sampling_depth = max(0, sample_depth)
        self.applyRequested.emit(scan_freq_hz, self._sampling_depth)
