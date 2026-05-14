"""Parameter card for online analysis model and acquisition settings."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QVBoxLayout
from qfluentwidgets import FluentIcon, HeaderCardWidget, LineEdit, PushButton, SpinBox, DoubleSpinBox


class ClickableLineEdit(LineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, a0):
        self.clicked.emit()
        super().mousePressEvent(a0)


class AnalyzeParamCard(HeaderCardWidget):
    modelSelectRequested = pyqtSignal()
    applyRequested = pyqtSignal(int, int)
    alphaChanged = pyqtSignal(float)
    voteWindowChanged = pyqtSignal(int)

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

        self.emaAlphaLabel = QLabel("EMA α")
        config_row.addWidget(self.emaAlphaLabel)

        self.emaAlphaSpin = DoubleSpinBox(self)
        self.emaAlphaSpin.setRange(0.0, 1.0)
        self.emaAlphaSpin.setSingleStep(0.05)
        self.emaAlphaSpin.setDecimals(2)
        self.emaAlphaSpin.setValue(0.5)
        self.emaAlphaSpin.setFixedWidth(120)
        config_row.addWidget(self.emaAlphaSpin)

        self.voteWindowLabel = QLabel("投票窗口 N")
        config_row.addWidget(self.voteWindowLabel)

        self.voteWindowSpin = SpinBox(self)
        self.voteWindowSpin.setRange(1, 99)
        self.voteWindowSpin.setValue(5)
        self.voteWindowSpin.setFixedWidth(120)
        config_row.addWidget(self.voteWindowSpin)

        self.applyButton = PushButton(FluentIcon.SEND, "应用")
        config_row.addWidget(self.applyButton)
        config_row.addStretch()

        self.vBoxLayout.addLayout(config_row)
        self.viewLayout.addLayout(self.vBoxLayout)

        self.modelPathEdit.clicked.connect(self.modelSelectRequested.emit)
        self.modelBrowseButton.clicked.connect(self.modelSelectRequested.emit)
        self.applyButton.clicked.connect(self._onApplyClicked)
        self.emaAlphaSpin.valueChanged.connect(lambda value: self.alphaChanged.emit(float(value)))
        self.voteWindowSpin.valueChanged.connect(lambda value: self.voteWindowChanged.emit(int(value)))

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

    def setEmaAlpha(self, alpha: float):
        self.emaAlphaSpin.setValue(max(0.0, min(1.0, float(alpha))))

    def getEmaAlpha(self) -> float:
        return float(self.emaAlphaSpin.value())

    def setVoteWindowSize(self, n: int):
        try:
            value = int(n)
        except (TypeError, ValueError):
            value = 5
        self.voteWindowSpin.setValue(max(1, min(99, value)))

    def getVoteWindowSize(self) -> int:
        return int(self.voteWindowSpin.value())

    def _onApplyClicked(self):
        scan_freq_hz = int(self.scanFreqSpin.value())
        depth_text = (self.sampleDepthEdit.text() or "").strip()
        try:
            sample_depth = int(depth_text)
        except ValueError:
            sample_depth = 0

        self._sampling_depth = max(0, sample_depth)
        self.applyRequested.emit(scan_freq_hz, self._sampling_depth)
