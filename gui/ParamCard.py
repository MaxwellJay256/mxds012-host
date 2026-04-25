from PyQt5.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout
from qfluentwidgets import FluentIcon, HeaderCardWidget, PushButton, SpinBox


class ParamCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("参数设置")
        self.vBoxLayout = QVBoxLayout()

        paramRow = QHBoxLayout()
        paramRow.setSpacing(10)

        self.scanFreqLabel = QLabel("扫描频率 (Hz)")
        paramRow.addWidget(self.scanFreqLabel)

        self.scanFreqSpin = SpinBox(self)
        self.scanFreqSpin.setRange(1, 60)
        self.scanFreqSpin.setValue(20)
        self.scanFreqSpin.setFixedWidth(120)
        paramRow.addWidget(self.scanFreqSpin)

        self.sampleDepthLabel = QLabel("采样深度")
        paramRow.addWidget(self.sampleDepthLabel)

        self.sampleDepthSpin = SpinBox(self)
        self.sampleDepthSpin.setRange(1, 1024)
        self.sampleDepthSpin.setValue(128)
        self.sampleDepthSpin.setFixedWidth(120)
        paramRow.addWidget(self.sampleDepthSpin)

        self.applyParamButton = PushButton(FluentIcon.SEND, "应用")
        paramRow.addWidget(self.applyParamButton)
        paramRow.addStretch()

        self.vBoxLayout.addLayout(paramRow)
        self.viewLayout.addLayout(self.vBoxLayout)