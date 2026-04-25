from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from qfluentwidgets import BodyLabel, ComboBox, FluentIcon, HeaderCardWidget, PrimaryPushButton, PushButton


class ConnectionCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("设备连接")
        self.vBoxLayout = QVBoxLayout()

        connectionRow = QHBoxLayout()
        connectionRow.setSpacing(10)

        self.portCombo = ComboBox()
        self.portCombo.setMinimumWidth(180)
        connectionRow.addWidget(self.portCombo)

        self.refreshButton = PushButton(FluentIcon.SYNC, "刷新串口")
        connectionRow.addWidget(self.refreshButton)

        self.startButton = PrimaryPushButton(FluentIcon.PLAY_SOLID, "开始接收")
        connectionRow.addWidget(self.startButton)

        self.stopButton = PushButton(FluentIcon.PAUSE, "停止")
        self.stopButton.setEnabled(False)
        connectionRow.addWidget(self.stopButton)

        connectionRow.addStretch()
        self.vBoxLayout.addLayout(connectionRow)

        powerRow = QHBoxLayout()
        powerRow.setSpacing(10)
        self.devVoltageLabel = BodyLabel("总线电压: V")
        powerRow.addWidget(self.devVoltageLabel)
        self.devCurrentLabel = BodyLabel("电流: mA")
        powerRow.addWidget(self.devCurrentLabel)
        self.devPowerLabel = BodyLabel("功率: mW")
        powerRow.addWidget(self.devPowerLabel)
        powerRow.addStretch()
        self.vBoxLayout.addLayout(powerRow)

        self.viewLayout.addLayout(self.vBoxLayout)