"""Recording control page for the function area."""
from typing import Optional

from pathlib import Path

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QGroupBox,
)
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from serial.tools import list_ports

from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    FluentIcon,
    LineEdit,
    PrimaryPushButton,
    ProgressRing,
    PushButton,
    SpinBox, 
    ToolButton
)

from RecordingService import RecordingService
from RecordingUtils import RecordingError, buildOutputPath, checkFileExists
from SettingService import SettingService


class FolderPathLineEdit(LineEdit):
    pathClicked = pyqtSignal()

    def mousePressEvent(self, a0):
        self.pathClicked.emit()
        super().mousePressEvent(a0)


class RecordingWidget(QWidget):
    """Recording and serial control page."""

    startSerialRequested = pyqtSignal(str)
    stopSerialRequested = pyqtSignal()
    infoRequested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.progressRingFormat = "0.0 s"
        self.serialRunning = False
        self.currentScanFreq = 20  # Current scan frequency in Hz (default 20)

        self.isRecording = False
        self.isRecordPausedPendingSave = False
        self.recordingService: Optional[RecordingService] = None
        self.recordedFrameCount = 0
        self.recordOutputDir = ""
        self.recordFrameCounter = 0
        self.recordFilePath = ""
        self.currentRecordMode: Optional[str] = None
        self.timedTargetFrames = 0
        self._isApplyingSettings = False
        self._pendingLastPort = ""

        self.settings = SettingService.get_settings()

        self.timedProgressTimer = QTimer(self)
        self.timedProgressTimer.setInterval(100)
        self.timedProgressTimer.timeout.connect(self.onTimedProgressTick)

        self.configWaitTimer = QTimer(self)
        self.configWaitTimer.setSingleShot(True)
        self.configWaitTimer.setInterval(1000)
        self.configWaitTimer.timeout.connect(self.onConfigWaitTimeout)

        rootLayout = QVBoxLayout(self)
        rootLayout.setContentsMargins(12, 12, 12, 12)
        rootLayout.setSpacing(12)

        connectionGroup = QGroupBox("设备连接")
        connectionLayout = QVBoxLayout(connectionGroup)
        connectionLayout.setContentsMargins(12, 10, 12, 10)

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
        connectionLayout.addLayout(connectionRow)

        powerRow = QHBoxLayout()
        powerRow.setSpacing(10)

        self.devVoltageLabel = BodyLabel("总线电压: V")
        powerRow.addWidget(self.devVoltageLabel)

        self.devCurrentLabel = BodyLabel("电流: mA")
        powerRow.addWidget(self.devCurrentLabel)

        self.devPowerLabel = BodyLabel("功率: mW")
        powerRow.addWidget(self.devPowerLabel)
        powerRow.addStretch()

        connectionLayout.addLayout(powerRow)
        rootLayout.addWidget(connectionGroup)

        paramGroup = QGroupBox("参数设置")
        paramLayout = QVBoxLayout(paramGroup)
        paramLayout.setContentsMargins(12, 10, 12, 10)

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

        paramLayout.addLayout(paramRow)
        rootLayout.addWidget(paramGroup)

        outputGroup = QGroupBox("录制输出")
        outputLayout = QVBoxLayout(outputGroup)
        outputLayout.setContentsMargins(12, 10, 12, 10)

        self.recordDirButton = ToolButton(FluentIcon.FOLDER)
        self.outputPathEdit = FolderPathLineEdit()
        self.outputPathEdit.setPlaceholderText("录制目录路径")
        self.outputPathEdit.setReadOnly(True)
        self.outputPathEdit.setText("")
        self.filenameEdit = LineEdit()
        self.filenameEdit.setPlaceholderText("输入文件名 (不需要 .txt)")

        recordPathRow = QHBoxLayout()
        recordPathRow.setSpacing(10)
        recordPathRow.addWidget(self.recordDirButton)
        recordPathRow.addWidget(self.outputPathEdit, 2)
        recordPathRow.addWidget(self.filenameEdit, 1)
        outputLayout.addLayout(recordPathRow)
        rootLayout.addWidget(outputGroup)

        self.recordContinuousStartButton = PushButton("开始连续录制")
        self.recordContinuousStartButton.setEnabled(False)

        self.recordDurationSpin = SpinBox()
        self.recordDurationSpin.setRange(1, 60)
        self.recordDurationSpin.setValue(5)
        self.recordDurationSpin.setSuffix(" s")
        self.recordDurationSpin.setToolTip("定时录制秒数，实际录制帧数=秒数x当前扫描频率")

        self.recordTimedStartButton = PushButton("开始定时录制")
        self.recordTimedStartButton.setEnabled(False)

        self.recordStopButton = PushButton("停止录制")
        self.recordStopButton.setEnabled(False)

        recordActionRow = QHBoxLayout()
        recordActionRow.setSpacing(10)
        recordActionRow.addWidget(self.recordContinuousStartButton)
        recordActionRow.addWidget(self.recordDurationSpin)
        recordActionRow.addWidget(self.recordTimedStartButton)
        recordActionRow.addWidget(self.recordStopButton)
        rootLayout.addLayout(recordActionRow)

        statusRow = QHBoxLayout()
        statusRow.setSpacing(12)

        self.recordProgressRing = ProgressRing(self)
        self.recordProgressRing.setRange(0, 100)
        self.recordProgressRing.setValue(0)
        self.recordProgressRing.setTextVisible(True)
        self.recordProgressRing.setFixedSize(60, 60)
        self.recordProgressRing.setStrokeWidth(4)
        self.recordProgressRing.setFormat(self.progressRingFormat)
        statusRow.addWidget(self.recordProgressRing)

        self.frameCounterLabel = QLabel("已录制: 0 帧")
        statusRow.addWidget(self.frameCounterLabel)
        statusRow.addStretch()
        rootLayout.addLayout(statusRow)

        rootLayout.addStretch()

        self.refreshButton.clicked.connect(self.refreshPorts)
        self.startButton.clicked.connect(self.onStartSerialClicked)
        self.stopButton.clicked.connect(self.onStopSerialClicked)
        self.recordDirButton.clicked.connect(self.selectOutputDirectory)
        self.outputPathEdit.pathClicked.connect(self.onOutputPathEditClicked)
        self.applyParamButton.clicked.connect(self.onApplyParamClicked)
        self.recordContinuousStartButton.clicked.connect(self.onRecordStartContinuous)
        self.recordTimedStartButton.clicked.connect(self.onRecordStartTimed)
        self.recordStopButton.clicked.connect(self.onRecordStop)
        self.portCombo.currentIndexChanged.connect(self.onPortSelectionChanged)
        self.scanFreqSpin.valueChanged.connect(self.onScanFreqChanged)
        self.sampleDepthSpin.valueChanged.connect(self.onSampleDepthChanged)
        self.filenameEdit.editingFinished.connect(self.onFilenameEditingFinished)

        self.loadSettings()
        self.refreshPorts()
        self.updateRecordingUiState()

    def loadSettings(self):
        self._isApplyingSettings = True
        try:
            scanFreq = self.settings.value("Hardware/ScanFrequencyHz", 20, type=int)
            sampleDepth = self.settings.value("Hardware/SampleDepth", 128, type=int)
            outputDir = self.settings.value("Recording/OutputDirectory", "", type=str)
            filename = self.settings.value("Recording/FileName", "", type=str)
            self._pendingLastPort = self.settings.value("Serial/LastPort", "", type=str)

            self.scanFreqSpin.setValue(max(1, min(60, scanFreq)))
            self.sampleDepthSpin.setValue(max(1, min(1024, sampleDepth)))
            self.filenameEdit.setText(filename)

            if outputDir and Path(outputDir).exists():
                self.recordOutputDir = outputDir
                self.outputPathEdit.setText(outputDir)
            else:
                self.recordOutputDir = ""
                self.outputPathEdit.setText("")
                if outputDir:
                    self.showInfo("上次录制目录不存在，已清空，请重新选择", "warning")
        finally:
            self._isApplyingSettings = False

    def _saveSetting(self, key: str, value):
        self.settings.setValue(key, value)
        self.settings.sync()

    def saveAllSettings(self):
        port = self.portCombo.currentData()
        self._saveSetting("Serial/LastPort", port if port else "")
        self._saveSetting("Hardware/ScanFrequencyHz", int(self.scanFreqSpin.value()))
        self._saveSetting("Hardware/SampleDepth", int(self.sampleDepthSpin.value()))
        self._saveSetting("Recording/OutputDirectory", self.recordOutputDir.strip())
        self._saveSetting("Recording/FileName", self.filenameEdit.text().strip())

    def onPortSelectionChanged(self, _index: int):
        if self._isApplyingSettings:
            return
        port = self.portCombo.currentData()
        if port:
            self._saveSetting("Serial/LastPort", port)

    def onScanFreqChanged(self, value: int):
        if self._isApplyingSettings:
            return
        self._saveSetting("Hardware/ScanFrequencyHz", int(value))

    def onSampleDepthChanged(self, value: int):
        if self._isApplyingSettings:
            return
        self._saveSetting("Hardware/SampleDepth", int(value))

    def onFilenameEditingFinished(self):
        if self._isApplyingSettings:
            return
        self._saveSetting("Recording/FileName", self.filenameEdit.text().strip())

    def showInfo(self, message: str, level: str = "info"):
        self.infoRequested.emit(message, level)

    def refreshPorts(self):
        previousDevice = self.portCombo.currentData()
        self.portCombo.clear()

        ports = list_ports.comports()
        for portInfo in ports:
            displayText = f"{portInfo.device} - {portInfo.description}"
            self.portCombo.addItem(displayText, portInfo.device)

        if not ports:
            self.showInfo("未检测到串口设备", "warning")
            return

        preferredDevice = previousDevice or self._pendingLastPort
        if preferredDevice:
            index = self.portCombo.findData(preferredDevice)
            if index >= 0:
                self._isApplyingSettings = True
                self.portCombo.setCurrentIndex(index)
                self._isApplyingSettings = False

        currentPort = self.portCombo.currentData()
        if currentPort:
            self._saveSetting("Serial/LastPort", currentPort)

    def onStartSerialClicked(self):
        if self.serialRunning:
            self.showInfo("串口已在运行", "warning")
            return

        if self.portCombo.count() == 0:
            self.showInfo("没有可用串口", "warning")
            return

        port = self.portCombo.currentData()
        if not port:
            display = self.portCombo.currentText().strip()
            if display:
                portCandidate = display.split()[0]
                if portCandidate:
                    port = portCandidate

        if not port:
            self.showInfo("请选择串口", "warning")
            return

        # self.showInfo(f"已选择串口: {port}", "info")
        self.startSerialRequested.emit(port)

    def onStopSerialClicked(self):
        self.stopSerialRequested.emit()

    def onSerialStarted(self):
        self.serialRunning = True
        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.currentScanFreq = self.scanFreqSpin.value()
        self.clearPowerLabels()
        self.updateRecordingUiState()

    def onSerialStopped(self):
        self.serialRunning = False
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.clearPowerLabels()
        self.updateRecordingUiState()

    def clearPowerLabels(self):
        self.devVoltageLabel.setText("总线电压: V")
        self.devCurrentLabel.setText("电流: mA")
        self.devPowerLabel.setText("功率: mW")

    def onPowerReady(self, powerData: dict):
        busMilliVolt = int(powerData.get("bus_mv", 0))
        currentMilliAmp = int(powerData.get("current_ma", 0))
        powerMilliWatt = int(powerData.get("power_mw", 0))

        busVolt = busMilliVolt / 1000.0
        self.devVoltageLabel.setText(f"总线电压: {busVolt:.2f}V")
        self.devCurrentLabel.setText(f"电流: {currentMilliAmp}mA")
        self.devPowerLabel.setText(f"功率: {powerMilliWatt}mW")

    def selectOutputDirectory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择录制文件保存目录",
            "",
        )
        if directory:
            self.recordOutputDir = directory
            self.outputPathEdit.setText(directory)
            self._saveSetting("Recording/OutputDirectory", directory)
            # self.showInfo(f"录制目录: {directory}", "info")
        else:
            self.recordOutputDir = ""
            self.outputPathEdit.setText("")
            self._saveSetting("Recording/OutputDirectory", "")

    def onOutputPathEditClicked(self):
        path = self.outputPathEdit.text().strip()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def getRecordingState(self) -> str:
        if not self.recordingService:
            return RecordingService.STATE_IDLE
        return self.recordingService.getState()

    def prepareNewRecordingService(self) -> bool:
        if not self.recordOutputDir:
            self.showInfo("请先选择录制目录", "warning")
            return False

        filename = self.filenameEdit.text().strip()
        if not filename:
            self.showInfo("请输入文件名", "warning")
            return False

        fullPath, errorMsg = buildOutputPath(self.recordOutputDir, filename)
        if errorMsg:
            self.showInfo(f"路径错误: {errorMsg}", "error")
            return False

        if checkFileExists(fullPath):
            self.showInfo(f"文件已存在，将覆盖: {fullPath}", "warning")

        self.recordFilePath = fullPath
        self.recordingService = RecordingService(fullPath)
        self.recordedFrameCount = 0
        self.recordFrameCounter = 0
        self.updateFrameCounter()
        return True

    def startTimedProgress(self, selectedSeconds: int):
        maxTenths = int(selectedSeconds * 10)
        self.recordProgressRing.setRange(0, maxTenths)
        self.recordProgressRing.setValue(maxTenths)
        self.updateCountdownDisplay(self.timedTargetFrames)
        self.timedProgressTimer.start()

    def updateCountdownDisplay(self, remainingFrames: int):
        remainingSeconds = max(0.0, remainingFrames / float(self.currentScanFreq))
        remainingTenths = max(0, int(round(remainingSeconds * 10)))
        self.recordProgressRing.setValue(remainingTenths)
        self.recordProgressRing.setFormat(f"{remainingSeconds:.1f} s")

    def onRecordStartContinuous(self):
        if not self.serialRunning:
            self.showInfo("请先连接串口", "warning")
            return

        state = self.getRecordingState()
        if state == RecordingService.STATE_ACTIVE_CONTINUOUS:
            self.showInfo("连续录制已在进行中", "warning")
            return
        if state == RecordingService.STATE_ACTIVE_TIMED:
            self.showInfo("定时录制进行中，无法切换模式", "warning")
            return

        try:
            if state == RecordingService.STATE_PAUSED_PENDING_SAVE and self.recordingService:
                self.recordingService.startContinuous(new_session=False)
                self.showInfo("已从暂停状态恢复为连续录制", "info")
            else:
                if not self.prepareNewRecordingService():
                    return
                service = self.recordingService
                if service is None:
                    return
                service.startContinuous(new_session=True)
                self.showInfo(f"开始连续录制: {self.recordFilePath}", "info")

            self.currentRecordMode = "continuous"
            self.isRecording = True
            self.isRecordPausedPendingSave = False
            self.timedTargetFrames = 0
            self.timedProgressTimer.stop()
            self.recordProgressRing.setRange(0, 100)
            self.recordProgressRing.setValue(0)
            self.recordProgressRing.setFormat(self.progressRingFormat)
            self.updateRecordingUiState()
        except RecordingError as error:
            self.showInfo(f"连续录制启动失败: {error}", "error")
            self.isRecording = False
            self.updateRecordingUiState()

    def onRecordStartTimed(self):
        if not self.serialRunning:
            self.showInfo("请先连接串口", "warning")
            return

        selectedSeconds = int(self.recordDurationSpin.value())
        targetFrames = selectedSeconds * self.currentScanFreq
        state = self.getRecordingState()

        if state == RecordingService.STATE_ACTIVE_TIMED:
            self.showInfo("定时录制已在进行中", "warning")
            return
        if state == RecordingService.STATE_ACTIVE_CONTINUOUS:
            self.showInfo("连续录制进行中，无法切换模式", "warning")
            return

        try:
            if state == RecordingService.STATE_PAUSED_PENDING_SAVE and self.recordingService:
                self.recordingService.startTimed(target_frames=targetFrames, new_session=False)
                self.showInfo(f"恢复定时录制: {selectedSeconds}s ({targetFrames} 帧)", "info")
            else:
                if not self.prepareNewRecordingService():
                    return
                service = self.recordingService
                if service is None:
                    return
                service.startTimed(target_frames=targetFrames, new_session=True)
                self.showInfo(f"开始定时录制: {selectedSeconds}s ({targetFrames} 帧)", "info")

            self.currentRecordMode = "timed"
            self.timedTargetFrames = targetFrames
            self.isRecording = True
            self.isRecordPausedPendingSave = False
            self.startTimedProgress(selectedSeconds)
            self.updateRecordingUiState()
        except RecordingError as error:
            self.showInfo(f"定时录制启动失败: {error}", "error")
            self.isRecording = False
            self.updateRecordingUiState()

    def onTimedProgressTick(self):
        if not self.recordingService:
            self.timedProgressTimer.stop()
            return

        if self.recordingService.getState() != RecordingService.STATE_ACTIVE_TIMED:
            self.timedProgressTimer.stop()
            return

        remainingFrames = self.recordingService.getRemainingTimedFrames()
        self.updateCountdownDisplay(remainingFrames)

    def onRecordStop(self):
        self.stopRecording()

    def stopRecording(self):
        if not self.recordingService:
            return

        try:
            frameCount = self.recordingService.getFrameCount()
            self.recordingService.stop()
            self.showInfo(f"停止录制: {frameCount} 帧已保存", "success")
        except RecordingError as error:
            self.showInfo(f"录制停止失败: {error}", "error")
        finally:
            self.isRecording = False
            self.isRecordPausedPendingSave = False
            self.recordingService = None
            self.currentRecordMode = None
            self.timedTargetFrames = 0
            self.timedProgressTimer.stop()
            self.recordProgressRing.setRange(0, 100)
            self.recordProgressRing.setValue(0)
            self.recordProgressRing.setFormat(self.progressRingFormat)
            self.recordedFrameCount = 0
            self.recordFrameCounter = 0
            self.updateFrameCounter()
            self.updateRecordingUiState()

    def handleFrameReceived(self, channels):
        if not self.recordingService:
            return

        state = self.recordingService.getState()
        if state not in (RecordingService.STATE_ACTIVE_CONTINUOUS, RecordingService.STATE_ACTIVE_TIMED):
            return

        try:
            self.recordingService.writeFrame(self.recordFrameCounter, channels)
            self.recordFrameCounter += 1
            self.recordedFrameCount = self.recordingService.getFrameCount()
            self.updateFrameCounter()

            if self.currentRecordMode == "timed":
                remainingFrames = self.recordingService.getRemainingTimedFrames()
                self.updateCountdownDisplay(remainingFrames)

                if self.recordingService.getState() == RecordingService.STATE_PAUSED_PENDING_SAVE:
                    self.isRecording = False
                    self.isRecordPausedPendingSave = True
                    self.timedProgressTimer.stop()
                    self.showInfo("定时录制完成，已进入暂停状态（待停止后保存）", "success")
                    self.updateRecordingUiState()
            else:
                self.isRecording = True
                self.isRecordPausedPendingSave = False
        except RecordingError as error:
            self.showInfo(f"写入帧失败: {error}", "error")
            self.stopRecording()

    def updateFrameCounter(self):
        self.frameCounterLabel.setText(f"已录制: {self.recordedFrameCount} 帧")

    def updateRecordingUiState(self):
        state = self.getRecordingState()
        canStart = self.serialRunning and state in (
            RecordingService.STATE_IDLE,
            RecordingService.STATE_PAUSED_PENDING_SAVE,
        )

        self.recordContinuousStartButton.setEnabled(canStart)
        self.recordTimedStartButton.setEnabled(canStart)
        self.recordDurationSpin.setEnabled(state != RecordingService.STATE_ACTIVE_TIMED)
        self.recordStopButton.setEnabled(state != RecordingService.STATE_IDLE)
        self.applyParamButton.setEnabled(self.serialRunning)
        self.scanFreqSpin.setEnabled(self.serialRunning)
        self.sampleDepthSpin.setEnabled(self.serialRunning)

    def onApplyParamClicked(self):
        """Handle apply parameter button click: send config to device and wait for confirmation."""
        scanFreq = self.scanFreqSpin.value()
        sampleDepth = self.sampleDepthSpin.value()

        # Validate ranges (should be guaranteed by spinbox settings, but be defensive)
        if not (1 <= scanFreq <= 60):
            self.showInfo(f"扫描频率超出范围 [1, 60]: {scanFreq}", "error")
            return
        if not (1 <= sampleDepth <= 1024):
            self.showInfo(f"采样深度超出范围 [1, 1024]: {sampleDepth}", "error")
            return

        # Send configuration command via serial reader
        try:
            # Need to get serial thread from main window or emit a signal
            # For now, we'll look up the serial thread from parent MainWindow
            from gui.MainWindow import MainWindow
            mainWindow = self.window()
            if isinstance(mainWindow, MainWindow) and mainWindow.serialThread:
                mainWindow.serialThread.sendSetConfig(scanFreq, sampleDepth)
                # self.showInfo("已发送参数设置，等待确认...", "info")
                # Disable button and spinboxes during wait
                self.applyParamButton.setEnabled(False)
                self.scanFreqSpin.setEnabled(False)
                self.sampleDepthSpin.setEnabled(False)
                # Start timeout timer
                self.configWaitTimer.start()
            else:
                self.showInfo("串口线程未就绪", "error")
        except Exception as e:
            self.showInfo(f"发送参数设置命令失败: {e}", "error")

    def onConfigReady(self, config_dict: dict):
        """Handle configuration response from device."""
        self.configWaitTimer.stop()
        
        try:
            scanFreqHz = config_dict.get('scan_frequency_hz', self.currentScanFreq)
            sampleDepth = config_dict.get('sample_depth', 0)
            
            # Update current scan frequency
            self.currentScanFreq = scanFreqHz
            self._isApplyingSettings = True
            self.scanFreqSpin.setValue(max(1, min(60, int(scanFreqHz))))
            self.sampleDepthSpin.setValue(max(1, min(1024, int(sampleDepth))))
            self._isApplyingSettings = False
            self._saveSetting("Hardware/ScanFrequencyHz", int(self.scanFreqSpin.value()))
            self._saveSetting("Hardware/SampleDepth", int(self.sampleDepthSpin.value()))
            
            # Re-enable controls
            self.applyParamButton.setEnabled(True)
            self.scanFreqSpin.setEnabled(True)
            self.sampleDepthSpin.setEnabled(True)
            
            # Show success message
            self.showInfo(
                f"设置成功：扫描频率 = {scanFreqHz}Hz，采样深度 = {sampleDepth}",
                "success"
            )
        except Exception as e:
            self.showInfo(f"处理配置响应失败: {e}", "error")
            # Re-enable controls anyway
            self.applyParamButton.setEnabled(True)
            self.scanFreqSpin.setEnabled(True)
            self.sampleDepthSpin.setEnabled(True)

    def onConfigWaitTimeout(self):
        """Handle configuration wait timeout (1 second)."""
        # Re-enable controls
        self.applyParamButton.setEnabled(True)
        self.scanFreqSpin.setEnabled(True)
        self.sampleDepthSpin.setEnabled(True)
        
        # Show timeout error
        self.showInfo("参数设置超时，请检查硬件连接", "error")

    def closeEvent(self, a0):
        self.saveAllSettings()
        super().closeEvent(a0)
