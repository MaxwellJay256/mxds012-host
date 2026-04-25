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
)
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl
from serial.tools import list_ports

from qfluentwidgets import (
    ComboBox,
    FluentIcon,
    HeaderCardWidget,
    LineEdit,
    ProgressRing,
    PushButton,
    SpinBox, 
    ToolButton
)

from gui.ParamCard import ParamCard
from utils.RecordingService import RecordingService
from utils.RecordingUtils import RecordingError, buildOutputPath, checkFileExists
from utils.SettingService import SettingService
from gui.ConnectionCard import ConnectionCard


class FolderPathLineEdit(LineEdit):
    pathClicked = pyqtSignal()

    def mousePressEvent(self, a0):
        self.pathClicked.emit()
        super().mousePressEvent(a0)

class OutputCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("录制输出")
        self.vBoxLayout = QVBoxLayout()

        outputRow = QHBoxLayout()
        outputRow.setSpacing(10)

        self.recordDirButton = ToolButton(FluentIcon.FOLDER)
        outputRow.addWidget(self.recordDirButton)

        self.outputPathEdit = FolderPathLineEdit()
        self.outputPathEdit.setPlaceholderText("录制目录路径")
        self.outputPathEdit.setReadOnly(True)
        outputRow.addWidget(self.outputPathEdit, 2)

        self.filenameEdit = LineEdit()
        self.filenameEdit.setPlaceholderText("输入文件名 (不需要 .txt)")
        outputRow.addWidget(self.filenameEdit, 1)

        self.recordDataSourceCombo = ComboBox(self)
        
        self.recordDataSourceCombo.setFixedWidth(120)
        outputRow.addWidget(self.recordDataSourceCombo)

        outputRow.addStretch()
        self.vBoxLayout.addLayout(outputRow)
        self.viewLayout.addLayout(self.vBoxLayout)

class RecordingWidget(QWidget):
    """Recording and serial control page."""

    startSerialRequested = pyqtSignal(str)
    stopSerialRequested = pyqtSignal()
    infoRequested = pyqtSignal(str, str)
    portsUpdated = pyqtSignal(list, str)
    serialStateChanged = pyqtSignal(bool)
    powerUpdated = pyqtSignal(dict)
    configApplied = pyqtSignal(dict)
    RECORD_SOURCE_KEY = "Recording/DataSource"
    RECORD_DURATION_KEY = "Recording/DurationSeconds"
    RECORD_SOURCE_RAW = "raw"
    RECORD_SOURCE_PROCESSED = "processed"
    RECORD_SOURCE_TEXT_RAW = "原始数据"
    RECORD_SOURCE_TEXT_PROCESSED = "预处理数据"

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
        self._recordDataSourceWarningActive = False
        self._recordDataSource = self.RECORD_SOURCE_PROCESSED

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
        rootLayout.setSpacing(6)

        self.connectionGroup = ConnectionCard()
        rootLayout.addWidget(self.connectionGroup)

        self.paramCard = ParamCard()
        rootLayout.addWidget(self.paramCard)

        self.outputGroup = OutputCard()
        self.outputGroup.recordDataSourceCombo.addItem(self.RECORD_SOURCE_TEXT_RAW)
        self.outputGroup.recordDataSourceCombo.addItem(self.RECORD_SOURCE_TEXT_PROCESSED)
        rootLayout.addWidget(self.outputGroup)

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

        self.connectionGroup.refreshButton.clicked.connect(self.refreshPorts)
        self.connectionGroup.startButton.clicked.connect(self.onStartSerialClicked)
        self.connectionGroup.stopButton.clicked.connect(self.onStopSerialClicked)
        self.outputGroup.recordDirButton.clicked.connect(self.selectOutputDirectory)
        self.outputGroup.outputPathEdit.pathClicked.connect(self.onOutputPathEditClicked)
        self.paramCard.applyParamButton.clicked.connect(self.onApplyParamClicked)
        self.recordContinuousStartButton.clicked.connect(self.onRecordStartContinuous)
        self.recordTimedStartButton.clicked.connect(self.onRecordStartTimed)
        self.recordStopButton.clicked.connect(self.onRecordStop)
        self.connectionGroup.portCombo.currentIndexChanged.connect(self.onPortSelectionChanged)
        self.paramCard.scanFreqSpin.valueChanged.connect(self.onScanFreqChanged)
        self.paramCard.sampleDepthSpin.valueChanged.connect(self.onSampleDepthChanged)
        self.recordDurationSpin.valueChanged.connect(self.onRecordDurationChanged)
        self.outputGroup.filenameEdit.editingFinished.connect(self.onFilenameEditingFinished)
        self.outputGroup.recordDataSourceCombo.currentIndexChanged.connect(self.onRecordDataSourceChanged)

        self.loadSettings()
        self.refreshPorts()
        self.updateRecordingUiState()
        self._updateRecordDataSourceWarning()

    def loadSettings(self):
        self._isApplyingSettings = True
        try:
            scanFreq = self.settings.value("Hardware/ScanFrequencyHz", 20, type=int)
            sampleDepth = self.settings.value("Hardware/SampleDepth", 128, type=int)
            outputDir = self.settings.value("Recording/OutputDirectory", "", type=str)
            filename = self.settings.value("Recording/FileName", "", type=str)
            recordDuration = self.settings.value(self.RECORD_DURATION_KEY, 5, type=int)
            recordDataSource = self.settings.value(self.RECORD_SOURCE_KEY, self.RECORD_SOURCE_PROCESSED, type=str)
            self._pendingLastPort = self.settings.value("Serial/LastPort", "", type=str)

            self.paramCard.scanFreqSpin.setValue(max(1, min(60, scanFreq)))
            self.paramCard.sampleDepthSpin.setValue(max(1, min(1024, sampleDepth)))
            self.recordDurationSpin.setValue(max(1, min(60, recordDuration)))
            self.outputGroup.filenameEdit.setText(filename)
            self._setRecordDataSource(recordDataSource)

            if outputDir and Path(outputDir).exists():
                self.recordOutputDir = outputDir
                self.outputGroup.outputPathEdit.setText(outputDir)
            else:
                self.recordOutputDir = ""
                self.outputGroup.outputPathEdit.setText("")
                if outputDir:
                    self.showInfo("上次录制目录不存在，已清空，请重新选择", "warning")
        finally:
            self._isApplyingSettings = False

    def _saveSetting(self, key: str, value):
        self.settings.setValue(key, value)
        self.settings.sync()

    def saveAllSettings(self):
        port = self.connectionGroup.portCombo.currentData()
        self._saveSetting("Serial/LastPort", port if port else "")
        self._saveSetting("Hardware/ScanFrequencyHz", int(self.paramCard.scanFreqSpin.value()))
        self._saveSetting("Hardware/SampleDepth", int(self.paramCard.sampleDepthSpin.value()))
        self._saveSetting("Recording/OutputDirectory", self.recordOutputDir.strip())
        self._saveSetting("Recording/FileName", self.outputGroup.filenameEdit.text().strip())
        self._saveSetting(self.RECORD_DURATION_KEY, int(self.recordDurationSpin.value()))
        self._saveSetting(self.RECORD_SOURCE_KEY, self.getRecordDataSource())

    def _recordSourceFromText(self, text: str) -> str:
        normalized = (text or "").strip()
        if normalized == self.RECORD_SOURCE_TEXT_RAW:
            return self.RECORD_SOURCE_RAW
        return self.RECORD_SOURCE_PROCESSED

    def _recordSourceText(self, source: str) -> str:
        if source == self.RECORD_SOURCE_RAW:
            return self.RECORD_SOURCE_TEXT_RAW
        return self.RECORD_SOURCE_TEXT_PROCESSED

    def _setRecordDataSource(self, source: str):
        if source not in (self.RECORD_SOURCE_RAW, self.RECORD_SOURCE_PROCESSED):
            source = self.RECORD_SOURCE_PROCESSED

        self._recordDataSource = source
        text = self._recordSourceText(source)
        if self.outputGroup.recordDataSourceCombo.currentText() != text:
            self.outputGroup.recordDataSourceCombo.setCurrentText(text)

    def onPortSelectionChanged(self, _index: int):
        if self._isApplyingSettings:
            return
        port = self.connectionGroup.portCombo.currentData()
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
        self._saveSetting("Recording/FileName", self.outputGroup.filenameEdit.text().strip())

    def onRecordDurationChanged(self, value: int):
        if self._isApplyingSettings:
            return
        self._saveSetting(self.RECORD_DURATION_KEY, int(value))

    def onRecordDataSourceChanged(self, _index: int):
        self._recordDataSource = self._recordSourceFromText(self.outputGroup.recordDataSourceCombo.currentText())

        if self._isApplyingSettings:
            return
        self._saveSetting(self.RECORD_SOURCE_KEY, self.getRecordDataSource())
        self._updateRecordDataSourceWarning()

    def getRecordDataSource(self) -> str:
        return self._recordDataSource

    def _isRealtimePreprocessEnabled(self) -> bool:
        try:
            from gui.MainWindow import MainWindow
            mainWindow = self.window()
            if isinstance(mainWindow, MainWindow):
                return bool(mainWindow.plotPanel.isRealtimePreprocessEnabled())
        except Exception:
            pass
        return False

    def _shouldWarnProcessedSourceWithoutPreprocess(self) -> bool:
        return (
            self.getRecordDataSource() == self.RECORD_SOURCE_PROCESSED
            and not self._isRealtimePreprocessEnabled()
        )

    def _updateRecordDataSourceWarning(self):
        shouldWarn = self._shouldWarnProcessedSourceWithoutPreprocess()

        if shouldWarn and not self._recordDataSourceWarningActive:
            self._recordDataSourceWarningActive = True
            self.showInfo(
                "当前选择“预处理数据”，但实时预处理未开启。若继续录制将导致数据源不匹配。",
                "warning",
            )
            return

        if not shouldWarn:
            self._recordDataSourceWarningActive = False

    def onRealtimePreprocessSwitchChanged(self, _enabled: bool):
        self._updateRecordDataSourceWarning()

    def showInfo(self, message: str, level: str = "info"):
        self.infoRequested.emit(message, level)

    def refreshPorts(self):
        previousDevice = self.connectionGroup.portCombo.currentData()
        self.connectionGroup.portCombo.clear()

        ports = list_ports.comports()
        emittedOptions = []
        for portInfo in ports:
            displayText = f"{portInfo.device} - {portInfo.description}"
            self.connectionGroup.portCombo.addItem(displayText, portInfo.device)
            emittedOptions.append((displayText, portInfo.device))

        if not ports:
            self.showInfo("未检测到串口设备", "warning")
            self.portsUpdated.emit([], "")
            return

        preferredDevice = previousDevice or self._pendingLastPort
        if preferredDevice:
            index = self.connectionGroup.portCombo.findData(preferredDevice)
            if index >= 0:
                self._isApplyingSettings = True
                self.connectionGroup.portCombo.setCurrentIndex(index)
                self._isApplyingSettings = False

        currentPort = self.connectionGroup.portCombo.currentData()
        if currentPort:
            self._saveSetting("Serial/LastPort", currentPort)
        self.portsUpdated.emit(emittedOptions, str(currentPort or ""))

    def onStartSerialClicked(self):
        if self.serialRunning:
            self.showInfo("串口已在运行", "warning")
            return

        if self.connectionGroup.portCombo.count() == 0:
            self.showInfo("没有可用串口", "warning")
            return

        port = self.connectionGroup.portCombo.currentData()
        if not port:
            display = self.connectionGroup.portCombo.currentText().strip()
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
        self.connectionGroup.startButton.setEnabled(False)
        self.connectionGroup.stopButton.setEnabled(True)
        self.currentScanFreq = self.paramCard.scanFreqSpin.value()
        self.clearPowerLabels()
        self.updateRecordingUiState()
        self.serialStateChanged.emit(True)

    def onSerialStopped(self):
        self.serialRunning = False
        self.connectionGroup.startButton.setEnabled(True)
        self.connectionGroup.stopButton.setEnabled(False)
        self.clearPowerLabels()
        self.updateRecordingUiState()
        self.serialStateChanged.emit(False)

    def clearPowerLabels(self):
        self.connectionGroup.devVoltageLabel.setText("总线电压: V")
        self.connectionGroup.devCurrentLabel.setText("电流: mA")
        self.connectionGroup.devPowerLabel.setText("功率: mW")

    def onPowerReady(self, powerData: dict):
        busMilliVolt = int(powerData.get("bus_mv", 0))
        currentMilliAmp = int(powerData.get("current_ma", 0))
        powerMilliWatt = int(powerData.get("power_mw", 0))

        busVolt = busMilliVolt / 1000.0
        self.connectionGroup.devVoltageLabel.setText(f"总线电压: {busVolt:.2f}V")
        self.connectionGroup.devCurrentLabel.setText(f"电流: {currentMilliAmp}mA")
        self.connectionGroup.devPowerLabel.setText(f"功率: {powerMilliWatt}mW")
        self.powerUpdated.emit(dict(powerData))

    def selectOutputDirectory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择录制文件保存目录",
            "",
        )
        if directory:
            self.recordOutputDir = directory
            self.outputGroup.outputPathEdit.setText(directory)
            self._saveSetting("Recording/OutputDirectory", directory)
            # self.showInfo(f"录制目录: {directory}", "info")
        else:
            self.recordOutputDir = ""
            self.outputGroup.outputPathEdit.setText("")
            self._saveSetting("Recording/OutputDirectory", "")

    def onOutputPathEditClicked(self):
        path = self.outputGroup.outputPathEdit.text().strip()
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

        filename = self.outputGroup.filenameEdit.text().strip()
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

        if self._shouldWarnProcessedSourceWithoutPreprocess():
            self.showInfo("已选择预处理数据，但实时预处理未开启。请先开启“实时预处理”后再录制。", "warning")
            self._updateRecordDataSourceWarning()
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

        if self._shouldWarnProcessedSourceWithoutPreprocess():
            self.showInfo("已选择预处理数据，但实时预处理未开启。请先开启“实时预处理”后再录制。", "warning")
            self._updateRecordDataSourceWarning()
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

    def _handleRecordingChannels(self, channels):
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

    def handleFrameReceived(self, channels):
        if self.getRecordDataSource() != self.RECORD_SOURCE_PROCESSED:
            return
        self._handleRecordingChannels(channels)

    def handleRawFrameReceived(self, channels):
        if self.getRecordDataSource() != self.RECORD_SOURCE_RAW:
            return
        self._handleRecordingChannels(channels)

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
        self.paramCard.applyParamButton.setEnabled(self.serialRunning)
        self.paramCard.scanFreqSpin.setEnabled(self.serialRunning)
        self.paramCard.sampleDepthSpin.setEnabled(self.serialRunning)
        self.outputGroup.recordDataSourceCombo.setEnabled(state == RecordingService.STATE_IDLE)
        self._updateRecordDataSourceWarning()

    def onApplyParamClicked(self):
        """Handle apply parameter button click: send config to device and wait for confirmation."""
        scanFreq = self.paramCard.scanFreqSpin.value()
        sampleDepth = self.paramCard.sampleDepthSpin.value()

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
                self.paramCard.applyParamButton.setEnabled(False)
                self.paramCard.scanFreqSpin.setEnabled(False)
                self.paramCard.sampleDepthSpin.setEnabled(False)
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
            self.paramCard.scanFreqSpin.setValue(max(1, min(60, int(scanFreqHz))))
            self.paramCard.sampleDepthSpin.setValue(max(1, min(1024, int(sampleDepth))))
            self._isApplyingSettings = False
            self._saveSetting("Hardware/ScanFrequencyHz", int(self.paramCard.scanFreqSpin.value()))
            self._saveSetting("Hardware/SampleDepth", int(self.paramCard.sampleDepthSpin.value()))
            
            # Re-enable controls
            self.paramCard.applyParamButton.setEnabled(True)
            self.paramCard.scanFreqSpin.setEnabled(True)
            self.paramCard.sampleDepthSpin.setEnabled(True)
            
            # Show success message
            self.showInfo(
                f"设置成功：扫描频率 = {scanFreqHz}Hz，采样深度 = {sampleDepth}",
                "success"
            )
            self.configApplied.emit(dict(config_dict))
        except Exception as e:
            self.showInfo(f"处理配置响应失败: {e}", "error")
            # Re-enable controls anyway
            self.paramCard.applyParamButton.setEnabled(True)
            self.paramCard.scanFreqSpin.setEnabled(True)
            self.paramCard.sampleDepthSpin.setEnabled(True)

    def onConfigWaitTimeout(self):
        """Handle configuration wait timeout (1 second)."""
        # Re-enable controls
        self.paramCard.applyParamButton.setEnabled(True)
        self.paramCard.scanFreqSpin.setEnabled(True)
        self.paramCard.sampleDepthSpin.setEnabled(True)
        
        # Show timeout error
        self.showInfo("参数设置超时，请检查硬件连接", "error")

    def closeEvent(self, a0):
        self.saveAllSettings()
        super().closeEvent(a0)
