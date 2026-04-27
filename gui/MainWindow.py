"""
Main application window and UI orchestration.
"""
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
from qfluentwidgets import (
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    Theme,
    SystemThemeListener,
    setTheme,
    setThemeColor,
    isDarkTheme
)

from utils.SerialReader import SerialReader
from utils.RTAnalyzer import RTAnalyzer
from gui import FunctionArea, PlotPanel
from utils.SettingService import SettingService


class MainWindow(QWidget):
    """Main application window."""

    WINDOW_WIDTH_KEY = "Window/Width"
    WINDOW_HEIGHT_KEY = "Window/Height"
    WINDOW_MAXIMIZED_KEY = "Window/Maximized"
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MXDS012 Host")
        self.resize(1920, 1080)
        self.setWindowIcon(FluentIcon.CONNECT.icon())

        setTheme(Theme.LIGHT)
        setThemeColor("#e4360f")
        self.themeListener = SystemThemeListener(self)

        pg.setConfigOptions(antialias=True, background=None, foreground='#f0f0f0')

        self.serialThread: Optional[SerialReader] = None
        self.trainingWorker = None
        self.rtAnalyzer = RTAnalyzer.getInstance()
        self.rtAnalyzer.setEnabled(False)
        self._activeFunctionPage = ""

        self._centralLayout = QHBoxLayout()
        self._centralLayout.setContentsMargins(24, 24, 24, 24)
        self._centralLayout.setSpacing(18)
        self.setLayout(self._centralLayout)

        self.plotPanel = PlotPanel(self)
        self.functionArea = FunctionArea(self)
        self.recordingWidget = self.functionArea.recordingWidget
        self.realtimeAnalyzeWidget = self.functionArea.realtimeAnalyzeWidget

        self._centralLayout.addWidget(self.plotPanel, 3)
        self._centralLayout.addWidget(self.functionArea, 2)

        self.recordingWidget.startSerialRequested.connect(self.startSerial)
        self.recordingWidget.stopSerialRequested.connect(self.stopSerial)
        self.recordingWidget.infoRequested.connect(self.onRecordingInfoRequested)
        self.realtimeAnalyzeWidget.startSerialRequested.connect(self.startSerial)
        self.realtimeAnalyzeWidget.stopSerialRequested.connect(self.stopSerial)
        self.realtimeAnalyzeWidget.refreshPortsRequested.connect(self.recordingWidget.refreshPorts)
        self.realtimeAnalyzeWidget.applyConfigRequested.connect(self.onRealtimeAnalyzeApplyConfig)
        self.realtimeAnalyzeWidget.infoRequested.connect(self.onRecordingInfoRequested)
        self.playbackWidget = self.functionArea.playbackWidget
        self.playbackWidget.frameReady.connect(self.plotPanel.updatePlaybackFrames)
        self.playbackWidget.rawFrameReady.connect(self.plotPanel.updateRawOverlay)
        self.playbackWidget.configReady.connect(self.plotPanel.onConfigReady)
        self.playbackWidget.infoRequested.connect(self.onRecordingInfoRequested)
        self.plotPanel.preprocessToggled.connect(self.onPlotPanelPreprocessToggled)
        self.plotPanel.preprocessToggled.connect(self.recordingWidget.onRealtimePreprocessSwitchChanged)
        self.plotPanel.preprocessToggled.connect(self.playbackWidget.onRealtimePreprocessSwitchChanged)
        self.playbackWidget.onRealtimePreprocessSwitchChanged(self.plotPanel.isRealtimePreprocessEnabled())
        self.functionArea.stackedWidget.currentChanged.connect(self.onFunctionPageChanged)

        self.recordingWidget.portsUpdated.connect(self.realtimeAnalyzeWidget.updatePortOptions)
        self.recordingWidget.serialStateChanged.connect(self.realtimeAnalyzeWidget.setSerialRunning)
        self.recordingWidget.powerUpdated.connect(self.realtimeAnalyzeWidget.updatePowerData)
        self.recordingWidget.configApplied.connect(self.realtimeAnalyzeWidget.onConfigReady)
        self.trainingWidget = self.functionArea.trainingWidget
        self.trainingWidget.refreshRequested.connect(self.onTrainingDatasetRefreshRequested)
        self.trainingWidget.startTrainingRequested.connect(self.onTrainingStartRequested)
        self.trainingWidget.infoRequested.connect(self.onRecordingInfoRequested)

        self.realtimeAnalyzeWidget.setSerialRunning(self.recordingWidget.serialRunning)
        self.realtimeAnalyzeWidget.onConfigReady(
            {
                "scan_frequency_hz": int(self.recordingWidget.paramCard.scanFreqSpin.value()),
                "sample_depth": int(self.recordingWidget.paramCard.sampleDepthSpin.value()),
            }
        )
        self.recordingWidget.refreshPorts()
        self.onFunctionPageChanged(self.functionArea.stackedWidget.currentIndex())

        self._loadWindowSettings()
        self.themeListener.start()

    def _loadWindowSettings(self):
        width = SettingService.value(self.WINDOW_WIDTH_KEY, 1920, int)
        height = SettingService.value(self.WINDOW_HEIGHT_KEY, 1080, int)
        isMaximized = SettingService.value(self.WINDOW_MAXIMIZED_KEY, False, bool)

        self.resize(max(640, int(width)), max(480, int(height)))
        if isMaximized:
            self.showMaximized()

    def _saveWindowSettings(self):
        isMaximized = self.isMaximized()
        SettingService.set_value(self.WINDOW_MAXIMIZED_KEY, isMaximized)

        if not isMaximized:
            SettingService.set_value(self.WINDOW_WIDTH_KEY, int(self.width()))
            SettingService.set_value(self.WINDOW_HEIGHT_KEY, int(self.height()))

        SettingService.sync()

    def startSerial(self, port: str):
        if self.serialThread and self.serialThread.isRunning():
            self.showInfo("串口已在运行", InfoBarPosition.BOTTOM, "warning")
            return

        self.serialThread = SerialReader(port)
        self.serialThread.setRealtimePreprocessEnabled(self.plotPanel.isRealtimePreprocessEnabled())
        self.serialThread.plotFrameReady.connect(self.plotPanel.updatePlots)
        self.serialThread.rawFrameReady.connect(self.plotPanel.updateRawOverlay)
        self.serialThread.frameReady.connect(self.recordingWidget.handleFrameReceived)
        self.serialThread.modelFrameReady.connect(self.rtAnalyzer.analyzeFrame)
        self.serialThread.rawFrameReady.connect(self.recordingWidget.handleRawFrameReceived)
        self.serialThread.powerReady.connect(self.recordingWidget.onPowerReady)
        self.serialThread.statusChanged.connect(lambda msg: self.showInfo(msg, InfoBarPosition.BOTTOM, "info"))
        self.serialThread.errorOccurred.connect(lambda msg: self.showInfo(msg, InfoBarPosition.BOTTOM, "error"))
        self.serialThread.configReady.connect(self.plotPanel.onConfigReady)
        self.serialThread.configReady.connect(self.recordingWidget.onConfigReady)
        self.serialThread.stopped.connect(self.onSerialStopped)

        self.serialThread.start()
        self.recordingWidget.onSerialStarted()
        self._syncRealtimeAnalyzerState()

    def onPlotPanelPreprocessToggled(self, enabled: bool):
        if self.serialThread and self.serialThread.isRunning():
            self.serialThread.setRealtimePreprocessEnabled(enabled)
        self._syncRealtimeAnalyzerState()

    def _syncRealtimeAnalyzerState(self):
        enabled = (
            self._activeFunctionPage == "realtimeAnalyze"
            and self.serialThread is not None
            and self.serialThread.isRunning()
            and self.plotPanel.isRealtimePreprocessEnabled()
        )
        self.rtAnalyzer.setEnabled(bool(enabled))

    def onFunctionPageChanged(self, index: int):
        widget = self.functionArea.stackedWidget.widget(index)
        page_name = widget.objectName() if widget is not None else ""
        previous_page_name = self._activeFunctionPage
        self._activeFunctionPage = page_name

        if previous_page_name == "realtimeAnalyze" and page_name != "realtimeAnalyze":
            self.realtimeAnalyzeWidget.stopOnlineTest("已离开在线识别页面，在线测试已停止", level="warning")

        if page_name in {"recording", "playback", "realtimeAnalyze"}:
            if not self.plotPanel.isRealtimePreprocessEnabled():
                self.plotPanel.setRealtimePreprocessEnabled(True, emit_signal=True)
        else:
            if self.plotPanel.isRealtimePreprocessEnabled():
                self.plotPanel.setRealtimePreprocessEnabled(False, emit_signal=True)

        if page_name in {"recording", "training"}:
            self.rtAnalyzer.setEnabled(False)

        self._syncRealtimeAnalyzerState()

    def onTrainingDatasetRefreshRequested(self, dataset_dir: str):
        try:
            from ml.training_pipeline import analyze_dataset
            analysis = analyze_dataset(dataset_dir, strict=False)
        except Exception as error:
            self.trainingWidget.setDatasetSummary("数据集分析失败")
            self.trainingWidget.setDatasetIssues([str(error)])
            self.showInfo(f"数据集分析失败: {error}", InfoBarPosition.BOTTOM, "error")
            return

        summary = (
            f"有效文件: {analysis.valid_files} | 识别类别: {analysis.num_classes} | "
            f"采样深度: {analysis.sampling_depth} | 忽略文件: {analysis.ignored_files}"
        )
        self.trainingWidget.setDatasetSummary(summary)
        self.trainingWidget.setDatasetIssues(analysis.issues)
        self.showInfo("数据集分析完成", InfoBarPosition.BOTTOM, "success")

    def onTrainingStartRequested(self, payload: dict):
        try:
            from gui.TrainingWorker import TrainingWorker
        except Exception as error:
            self.showInfo(f"训练模块不可用: {error}", InfoBarPosition.BOTTOM, "error")
            return

        if self.trainingWorker is not None and self.trainingWorker.isRunning():
            self.showInfo("训练任务正在运行", InfoBarPosition.BOTTOM, "warning")
            return

        self.trainingWorker = TrainingWorker(payload, self)
        self.trainingWorker.logMessage.connect(self.trainingWidget.appendLog)
        self.trainingWorker.progressChanged.connect(lambda value: self.trainingWidget.appendLog(f"训练进度: {value}%"))
        self.trainingWorker.failed.connect(self.onTrainingWorkerFailed)
        self.trainingWorker.finishedSuccessfully.connect(self.onTrainingWorkerFinished)
        self.trainingWorker.finished.connect(lambda: self.trainingWidget.setTrainingRunning(False))

        self.trainingWidget.setTrainingRunning(True)
        self.trainingWidget.appendLog("开始训练任务...")
        self.trainingWorker.start()

    def onTrainingWorkerFinished(self, result: dict):
        self.trainingWidget.appendLog(f"训练完成: {result.get('onnx_path', '')}")
        self.showInfo("训练完成，模型已导出", InfoBarPosition.BOTTOM, "success")

    def onTrainingWorkerFailed(self, message: str):
        self.trainingWidget.appendLog(f"训练失败: {message}")
        self.showInfo(f"训练失败: {message}", InfoBarPosition.BOTTOM, "error")

    def stopSerial(self):
        self.recordingWidget.stopRecording()
        self.realtimeAnalyzeWidget.stopOnlineTest("串口停止，在线测试已停止", level="warning")

        if self.serialThread and self.serialThread.isRunning():
            self.serialThread.stop()
            self.serialThread.wait(2000)
        else:
            self.onSerialStopped()

    def onRealtimeAnalyzeApplyConfig(self, scan_freq_hz: int, sample_depth: int):
        self.recordingWidget.paramCard.scanFreqSpin.setValue(max(1, min(60, int(scan_freq_hz))))
        self.recordingWidget.paramCard.sampleDepthSpin.setValue(max(1, min(1024, int(sample_depth))))
        self.recordingWidget.onApplyParamClicked()

    def onSerialStopped(self):
        self.rtAnalyzer.setEnabled(False)
        self.serialThread = None
        self.recordingWidget.onSerialStopped()
        self._syncRealtimeAnalyzerState()

    def onRecordingInfoRequested(self, message: str, level: str):
        self.showInfo(message, InfoBarPosition.BOTTOM, level)

    def _onThemeChangedFinished(self):
        super()._onThemeChangedFinished()
        if self.isMicaEffectEnabled():
            QTimer.singleShot(100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme()))

    def showInfo(self, message: str, position: InfoBarPosition, level: str = "success"):
        """Show info bar message."""
        builders = {
            "success": InfoBar.success,
            "info": InfoBar.info,
            "warning": InfoBar.warning,
            "error": InfoBar.error
        }
        builder = builders.get(level, InfoBar.info)
        builder(
            title="",
            content=message,
            parent=self,
            position=position,
            duration=3000,
            isClosable=True
        )

    def closeEvent(self, a0):
        """Handle window close event."""
        self._saveWindowSettings()
        self.recordingWidget.stopRecording()
        self.realtimeAnalyzeWidget.stopOnlineTest("窗口关闭，在线测试已停止", level="warning")

        if self.serialThread and self.serialThread.isRunning():
            self.serialThread.stop()
            self.serialThread.wait(2000)

        self.rtAnalyzer.setEnabled(False)
        
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        
        super().closeEvent(a0)
