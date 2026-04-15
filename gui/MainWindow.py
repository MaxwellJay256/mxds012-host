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

from SerialReader import SerialReader
from gui import FunctionArea, PlotPanel
from SettingService import SettingService


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

        self._centralLayout = QHBoxLayout()
        self._centralLayout.setContentsMargins(24, 24, 24, 24)
        self._centralLayout.setSpacing(18)
        self.setLayout(self._centralLayout)

        self.plotPanel = PlotPanel(self)
        self.functionArea = FunctionArea(self)
        self.recordingWidget = self.functionArea.recordingWidget

        self._centralLayout.addWidget(self.plotPanel, 1)
        self._centralLayout.addWidget(self.functionArea, 1)

        self.recordingWidget.startSerialRequested.connect(self.startSerial)
        self.recordingWidget.stopSerialRequested.connect(self.stopSerial)
        self.recordingWidget.infoRequested.connect(self.onRecordingInfoRequested)
        self.plotPanel.preprocessToggled.connect(self.onPlotPanelPreprocessToggled)
        self.plotPanel.preprocessToggled.connect(self.recordingWidget.onRealtimePreprocessSwitchChanged)

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
        self.serialThread.frameReady.connect(self.plotPanel.updatePlots)
        self.serialThread.rawFrameReady.connect(self.plotPanel.updateRawOverlay)
        self.serialThread.frameReady.connect(self.recordingWidget.handleFrameReceived)
        self.serialThread.rawFrameReady.connect(self.recordingWidget.handleRawFrameReceived)
        self.serialThread.powerReady.connect(self.recordingWidget.onPowerReady)
        self.serialThread.statusChanged.connect(lambda msg: self.showInfo(msg, InfoBarPosition.BOTTOM, "info"))
        self.serialThread.errorOccurred.connect(lambda msg: self.showInfo(msg, InfoBarPosition.BOTTOM, "error"))
        self.serialThread.configReady.connect(self.plotPanel.onConfigReady)
        self.serialThread.configReady.connect(self.recordingWidget.onConfigReady)
        self.serialThread.stopped.connect(self.onSerialStopped)

        self.serialThread.start()
        self.recordingWidget.onSerialStarted()

    def onPlotPanelPreprocessToggled(self, enabled: bool):
        if self.serialThread and self.serialThread.isRunning():
            self.serialThread.setRealtimePreprocessEnabled(enabled)

    def stopSerial(self):
        self.recordingWidget.stopRecording()

        if self.serialThread and self.serialThread.isRunning():
            self.serialThread.stop()
            self.serialThread.wait(2000)
        else:
            self.onSerialStopped()

    def onSerialStopped(self):
        self.serialThread = None
        self.recordingWidget.onSerialStopped()

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

        if self.serialThread and self.serialThread.isRunning():
            self.serialThread.stop()
            self.serialThread.wait(2000)
        
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        
        super().closeEvent(a0)
