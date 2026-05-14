"""Realtime analysis page UI (frontend only)."""

from __future__ import annotations

import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CompactSpinBox,
    FluentIcon,
    HeaderCardWidget,
    LineEdit,
    ProgressBar,
    ProgressRing,
    PushButton,
    ToolButton,
)

from gui.AnalyzeParamCard import AnalyzeParamCard
from gui.ConnectionCard import ConnectionCard
from gui.GesturePanelCard import GesturePanelCard
from utils.GestureConstants import DEFAULT_GESTURE_ORDER
from utils.ModelMetadataService import load_model_metadata
from utils.PreprocessParamService import PreprocessParamService
from utils.RecordingUtils import ensureOutputDir
from utils.RTAnalyzer import RTAnalyzer
from utils.SettingService import SettingService


class OnlineTestRoundResult(TypedDict):
    index: float
    gesture: str
    correct: float
    total: float
    accuracy: float


class RealtimeAnalyzeWidget(QWidget):
    """Online analysis page for model selection and gesture class display."""

    MODEL_PATH_KEY = "Analysis/ModelPath"
    EMA_ALPHA_KEY = "Analysis/EmaAlpha"
    VOTE_WINDOW_KEY = "Analysis/VoteWindowSize"
    ONLINE_OUTPUT_DIR_KEY = "Analysis/OnlineTestOutputDir"
    ONLINE_ROUNDS_KEY = "Analysis/OnlineTestRounds"
    ONLINE_ACTIVE_SECONDS_KEY = "Analysis/OnlineTestActiveSeconds"
    ONLINE_REST_SECONDS_KEY = "Analysis/OnlineTestRestSeconds"
    STATUS_COLOR_STOPPED_OR_REST = "#C62828"
    STATUS_COLOR_ACTIVE = "#1B5E20"

    startSerialRequested = pyqtSignal(str)
    stopSerialRequested = pyqtSignal()
    refreshPortsRequested = pyqtSignal()
    applyConfigRequested = pyqtSignal(int, int)
    infoRequested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._serial_running = False
        self._sampling_depth = 0
        self._model_path = ""
        self._model_classes: List[str] = []
        self._analyzer = RTAnalyzer.getInstance()
        self._onlineTestTimer = QTimer(self)
        self._onlineTestTimer.setInterval(100)
        self._onlineTestTimer.timeout.connect(self._onOnlineTestTimerTick)

        self._onlineTestActive = False
        self._onlineTestCollecting = False
        self._onlineTestPlan: List[str] = []
        self._onlineTestGestureIndex = -1
        self._onlineTestTotalTasks = 0
        self._onlineTestCompletedTasks = 0
        self._onlineTestRemainingTenths = 0
        self._onlineTestCurrentTarget = ""
        self._onlineTestCurrentCorrectFrames = 0
        self._onlineTestCurrentTotalFrames = 0
        self._onlineTestResults: List[OnlineTestRoundResult] = []
        self._onlineTestGestureStats: Dict[str, Dict[str, int]] = {}
        self._onlineTestCurrentGesturePixmap: Optional[QPixmap] = None
        self._onlineTestNextGesturePixmap: Optional[QPixmap] = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(6)

        self.connectionCard = ConnectionCard(self)
        root_layout.addWidget(self.connectionCard)

        self.paramCard = AnalyzeParamCard(self)
        root_layout.addWidget(self.paramCard)

        self.onlineTestConfigCard = HeaderCardWidget(self)
        self.onlineTestConfigCard.setTitle("在线测试参数")
        self.onlineTestConfigLayout = QVBoxLayout()
        self.onlineTestConfigLayout.setSpacing(8)

        output_row = QHBoxLayout()
        output_row.setSpacing(10)
        self.onlineTestOutputDirButton = ToolButton(FluentIcon.FOLDER)
        output_row.addWidget(self.onlineTestOutputDirButton)

        self.onlineTestOutputDirEdit = LineEdit(self)
        self.onlineTestOutputDirEdit.setPlaceholderText("选择在线测试报告输出目录")
        output_row.addWidget(self.onlineTestOutputDirEdit, 1)
        self.onlineTestConfigLayout.addLayout(output_row)

        config_row = QHBoxLayout()
        config_row.setSpacing(10)
        config_row.addWidget(BodyLabel("轮次"))
        self.onlineTestRoundsSpin = CompactSpinBox(self)
        self.onlineTestRoundsSpin.setRange(1, 100)
        self.onlineTestRoundsSpin.setValue(3)
        self.onlineTestRoundsSpin.setSuffix(" 轮")
        config_row.addWidget(self.onlineTestRoundsSpin)

        config_row.addWidget(BodyLabel("持续"))
        self.onlineTestActiveSecondsSpin = CompactSpinBox(self)
        self.onlineTestActiveSecondsSpin.setRange(1, 60)
        self.onlineTestActiveSecondsSpin.setValue(5)
        self.onlineTestActiveSecondsSpin.setSuffix(" s")
        config_row.addWidget(self.onlineTestActiveSecondsSpin)

        config_row.addWidget(BodyLabel("休息"))
        self.onlineTestRestSecondsSpin = CompactSpinBox(self)
        self.onlineTestRestSecondsSpin.setRange(0, 30)
        self.onlineTestRestSecondsSpin.setValue(2)
        self.onlineTestRestSecondsSpin.setSuffix(" s")
        config_row.addWidget(self.onlineTestRestSecondsSpin)
        config_row.addStretch()
        self.onlineTestConfigLayout.addLayout(config_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.onlineTestStartButton = PushButton("开始在线测试")
        self.onlineTestStopButton = PushButton("停止在线测试")
        self.onlineTestStopButton.setEnabled(False)
        action_row.addWidget(self.onlineTestStartButton)
        action_row.addWidget(self.onlineTestStopButton)
        action_row.addStretch()
        self.onlineTestConfigLayout.addLayout(action_row)

        self.onlineTestConfigCard.viewLayout.addLayout(self.onlineTestConfigLayout)
        root_layout.addWidget(self.onlineTestConfigCard)

        self.onlineTestStatusCard = HeaderCardWidget(self)
        self.onlineTestStatusCard.setTitle("在线测试状态")
        self.onlineTestStatusLayout = QVBoxLayout()
        self.onlineTestStatusLayout.setSpacing(8)

        self.onlineTestStateLabel = BodyLabel("状态: 待开始")
        self.onlineTestCurrentGestureLabel = BodyLabel("当前目标手势: -")
        self.onlineTestNextGestureLabel = BodyLabel("下一个手势: -")
        self.onlineTestProgressLabel = BodyLabel("进度: 0/0")
        self.onlineTestStatusLayout.addWidget(self.onlineTestStateLabel)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        progress_row.addWidget(self.onlineTestProgressLabel)
        self.onlineTestOverallProgressBar = ProgressBar(self)
        self.onlineTestOverallProgressBar.setRange(0, 1)
        self.onlineTestOverallProgressBar.setValue(0)
        progress_row.addWidget(self.onlineTestOverallProgressBar, 1)
        self.onlineTestStatusLayout.addLayout(progress_row)

        self.onlineTestCountdownRing = ProgressRing(self)
        self.onlineTestCountdownRing.setRange(0, 10)
        self.onlineTestCountdownRing.setValue(0)
        self.onlineTestCountdownRing.setTextVisible(True)
        self.onlineTestCountdownRing.setFixedSize(72, 72)
        self.onlineTestCountdownRing.setStrokeWidth(5)
        self.onlineTestCountdownRing.setFormat("0.0 s")

        self.onlineTestCurrentGestureImage = QLabel("无图片")
        self.onlineTestCurrentGestureImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.onlineTestCurrentGestureImage.setMinimumHeight(56)
        self.onlineTestCurrentGestureImage.setMaximumHeight(96)
        self.onlineTestCurrentGestureImage.setStyleSheet(
            "QLabel {"
            " border: 1px dashed #999999;"
            " border-radius: 6px;"
            " color: #777777;"
            " background-color: rgba(0, 0, 0, 0.02);"
            "}"
        )

        self.onlineTestNextGestureImage = QLabel("无图片")
        self.onlineTestNextGestureImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.onlineTestNextGestureImage.setMinimumHeight(56)
        self.onlineTestNextGestureImage.setMaximumHeight(96)
        self.onlineTestNextGestureImage.setStyleSheet(
            "QLabel {"
            " border: 1px dashed #999999;"
            " border-radius: 6px;"
            " color: #777777;"
            " background-color: rgba(0, 0, 0, 0.02);"
            "}"
        )

        target_row = QHBoxLayout()
        target_row.setSpacing(10)
        target_row.addWidget(self.onlineTestCountdownRing)

        current_column = QVBoxLayout()
        current_column.setSpacing(4)
        current_column.addWidget(self.onlineTestCurrentGestureLabel)
        current_column.addWidget(self.onlineTestCurrentGestureImage)

        next_column = QVBoxLayout()
        next_column.setSpacing(4)
        next_column.addWidget(self.onlineTestNextGestureLabel)
        next_column.addWidget(self.onlineTestNextGestureImage)

        target_row.addLayout(current_column, 1)
        target_row.addLayout(next_column, 1)
        self.onlineTestStatusLayout.addLayout(target_row)

        self.onlineTestStatusCard.viewLayout.addLayout(self.onlineTestStatusLayout)
        root_layout.addWidget(self.onlineTestStatusCard)

        self.gesturePanelCard = GesturePanelCard(self)
        root_layout.addWidget(self.gesturePanelCard, 1)

        root_layout.addStretch()

        self.connectionCard.refreshButton.clicked.connect(self.refreshPortsRequested.emit)
        self.connectionCard.startButton.clicked.connect(self._onStartSerialClicked)
        self.connectionCard.stopButton.clicked.connect(self.stopSerialRequested.emit)
        self.paramCard.modelSelectRequested.connect(self._onSelectModel)
        self.paramCard.applyRequested.connect(self._onApplyConfigClicked)
        self.paramCard.alphaChanged.connect(self._onEmaAlphaChanged)
        self.paramCard.voteWindowChanged.connect(self._onVoteWindowChanged)
        # Debug: print when vote window changes to verify handler and analyzer sync
        self.paramCard.voteWindowChanged.connect(
            lambda n: print(f"[DEBUG] voteWindowChanged signal: {int(n)} -> analyzer window now {self._analyzer.getVoteWindowSize()}")
        )
        self.onlineTestOutputDirButton.clicked.connect(self._onSelectOnlineTestOutputDir)
        self.onlineTestStartButton.clicked.connect(self._onStartOnlineTest)
        self.onlineTestStopButton.clicked.connect(lambda: self.stopOnlineTest("在线测试已手动停止", level="warning"))
        self.onlineTestRoundsSpin.valueChanged.connect(self._saveOnlineTestSettings)
        self.onlineTestActiveSecondsSpin.valueChanged.connect(self._saveOnlineTestSettings)
        self.onlineTestRestSecondsSpin.valueChanged.connect(self._saveOnlineTestSettings)
        self.onlineTestOutputDirEdit.editingFinished.connect(self._onOnlineTestOutputDirEdited)
        self._analyzer.inferenceReady.connect(self._onInferenceReady)
        self._analyzer.warningRaised.connect(lambda message: self.infoRequested.emit(message, "warning"))
        self._analyzer.errorRaised.connect(lambda message: self.infoRequested.emit(message, "error"))

        self._loadSettings()
        self._resetOnlineTestStatusUi()
        self._updateConnectionUi()
        self._updateOnlineTestUiState()

    def _loadSettings(self):
        saved_scan_freq = SettingService.value("Hardware/ScanFrequencyHz", 20, int)
        self.paramCard.setScanFrequency(int(saved_scan_freq))

        ema_alpha = SettingService.value(self.EMA_ALPHA_KEY, 0.5, float)
        self.paramCard.setEmaAlpha(float(ema_alpha))
        self._analyzer.setEmaAlpha(self.paramCard.getEmaAlpha())

        vote_window = SettingService.value(self.VOTE_WINDOW_KEY, 5, int)
        self.paramCard.setVoteWindowSize(int(vote_window))
        self._analyzer.setVoteWindowSize(int(self.paramCard.getVoteWindowSize()))
        self._analyzer.setVoteWindowSize(self.paramCard.getVoteWindowSize())

        output_dir = SettingService.value(self.ONLINE_OUTPUT_DIR_KEY, "", str)
        if output_dir:
            self.onlineTestOutputDirEdit.setText(output_dir)

        rounds = SettingService.value(self.ONLINE_ROUNDS_KEY, 3, int)
        active_seconds = SettingService.value(self.ONLINE_ACTIVE_SECONDS_KEY, 5, int)
        rest_seconds = SettingService.value(self.ONLINE_REST_SECONDS_KEY, 2, int)
        self.onlineTestRoundsSpin.setValue(max(1, min(100, int(rounds))))
        self.onlineTestActiveSecondsSpin.setValue(max(1, min(60, int(active_seconds))))
        self.onlineTestRestSecondsSpin.setValue(max(0, min(30, int(rest_seconds))))

        model_path = SettingService.value(self.MODEL_PATH_KEY, "", str)
        if model_path and Path(model_path).exists() and Path(model_path).suffix.lower() == ".onnx":
            self._loadModelAndUpdateUi(model_path, silent=True)

    def _saveSetting(self, key: str, value):
        SettingService.set_value(key, value, sync=True)

    def _updateConnectionUi(self):
        self.connectionCard.startButton.setEnabled(not self._serial_running)
        self.connectionCard.stopButton.setEnabled(self._serial_running)
        self._updateOnlineTestUiState()

    def _saveOnlineTestSettings(self):
        self._saveSetting(self.ONLINE_ROUNDS_KEY, int(self.onlineTestRoundsSpin.value()))
        self._saveSetting(self.ONLINE_ACTIVE_SECONDS_KEY, int(self.onlineTestActiveSecondsSpin.value()))
        self._saveSetting(self.ONLINE_REST_SECONDS_KEY, int(self.onlineTestRestSecondsSpin.value()))

    def _onStartSerialClicked(self):
        if self.connectionCard.portCombo.count() == 0:
            self.infoRequested.emit("没有可用串口", "warning")
            return

        port = self.connectionCard.portCombo.currentData()
        if not port:
            display_text = self.connectionCard.portCombo.currentText().strip()
            if display_text:
                tokens = display_text.split()
                if tokens:
                    port = tokens[0]

        if not port:
            self.infoRequested.emit("请选择串口", "warning")
            return

        self.startSerialRequested.emit(str(port))

    def _onSelectModel(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 ONNX 模型",
            "",
            "ONNX Model (*.onnx);;All Files (*)",
        )
        if not model_path:
            return
        self._loadModelAndUpdateUi(model_path, silent=False)

    def _loadModelAndUpdateUi(self, model_path: str, silent: bool):
        try:
            metadata, warnings = load_model_metadata(model_path)
            self._analyzer.loadModel(
                metadata.model_path,
                metadata.classes,
                metadata.sampling_depth,
                metadata.model_type,
            )
            self._analyzer.setEmaAlpha(self.paramCard.getEmaAlpha())
        except Exception as error:
            self.paramCard.setModelPath(model_path)
            self.paramCard.setSamplingDepth(None)
            self.gesturePanelCard.populateByClasses([], 0)
            self.gesturePanelCard.updateProbabilities({}, "")
            self.infoRequested.emit(f"模型元数据读取失败: {error}", "error")
            return

        self.paramCard.setModelPath(metadata.model_path)
        self.paramCard.setSamplingDepth(metadata.sampling_depth)
        self.gesturePanelCard.populateByClasses(metadata.classes, metadata.num_classes)
        self.gesturePanelCard.updateProbabilities({}, "")

        self._model_path = str(metadata.model_path)
        self._model_classes = list(metadata.classes)
        self._sampling_depth = int(metadata.sampling_depth)
        self._saveSetting(self.MODEL_PATH_KEY, metadata.model_path)
        self._saveSetting("Hardware/SampleDepth", int(metadata.sampling_depth))
        self._updateOnlineTestUiState()

        for warning in warnings:
            self.infoRequested.emit(warning, "warning")

        if not silent:
            self.infoRequested.emit(
                f"模型加载成功：{Path(metadata.model_path).name}（类别 {len(metadata.classes)}）",
                "success",
            )

    def _onApplyConfigClicked(self, scan_freq_hz: int, sample_depth: int):
        scan_freq_hz = max(1, min(60, int(scan_freq_hz)))
        sample_depth = int(sample_depth)
        if sample_depth <= 0:
            self.infoRequested.emit("请先加载模型以自动获取采样深度", "warning")
            return
        if sample_depth > 1024:
            self.infoRequested.emit(f"采样深度超出范围 [1, 1024]: {sample_depth}", "error")
            return

        self.applyConfigRequested.emit(scan_freq_hz, sample_depth)
        # self.infoRequested.emit("已发送参数设置，等待确认...", "info")

    def setSerialRunning(self, running: bool):
        self._serial_running = bool(running)
        if not self._serial_running and self._onlineTestActive:
            self.stopOnlineTest("串口已断开，在线测试已停止", level="warning")
        self._updateConnectionUi()

    def updatePortOptions(self, options: List[Tuple[str, str]], selected_port: str = ""):
        selected = str(selected_port or "")
        self.connectionCard.portCombo.blockSignals(True)
        self.connectionCard.portCombo.clear()

        for display_text, device in options:
            self.connectionCard.portCombo.addItem(str(display_text), str(device))

        if selected:
            index = self.connectionCard.portCombo.findData(selected)
            if index >= 0:
                self.connectionCard.portCombo.setCurrentIndex(index)

        self.connectionCard.portCombo.blockSignals(False)

    def updatePowerData(self, power_data: dict):
        bus_millivolt = int(power_data.get("bus_mv", 0))
        current_milliamp = int(power_data.get("current_ma", 0))
        power_milliwatt = int(power_data.get("power_mw", 0))

        self.connectionCard.devVoltageLabel.setText(f"总线电压: {bus_millivolt / 1000.0:.2f}V")
        self.connectionCard.devCurrentLabel.setText(f"电流: {current_milliamp}mA")
        self.connectionCard.devPowerLabel.setText(f"功率: {power_milliwatt}mW")

    def onConfigReady(self, config_dict: dict):
        scan_freq_hz = int(config_dict.get("scan_frequency_hz", self.paramCard.scanFreqSpin.value()) or self.paramCard.scanFreqSpin.value())
        sample_depth = int(config_dict.get("sample_depth", self.paramCard.getSamplingDepth()) or self.paramCard.getSamplingDepth())
        self.paramCard.setScanFrequency(scan_freq_hz)
        if sample_depth > 0:
            self.paramCard.setSamplingDepth(sample_depth)
            self._sampling_depth = sample_depth

    def _onInferenceReady(self, payload: dict):
        probabilities = payload.get("probabilities", {})
        top_name = str(payload.get("top_name", "") or "")
        top_prob = float(payload.get("top_prob", 0.0) or 0.0)
        if top_prob < 0.5:
            top_name = ""
        self.gesturePanelCard.updateProbabilities(probabilities, top_name, threshold=0.5)

        if self._onlineTestActive and self._onlineTestCollecting and self._onlineTestCurrentTarget:
            self._onlineTestCurrentTotalFrames += 1
            if top_name == self._onlineTestCurrentTarget:
                self._onlineTestCurrentCorrectFrames += 1
            self._refreshOnlineTestProgressLabel()

    def _onEmaAlphaChanged(self, alpha: float):
        value = max(0.0, min(1.0, float(alpha)))
        self._analyzer.setEmaAlpha(value)
        self._saveSetting(self.EMA_ALPHA_KEY, value)

    def _onVoteWindowChanged(self, n: int):
        value = max(1, min(99, int(n)))
        self._analyzer.setVoteWindowSize(value)
        self._saveSetting(self.VOTE_WINDOW_KEY, int(value))

    def _onSelectOnlineTestOutputDir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择在线测试报告目录", "")
        if not directory:
            return
        self.onlineTestOutputDirEdit.setText(directory)
        self._saveSetting(self.ONLINE_OUTPUT_DIR_KEY, directory)
        self._updateOnlineTestUiState()

    def _onOnlineTestOutputDirEdited(self):
        output_dir = self.onlineTestOutputDirEdit.text().strip()
        self._saveSetting(self.ONLINE_OUTPUT_DIR_KEY, output_dir)
        self._updateOnlineTestUiState()

    def _orderedModelGestures(self) -> List[str]:
        model_set = {name.strip().upper() for name in self._model_classes if name.strip()}
        ordered = [name for name in DEFAULT_GESTURE_ORDER if name in model_set]
        leftovers = [name for name in self._model_classes if name not in ordered]
        return ordered + leftovers

    def _buildOnlineTestPlan(self, gestures: List[str], rounds: int) -> List[str]:
        if not gestures or rounds <= 0:
            return []

        plan: List[str] = []
        previous_last = ""
        for _ in range(rounds):
            round_plan = list(gestures)
            random.shuffle(round_plan)
            if previous_last and round_plan and round_plan[0] == previous_last and len(round_plan) > 1:
                round_plan = round_plan[1:] + round_plan[:1]
            plan.extend(round_plan)
            previous_last = round_plan[-1]
        return plan

    def _onlineUpcomingGestures(self) -> Tuple[str, str]:
        first = self._onlineTestPlan[self._onlineTestGestureIndex + 1] if self._onlineTestGestureIndex + 1 < len(self._onlineTestPlan) else ""
        second = self._onlineTestPlan[self._onlineTestGestureIndex + 2] if self._onlineTestGestureIndex + 2 < len(self._onlineTestPlan) else ""
        return first, second

    def _setOnlineCountdown(self, total_seconds: float, remaining_tenths: int):
        total_tenths = max(1, int(round(max(0.1, float(total_seconds)) * 10.0)))
        remaining = max(0, min(total_tenths, int(remaining_tenths)))
        self.onlineTestCountdownRing.setRange(0, total_tenths)
        self.onlineTestCountdownRing.setValue(remaining)
        self.onlineTestCountdownRing.setFormat(f"{remaining / 10.0:.1f} s")

    def _onlineGestureImagePath(self, gesture: str) -> Path:
        return Path(__file__).resolve().parents[1] / "assets" / "gesture_figures" / f"{gesture}.png"

    def _applyOnlineGesturePixmap(self, target_label: QLabel, pixmap: Optional[QPixmap]):
        if pixmap is None or pixmap.isNull():
            target_label.setPixmap(QPixmap())
            target_label.setText("无图片")
            target_label.setStyleSheet(
                "QLabel {"
                " border: 1px dashed #999999;"
                " border-radius: 6px;"
                " color: #777777;"
                " background-color: rgba(0, 0, 0, 0.02);"
                "}"
            )
            return

        scaled = pixmap.scaled(
            max(1, target_label.width()),
            max(1, target_label.maximumHeight()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        target_label.setPixmap(scaled)
        target_label.setText("")
        target_label.setStyleSheet("QLabel { border: none; background: transparent; }")

    def _setOnlineGestureImages(self, current_target: str = "", next_target: str = ""):
        current_path = self._onlineGestureImagePath(current_target) if current_target else None
        next_path = self._onlineGestureImagePath(next_target) if next_target else None

        if current_path and current_path.exists():
            self._onlineTestCurrentGesturePixmap = QPixmap(str(current_path))
        else:
            self._onlineTestCurrentGesturePixmap = None

        if next_path and next_path.exists():
            self._onlineTestNextGesturePixmap = QPixmap(str(next_path))
        else:
            self._onlineTestNextGesturePixmap = None

        self._applyOnlineGesturePixmap(self.onlineTestCurrentGestureImage, self._onlineTestCurrentGesturePixmap)
        self._applyOnlineGesturePixmap(self.onlineTestNextGestureImage, self._onlineTestNextGesturePixmap)

    def _setOnlineStateText(self, title: str, current_target: str = "", next_target: str = ""):
        self.onlineTestStateLabel.setText(f"状态: {title}")
        self._applyOnlineStateColor(title)
        self.onlineTestCurrentGestureLabel.setText(f"当前目标手势: {current_target or '-'}")
        self.onlineTestNextGestureLabel.setText(f"下一个手势: {next_target or '-'}")
        self._setOnlineGestureImages(current_target=current_target, next_target=next_target)

    def _applyOnlineStateColor(self, title: str):
        if title in {"休息中", "已停止"}:
            self.onlineTestStateLabel.setStyleSheet(f"color: {self.STATUS_COLOR_STOPPED_OR_REST};")
            return
        if title == "采集中":
            self.onlineTestStateLabel.setStyleSheet(f"color: {self.STATUS_COLOR_ACTIVE};")
            return
        self.onlineTestStateLabel.setStyleSheet("")

    def _refreshOnlineTestProgressLabel(self):
        total = max(0, int(self._onlineTestTotalTasks))
        completed = max(0, min(total, int(self._onlineTestCompletedTasks)))
        self.onlineTestProgressLabel.setText(
            f"进度: {completed}/{total} | 本轮命中: {self._onlineTestCurrentCorrectFrames}/{self._onlineTestCurrentTotalFrames}"
        )
        self.onlineTestOverallProgressBar.setRange(0, max(1, total))
        self.onlineTestOverallProgressBar.setValue(completed)

    def _resetOnlineTestStatusUi(self):
        self._setOnlineStateText("待开始")
        self.onlineTestProgressLabel.setText("进度: 0/0")
        self.onlineTestOverallProgressBar.setRange(0, 1)
        self.onlineTestOverallProgressBar.setValue(0)
        self.onlineTestCountdownRing.setRange(0, 10)
        self.onlineTestCountdownRing.setValue(0)
        self.onlineTestCountdownRing.setFormat("0.0 s")

    def _resetOnlineTestRuntimeState(self):
        self._onlineTestActive = False
        self._onlineTestCollecting = False
        self._onlineTestPlan = []
        self._onlineTestGestureIndex = -1
        self._onlineTestTotalTasks = 0
        self._onlineTestCompletedTasks = 0
        self._onlineTestRemainingTenths = 0
        self._onlineTestCurrentTarget = ""
        self._onlineTestCurrentCorrectFrames = 0
        self._onlineTestCurrentTotalFrames = 0
        self._onlineTestResults = []
        self._onlineTestGestureStats = {}

    def _updateOnlineTestUiState(self):
        output_dir = self.onlineTestOutputDirEdit.text().strip()
        can_start = (
            not self._onlineTestActive
            and self._serial_running
            and bool(self._model_classes)
            and bool(self._model_path)
            and bool(output_dir)
        )
        self.onlineTestStartButton.setEnabled(can_start)
        self.onlineTestStopButton.setEnabled(self._onlineTestActive)
        self.onlineTestRoundsSpin.setEnabled(not self._onlineTestActive)
        self.onlineTestActiveSecondsSpin.setEnabled(not self._onlineTestActive)
        self.onlineTestRestSecondsSpin.setEnabled(not self._onlineTestActive)
        self.onlineTestOutputDirButton.setEnabled(not self._onlineTestActive)

    def _onStartOnlineTest(self):
        if self._onlineTestActive:
            self.infoRequested.emit("在线测试已在运行", "warning")
            return
        if not self._serial_running:
            self.infoRequested.emit("请先连接串口", "warning")
            return
        if not self._analyzer.hasModel() or not self._model_classes:
            self.infoRequested.emit("请先加载有效 ONNX 模型", "warning")
            return

        output_dir = self.onlineTestOutputDirEdit.text().strip()
        ok_dir, dir_error = ensureOutputDir(output_dir)
        if not ok_dir:
            self.infoRequested.emit(f"输出目录不可用: {dir_error}", "error")
            return

        preprocess_service = PreprocessParamService.get_instance()
        preprocess_service.update_param("standardizePerFrame", True)
        preprocess_service.update_param("normalizePerFrame", False)

        main_window = self.window()
        plot_panel = getattr(main_window, "plotPanel", None)
        if plot_panel is not None:
            plot_panel.setRealtimePreprocessEnabled(True, emit_signal=True)

        rounds = int(self.onlineTestRoundsSpin.value())
        gesture_order = self._orderedModelGestures()
        plan = self._buildOnlineTestPlan(gesture_order, rounds)
        if not plan:
            self.infoRequested.emit("测试计划为空，请检查模型类别", "warning")
            return

        self._onlineTestActive = True
        self._onlineTestCollecting = False
        self._onlineTestPlan = plan
        self._onlineTestGestureIndex = -1
        self._onlineTestTotalTasks = len(plan)
        self._onlineTestCompletedTasks = 0
        self._onlineTestRemainingTenths = 0
        self._onlineTestCurrentTarget = ""
        self._onlineTestCurrentCorrectFrames = 0
        self._onlineTestCurrentTotalFrames = 0
        self._onlineTestResults = []
        self._onlineTestGestureStats = {
            gesture: {"correct": 0, "total": 0}
            for gesture in gesture_order
        }

        initial_rest_seconds = int(self.onlineTestRestSecondsSpin.value())
        self._onlineTestCollecting = False
        self._onlineTestRemainingTenths = max(0, initial_rest_seconds * 10)
        first_target, second_target = self._onlineUpcomingGestures()
        self._setOnlineCountdown(float(initial_rest_seconds), self._onlineTestRemainingTenths)
        self._setOnlineStateText("休息中", current_target=first_target, next_target=second_target)
        self._refreshOnlineTestProgressLabel()

        self._onlineTestTimer.start()
        self._saveOnlineTestSettings()
        self._updateOnlineTestUiState()
        self.infoRequested.emit(f"在线测试已开始，共 {self._onlineTestTotalTasks} 个任务", "info")

    def _startNextOnlineTestGesture(self):
        self._onlineTestGestureIndex += 1
        if self._onlineTestGestureIndex >= len(self._onlineTestPlan):
            self._finishOnlineTest()
            return

        self._onlineTestCurrentTarget = self._onlineTestPlan[self._onlineTestGestureIndex]
        self._onlineTestCurrentCorrectFrames = 0
        self._onlineTestCurrentTotalFrames = 0
        self._onlineTestCollecting = True

        self._onlineTestRemainingTenths = int(self.onlineTestActiveSecondsSpin.value()) * 10
        next_target, _ = self._onlineUpcomingGestures()
        self._setOnlineCountdown(float(self.onlineTestActiveSecondsSpin.value()), self._onlineTestRemainingTenths)
        self._setOnlineStateText("采集中", current_target=self._onlineTestCurrentTarget, next_target=next_target)
        self._refreshOnlineTestProgressLabel()

    def _finalizeCurrentOnlineGesture(self):
        if not self._onlineTestCurrentTarget:
            return

        total = int(self._onlineTestCurrentTotalFrames)
        correct = int(self._onlineTestCurrentCorrectFrames)
        accuracy = (float(correct) / float(total)) if total > 0 else 0.0

        self._onlineTestResults.append(
            {
                "index": float(self._onlineTestCompletedTasks + 1),
                "gesture": self._onlineTestCurrentTarget,
                "correct": float(correct),
                "total": float(total),
                "accuracy": float(accuracy),
            }
        )

        stats = self._onlineTestGestureStats.setdefault(self._onlineTestCurrentTarget, {"correct": 0, "total": 0})
        stats["correct"] += correct
        stats["total"] += total

        self._onlineTestCompletedTasks += 1
        self._refreshOnlineTestProgressLabel()

    def _onOnlineTestTimerTick(self):
        if not self._onlineTestActive:
            self._onlineTestTimer.stop()
            return

        if self._onlineTestRemainingTenths > 0:
            self._onlineTestRemainingTenths -= 1
            self.onlineTestCountdownRing.setValue(max(0, self._onlineTestRemainingTenths))
            self.onlineTestCountdownRing.setFormat(f"{self._onlineTestRemainingTenths / 10.0:.1f} s")

            if self._onlineTestCollecting:
                next_target, _ = self._onlineUpcomingGestures()
                self._setOnlineStateText("采集中", current_target=self._onlineTestCurrentTarget, next_target=next_target)
            else:
                next_target, upcoming = self._onlineUpcomingGestures()
                self._setOnlineStateText("休息中", current_target=next_target, next_target=upcoming)
            return

        if self._onlineTestCollecting:
            self._finalizeCurrentOnlineGesture()
            self._onlineTestCollecting = False

            if self._onlineTestCompletedTasks >= self._onlineTestTotalTasks:
                self._finishOnlineTest()
                return

            rest_seconds = int(self.onlineTestRestSecondsSpin.value())
            self._onlineTestRemainingTenths = rest_seconds * 10
            if self._onlineTestRemainingTenths <= 0:
                self._startNextOnlineTestGesture()
                return

            next_target, upcoming = self._onlineUpcomingGestures()
            self._setOnlineCountdown(float(rest_seconds), self._onlineTestRemainingTenths)
            self._setOnlineStateText("休息中", current_target=next_target, next_target=upcoming)
            return

        self._startNextOnlineTestGesture()

    def _safeModelName(self) -> str:
        model_name = Path(self._model_path).stem if self._model_path else "model"
        cleaned = re.sub(r'[<>:"/\\|?*]+', "_", model_name).strip(" .")
        return cleaned or "model"

    def _writeOnlineTestReport(self, report_path: Path):
        classes = list(self._orderedModelGestures())
        lines: List[str] = []
        lines.append("在线测试报告")
        lines.append(f"模型名称: {self._safeModelName()}")
        lines.append(f"模型路径: {self._model_path}")
        lines.append(f"手势种类: {', '.join(classes)}")
        lines.append(f"采样深度: {int(self._sampling_depth or self.paramCard.getSamplingDepth())}")
        lines.append(f"扫描频率(Hz): {int(self.paramCard.scanFreqSpin.value())}")
        lines.append(f"轮次数: {int(self.onlineTestRoundsSpin.value())}")
        lines.append(f"单轮时长(s): {int(self.onlineTestActiveSecondsSpin.value())}")
        lines.append(f"休息时长(s): {int(self.onlineTestRestSecondsSpin.value())}")
        lines.append("")
        lines.append("[单轮识别结果]")

        for result in self._onlineTestResults:
            total = int(result["total"])
            correct = int(result["correct"])
            accuracy = (correct / total) if total > 0 else 0.0
            lines.append(
                f"轮次{int(result['index'])} | 手势={result['gesture']} | 正确帧={correct} | 总帧={total} | 正确率={accuracy * 100.0:.2f}%"
            )

        lines.append("")
        lines.append("[手势汇总]")
        overall_correct = 0
        overall_total = 0
        for gesture in classes:
            stats = self._onlineTestGestureStats.get(gesture, {"correct": 0, "total": 0})
            total = int(stats.get("total", 0))
            correct = int(stats.get("correct", 0))
            overall_total += total
            overall_correct += correct
            accuracy = (correct / total) if total > 0 else 0.0
            lines.append(
                f"手势={gesture} | 正确帧={correct} | 总帧={total} | 正确率={accuracy * 100.0:.2f}%"
            )

        overall_accuracy = (overall_correct / overall_total) if overall_total > 0 else 0.0
        lines.append("")
        lines.append("[总体统计]")
        lines.append(
            f"总正确帧={overall_correct} | 总帧={overall_total} | 总正确率={overall_accuracy * 100.0:.2f}%"
        )

        report_path.write_text("\n".join(lines), encoding="utf-8")

    def _exportOnlineTestBarChart(self, chart_path: Path) -> bool:
        classes = list(self._orderedModelGestures())
        x_labels: List[str] = []
        y_values: List[float] = []
        overall_correct = 0
        overall_total = 0
        for gesture in classes:
            stats = self._onlineTestGestureStats.get(gesture, {"correct": 0, "total": 0})
            total = int(stats.get("total", 0))
            correct = int(stats.get("correct", 0))
            overall_total += total
            overall_correct += correct
            acc = (correct / total) if total > 0 else 0.0
            x_labels.append(gesture)
            y_values.append(acc * 100.0)

        overall_acc = (overall_correct / overall_total) if overall_total > 0 else 0.0
        x_labels.append("ALL")
        y_values.append(overall_acc * 100.0)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(10, 5), dpi=140)
            ax = fig.add_subplot(111)
            bar_colors = ["#1b5e20"] * max(0, len(x_labels) - 1) + ["#e4360f"]
            bars = ax.bar(np.arange(len(x_labels)), y_values, color=bar_colors)
            ax.set_ylim(0.0, 100.0)
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Online Test Accuracy by Gesture")
            ax.grid(axis="y", alpha=0.25)
            for idx, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() * 0.5, bar.get_height() + 1.0, f"{y_values[idx]:.1f}", ha="center", va="bottom", fontsize=8)
            fig.tight_layout()
            fig.savefig(str(chart_path))
            plt.close(fig)
            return True
        except Exception as error:
            self.infoRequested.emit(f"柱状图导出失败（已跳过）: {error}", "warning")
            return False

    def _finishOnlineTest(self):
        if not self._onlineTestActive:
            return

        self._onlineTestTimer.stop()

        output_dir = self.onlineTestOutputDirEdit.text().strip()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        base_name = f"{self._safeModelName()}_{timestamp}_online_report"
        report_path = Path(output_dir) / f"{base_name}.txt"
        chart_path = Path(output_dir) / f"{base_name}.png"

        try:
            self._writeOnlineTestReport(report_path)
            chart_ok = self._exportOnlineTestBarChart(chart_path)
            if chart_ok:
                self.infoRequested.emit(f"在线测试完成，报告已导出: {report_path}", "success")
                self.infoRequested.emit(f"在线测试统计图已导出: {chart_path}", "success")
            else:
                self.infoRequested.emit(f"在线测试完成，报告已导出: {report_path}", "success")
        except Exception as error:
            self.infoRequested.emit(f"在线测试报告导出失败: {error}", "error")

        self._setOnlineStateText("已完成")
        self._onlineTestActive = False
        self._onlineTestCollecting = False
        self._onlineTestCurrentTarget = ""
        self._onlineTestCurrentCorrectFrames = 0
        self._onlineTestCurrentTotalFrames = 0
        self._updateOnlineTestUiState()

    def stopOnlineTest(self, message: str = "", level: str = "warning"):
        if not self._onlineTestActive:
            return

        self._onlineTestTimer.stop()
        self._onlineTestActive = False
        self._onlineTestCollecting = False
        self._onlineTestCurrentTarget = ""
        self._onlineTestCurrentCorrectFrames = 0
        self._onlineTestCurrentTotalFrames = 0
        self._setOnlineStateText("已停止")
        self.onlineTestCountdownRing.setValue(0)
        self.onlineTestCountdownRing.setFormat("0.0 s")
        self._updateOnlineTestUiState()
        if message:
            self.infoRequested.emit(message, level)

    def isOnlineTestActive(self) -> bool:
        return bool(self._onlineTestActive)

    def resizeEvent(self, a0):
        self._applyOnlineGesturePixmap(self.onlineTestCurrentGestureImage, self._onlineTestCurrentGesturePixmap)
        self._applyOnlineGesturePixmap(self.onlineTestNextGestureImage, self._onlineTestNextGesturePixmap)
        super().resizeEvent(a0)
