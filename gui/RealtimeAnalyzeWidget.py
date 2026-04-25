"""Realtime analysis page UI (frontend only)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QWidget

from gui.AnalyzeParamCard import AnalyzeParamCard
from gui.ConnectionCard import ConnectionCard
from gui.GesturePanelCard import GesturePanelCard
from utils.ModelMetadataService import load_model_metadata
from utils.RTAnalyzer import RTAnalyzer
from utils.SettingService import SettingService


class RealtimeAnalyzeWidget(QWidget):
    """Online analysis page for model selection and gesture class display."""

    MODEL_PATH_KEY = "Analysis/ModelPath"

    startSerialRequested = pyqtSignal(str)
    stopSerialRequested = pyqtSignal()
    refreshPortsRequested = pyqtSignal()
    applyConfigRequested = pyqtSignal(int, int)
    infoRequested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._serial_running = False
        self._sampling_depth = 0
        self._analyzer = RTAnalyzer.getInstance()

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(6)

        self.connectionCard = ConnectionCard(self)
        root_layout.addWidget(self.connectionCard)

        self.paramCard = AnalyzeParamCard(self)
        root_layout.addWidget(self.paramCard)

        self.gesturePanelCard = GesturePanelCard(self)
        root_layout.addWidget(self.gesturePanelCard, 1)

        root_layout.addStretch()

        self.connectionCard.refreshButton.clicked.connect(self.refreshPortsRequested.emit)
        self.connectionCard.startButton.clicked.connect(self._onStartSerialClicked)
        self.connectionCard.stopButton.clicked.connect(self.stopSerialRequested.emit)
        self.paramCard.modelSelectRequested.connect(self._onSelectModel)
        self.paramCard.applyRequested.connect(self._onApplyConfigClicked)
        self._analyzer.inferenceReady.connect(self._onInferenceReady)
        self._analyzer.warningRaised.connect(lambda message: self.infoRequested.emit(message, "warning"))
        self._analyzer.errorRaised.connect(lambda message: self.infoRequested.emit(message, "error"))

        self._loadSettings()
        self._updateConnectionUi()

    def _loadSettings(self):
        saved_scan_freq = SettingService.value("Hardware/ScanFrequencyHz", 20, int)
        self.paramCard.setScanFrequency(int(saved_scan_freq))

        model_path = SettingService.value(self.MODEL_PATH_KEY, "", str)
        if model_path and Path(model_path).exists() and Path(model_path).suffix.lower() == ".onnx":
            self._loadModelAndUpdateUi(model_path, silent=True)

    def _saveSetting(self, key: str, value):
        SettingService.set_value(key, value, sync=True)

    def _updateConnectionUi(self):
        self.connectionCard.startButton.setEnabled(not self._serial_running)
        self.connectionCard.stopButton.setEnabled(self._serial_running)

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
            self._analyzer.loadModel(metadata.model_path, metadata.classes, metadata.sampling_depth)
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

        self._sampling_depth = int(metadata.sampling_depth)
        self._saveSetting(self.MODEL_PATH_KEY, metadata.model_path)
        self._saveSetting("Hardware/SampleDepth", int(metadata.sampling_depth))

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

    def _onInferenceReady(self, payload: dict):
        probabilities = payload.get("probabilities", {})
        top_name = str(payload.get("top_name", "") or "")
        top_prob = float(payload.get("top_prob", 0.0) or 0.0)
        if top_prob < 0.5:
            top_name = ""
        self.gesturePanelCard.updateProbabilities(probabilities, top_name, threshold=0.5)
