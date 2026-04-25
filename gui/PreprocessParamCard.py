"""UI card for editing shared preprocessing parameters."""

from __future__ import annotations

from typing import Dict, List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    DoubleSpinBox,
    HeaderCardWidget,
    SingleDirectionScrollArea,
    SpinBox,
    StrongBodyLabel,
    SwitchButton,
)

from utils.PreProcess import PreprocessParams
from utils.PreprocessParamService import PreprocessParamService


class PreprocessParamCard(HeaderCardWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("预处理参数")
        self._service = PreprocessParamService.get_instance()
        self._params = self._service.get_params()
        self._updating_ui = False

        self._float_controls: Dict[str, DoubleSpinBox] = {}
        self._bool_controls: Dict[str, SwitchButton] = {}
        self._list_controls: Dict[str, List] = {"butterLowHzList": [], "butterHighHzList": [], "butterOrderList": []}

        self._scrollArea = SingleDirectionScrollArea(self)
        self._scrollArea.setWidgetResizable(True)

        content = QWidget(self)
        self._contentLayout = QVBoxLayout(content)
        self._contentLayout.setContentsMargins(12, 12, 12, 12)
        self._contentLayout.setSpacing(10)

        self._buildContent()

        self._scrollArea.setWidget(content)
        self.viewLayout.addWidget(self._scrollArea)

        self._service.parametersChanged.connect(self._applyParams)
        self._applyParams(self._params)

    def _buildContent(self):
        self._addSectionTitle("基础参数")
        self._addFloatRow("samplingFreqHz", "采样频率 (Hz)", 1.0, 20_000_000.0, 100_000.0, 0)
        self._addFloatRow("tgcAlpha", "TGC 强度", 0.0, 5.0, 0.01, 4)
        self._addFloatRow("samplingBalance", "ADC 基线", -100000.0, 100000.0, 1.0, 2)
        self._addFloatRow("burstToneFreqHz", "发射中心频率 (Hz)", 1.0, 20_000_000.0, 100_000.0, 0)
        self._addFloatRow("inspectDepth_m", "等效探测深度 (m)", 0.0, 1.0, 0.001, 4)

        self._addSectionTitle("带通滤波")
        self._addFloatListRows("butterLowHzList", "低截止频率 (Hz)", 1.0, 20_000_000.0, 100_000.0, 0)
        self._addFloatListRows("butterHighHzList", "高截止频率 (Hz)", 1.0, 20_000_000.0, 100_000.0, 0)
        self._addIntListRows("butterOrderList", "滤波器阶数", 1, 12, 1)

        self._addSectionTitle("图像处理")
        self._addFloatRow("logCompRatio", "对数压缩系数", 0.0, 10.0, 0.01, 4)
        self._addBoolRow("enableLogCompression", "启用对数压缩")
        self._addBoolRow("standardizePerFrame", "逐帧标准化")
        self._addBoolRow("normalizePerFrame", "逐帧归一化")

        self._contentLayout.addStretch()

    def _addSectionTitle(self, title: str):
        label = StrongBodyLabel(title)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._contentLayout.addWidget(label)

    def _createRowWidget(self, label_text: str, control_widget: QWidget) -> QWidget:
        row_widget = QWidget(self)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(10)

        label = BodyLabel(label_text)
        label.setMinimumWidth(160)
        row_layout.addWidget(label)
        row_layout.addWidget(control_widget, 1)
        return row_widget

    def _addFloatRow(self, field_name: str, label_text: str, minimum: float, maximum: float, step: float, decimals: int):
        spin = DoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.valueChanged.connect(lambda value, name=field_name: self._onFloatChanged(name, value))
        self._float_controls[field_name] = spin
        self._contentLayout.addWidget(self._createRowWidget(label_text, spin))

    def _addBoolRow(self, field_name: str, label_text: str):
        switch = SwitchButton(self)
        switch.checkedChanged.connect(lambda checked, name=field_name: self._onBoolChanged(name, checked))
        self._bool_controls[field_name] = switch
        self._contentLayout.addWidget(self._createRowWidget(label_text, switch))

    def _addFloatListRows(self, field_name: str, label_text: str, minimum: float, maximum: float, step: float, decimals: int):
        for index in range(4):
            spin = DoubleSpinBox(self)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.valueChanged.connect(
                lambda value, name=field_name, item_index=index: self._onFloatListChanged(name, item_index, value)
            )
            self._list_controls[field_name].append(spin)
            self._contentLayout.addWidget(self._createRowWidget(f"{label_text} - CH{index + 1}", spin))

    def _addIntListRows(self, field_name: str, label_text: str, minimum: int, maximum: int, step: int = 1):
        for index in range(4):
            spin = SpinBox(self)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.valueChanged.connect(
                lambda value, name=field_name, item_index=index: self._onIntListChanged(name, item_index, value)
            )
            self._list_controls[field_name].append(spin)
            self._contentLayout.addWidget(self._createRowWidget(f"{label_text} - CH{index + 1}", spin))

    def _onFloatChanged(self, field_name: str, value: float):
        if self._updating_ui:
            return
        self._service.update_param(field_name, float(value))

    def _onBoolChanged(self, field_name: str, checked: bool):
        if self._updating_ui:
            return
        self._service.update_param(field_name, bool(checked))

    def _onFloatListChanged(self, field_name: str, index: int, value: float):
        if self._updating_ui:
            return
        self._service.update_list_item(field_name, index, float(value))

    def _onIntListChanged(self, field_name: str, index: int, value: int):
        if self._updating_ui:
            return
        self._service.update_list_item(field_name, index, int(value))

    def _applyParams(self, params: PreprocessParams):
        self._updating_ui = True
        try:
            self._float_controls["samplingFreqHz"].setValue(float(params.samplingFreqHz))
            self._float_controls["tgcAlpha"].setValue(float(params.tgcAlpha))
            self._float_controls["samplingBalance"].setValue(float(params.samplingBalance))
            self._float_controls["burstToneFreqHz"].setValue(float(params.burstToneFreqHz))
            self._float_controls["inspectDepth_m"].setValue(float(params.inspectDepth_m))
            self._float_controls["logCompRatio"].setValue(float(params.logCompRatio))

            self._bool_controls["enableLogCompression"].setChecked(bool(params.enableLogCompression))
            self._bool_controls["standardizePerFrame"].setChecked(bool(params.standardizePerFrame))
            self._bool_controls["normalizePerFrame"].setChecked(bool(params.normalizePerFrame))

            for field_name in ("butterLowHzList", "butterHighHzList"):
                for index, spin in enumerate(self._list_controls[field_name]):
                    spin.setValue(float(getattr(params, field_name)[index]))

            for index, spin in enumerate(self._list_controls["butterOrderList"]):
                spin.setValue(int(getattr(params, "butterOrderList")[index]))
        finally:
            self._updating_ui = False