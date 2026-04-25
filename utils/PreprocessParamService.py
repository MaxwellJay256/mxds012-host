"""Shared persistence and mutation service for preprocessing parameters."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Optional

from PyQt5.QtCore import QObject, pyqtSignal

from utils.PreProcess import PreprocessParams
from utils.SettingService import SettingService


class PreprocessParamService(QObject):
    parametersChanged = pyqtSignal(object)

    _instance: Optional["PreprocessParamService"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = SettingService.get_settings()
        self._params = self._loadParams()

    @classmethod
    def get_instance(cls) -> "PreprocessParamService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_params(self) -> PreprocessParams:
        return self._params

    def get_params_copy(self) -> PreprocessParams:
        return deepcopy(self._params)

    def set_params(self, params: PreprocessParams, emit_signal: bool = True):
        self._params = params
        self._saveParams(params)
        if emit_signal:
            self.parametersChanged.emit(self._params)

    def update_param(self, name: str, value: Any, emit_signal: bool = True):
        if not hasattr(self._params, name):
            raise AttributeError(f"Unknown preprocessing parameter: {name}")

        setattr(self._params, name, value)
        self._saveValue(name, value)
        if emit_signal:
            self.parametersChanged.emit(self._params)

    def update_list_item(self, name: str, index: int, value: Any, emit_signal: bool = True):
        items = list(getattr(self._params, name))
        if index < 0 or index >= len(items):
            raise IndexError(f"{name}[{index}] is out of range")

        items[index] = value
        setattr(self._params, name, items)
        self._saveValue(name, items)
        if emit_signal:
            self.parametersChanged.emit(self._params)

    def refresh_from_settings(self, emit_signal: bool = True):
        self._params = self._loadParams()
        if emit_signal:
            self.parametersChanged.emit(self._params)

    def _settingsKey(self, field_name: str) -> str:
        return f"PreprocessParams/{field_name}"

    def _loadList(self, field_name: str, default_values):
        raw_value = self._settings.value(self._settingsKey(field_name), None)
        if raw_value is None:
            return list(default_values)

        if isinstance(raw_value, str):
            try:
                parsed = json.loads(raw_value)
            except Exception:
                return list(default_values)
        elif isinstance(raw_value, (list, tuple)):
            parsed = raw_value
        else:
            return list(default_values)

        result = []
        for item in parsed:
            try:
                result.append(float(item) if field_name != "butterOrderList" else int(item))
            except (TypeError, ValueError):
                return list(default_values)
        return result if result else list(default_values)

    def _loadParams(self) -> PreprocessParams:
        params = PreprocessParams()
        params.samplingFreqHz = float(self._settings.value(self._settingsKey("samplingFreqHz"), params.samplingFreqHz, type=float))
        params.tgcAlpha = float(self._settings.value(self._settingsKey("tgcAlpha"), params.tgcAlpha, type=float))
        params.samplingBalance = float(self._settings.value(self._settingsKey("samplingBalance"), params.samplingBalance, type=float))
        params.burstToneFreqHz = float(self._settings.value(self._settingsKey("burstToneFreqHz"), params.burstToneFreqHz, type=float))
        params.inspectDepth_m = float(self._settings.value(self._settingsKey("inspectDepth_m"), params.inspectDepth_m, type=float))
        params.butterLowHzList = self._loadList("butterLowHzList", params.butterLowHzList)
        params.butterHighHzList = self._loadList("butterHighHzList", params.butterHighHzList)
        params.butterOrderList = [int(value) for value in self._loadList("butterOrderList", params.butterOrderList)]
        params.enableLogCompression = bool(self._settings.value(self._settingsKey("enableLogCompression"), params.enableLogCompression, type=bool))
        params.logCompRatio = float(self._settings.value(self._settingsKey("logCompRatio"), params.logCompRatio, type=float))
        params.standardizePerFrame = bool(self._settings.value(self._settingsKey("standardizePerFrame"), params.standardizePerFrame, type=bool))
        params.normalizePerFrame = bool(self._settings.value(self._settingsKey("normalizePerFrame"), params.normalizePerFrame, type=bool))
        return params

    def _saveValue(self, field_name: str, value: Any):
        key = self._settingsKey(field_name)
        if isinstance(value, (list, tuple)):
            self._settings.setValue(key, json.dumps(list(value)))
        else:
            self._settings.setValue(key, value)
        self._settings.sync()

    def _saveParams(self, params: PreprocessParams):
        for field_name in (
            "samplingFreqHz",
            "tgcAlpha",
            "samplingBalance",
            "burstToneFreqHz",
            "inspectDepth_m",
            "butterLowHzList",
            "butterHighHzList",
            "butterOrderList",
            "enableLogCompression",
            "logCompRatio",
            "standardizePerFrame",
            "normalizePerFrame",
        ):
            self._saveValue(field_name, getattr(params, field_name))