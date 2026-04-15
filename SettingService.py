"""Centralized application settings service based on QSettings(INI)."""
import sys
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import QSettings, QStandardPaths


class SettingService:
    """Provide a shared QSettings instance and path strategy for the app."""

    ORG_NAME = "KeineStudio"
    APP_NAME = "MXDS012-Host"
    FILE_NAME = "mxds012-host.ini"

    _settings: Optional[QSettings] = None
    _settings_path: Optional[Path] = None
    _last_error: str = ""

    @classmethod
    def get_settings_path(cls) -> Path:
        if cls._settings_path is not None:
            return cls._settings_path

        if getattr(sys, "frozen", False):
            base_dir = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
            base_path = Path(base_dir) if base_dir else Path.home() / ".config"
            config_dir = base_path / cls.ORG_NAME / cls.APP_NAME
        else:
            project_root = Path(__file__).resolve().parent
            config_dir = project_root / "config"

        cls._settings_path = config_dir / cls.FILE_NAME
        return cls._settings_path

    @classmethod
    def get_settings(cls) -> QSettings:
        if cls._settings is not None:
            return cls._settings

        settings_path = cls.get_settings_path()
        try:
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            cls._last_error = ""
            cls._settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        except OSError as error:
            cls._last_error = f"创建配置目录失败: {error}"
            fallback_path = Path.cwd() / cls.FILE_NAME
            cls._settings = QSettings(str(fallback_path), QSettings.Format.IniFormat)

        return cls._settings

    @classmethod
    def get_last_error(cls) -> str:
        return cls._last_error

    @classmethod
    def value(cls, key: str, default: Any = None, value_type: type = str) -> Any:
        return cls.get_settings().value(key, default, type=value_type)

    @classmethod
    def set_value(cls, key: str, value: Any, sync: bool = False):
        settings = cls.get_settings()
        settings.setValue(key, value)
        if sync:
            settings.sync()

    @classmethod
    def sync(cls):
        cls.get_settings().sync()
