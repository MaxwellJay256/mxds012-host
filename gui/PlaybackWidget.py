"""Playback page for loading and replaying local TXT frames."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, FluentIcon, HeaderCardWidget, LineEdit, PushButton, Slider, SpinBox, ToolButton

from gui.PreprocessParamCard import PreprocessParamCard
from utils.PlaybackFileUtils import parsePlaybackTxtFile
from utils.PreProcess import RealtimePreprocessor
from utils.PreprocessParamService import PreprocessParamService
from utils.SettingService import SettingService


class ClickableLineEdit(LineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, a0):
        self.clicked.emit()
        super().mousePressEvent(a0)


class FileCard(HeaderCardWidget):
    def __init__(self, parent=None):  # type: ignore[override]
        super().__init__(parent)
        self.setTitle("文件读取")

        self.vBoxLayout = QVBoxLayout()
        file_row = QHBoxLayout()
        file_row.setSpacing(10)

        self.fileEdit = ClickableLineEdit()
        self.fileEdit.setPlaceholderText("点击选择 txt 文件")
        self.fileEdit.setReadOnly(True)
        file_row.addWidget(self.fileEdit, 1)

        self.browseButton = ToolButton(FluentIcon.FOLDER)
        file_row.addWidget(self.browseButton)

        self.vBoxLayout.addLayout(file_row)
        self.viewLayout.addLayout(self.vBoxLayout)


class PlaybackCard(HeaderCardWidget):
    def __init__(self, parent=None):  # type: ignore[override]
        super().__init__(parent)
        self.setTitle("回放")

        self.vBoxLayout = QVBoxLayout()
        self.vBoxLayout.setSpacing(10)

        param_row = QHBoxLayout()
        param_row.setSpacing(10)

        self.fpsLabel = BodyLabel("回放帧率 (fps)")
        param_row.addWidget(self.fpsLabel)

        self.fpsSpin = SpinBox()
        self.fpsSpin.setRange(1, 60)
        self.fpsSpin.setValue(20)
        self.fpsSpin.setFixedWidth(120)
        param_row.addWidget(self.fpsSpin)

        self.depthLabel = BodyLabel("采样深度")
        param_row.addWidget(self.depthLabel)

        self.sampleDepthEdit = LineEdit()
        self.sampleDepthEdit.setReadOnly(True)
        self.sampleDepthEdit.setPlaceholderText("-")
        self.sampleDepthEdit.setFixedWidth(140)
        param_row.addWidget(self.sampleDepthEdit)

        param_row.addStretch()
        self.vBoxLayout.addLayout(param_row)

        playback_row = QHBoxLayout()
        playback_row.setSpacing(10)

        self.playPauseButton = PushButton("播放")
        playback_row.addWidget(self.playPauseButton)

        self.playSlider = Slider(Qt.Orientation.Horizontal)
        self.playSlider.setRange(0, 0)
        self.playSlider.setValue(0)
        playback_row.addWidget(self.playSlider, 1)

        self.frameLabel = BodyLabel("0/0")
        self.frameLabel.setMinimumWidth(90)
        playback_row.addWidget(self.frameLabel)

        self.vBoxLayout.addLayout(playback_row)
        self.viewLayout.addLayout(self.vBoxLayout)


class PlaybackWidget(QWidget):
    PLAYBACK_FILE_KEY = "Playback/FilePath"

    frameReady = pyqtSignal(list, int)
    rawFrameReady = pyqtSignal(list)
    configReady = pyqtSignal(dict)
    infoRequested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._frames: List[List[List[int]]] = []
        self._current_frame_index = 0
        self._play_direction = 1
        self._is_playing = False
        self._is_slider_dragging = False
        self._was_playing_before_drag = False
        self._syncing_slider = False
        self._sample_depth = 0
        self._fps = 20
        self._realtime_preprocess_enabled = False
        self._preprocessor: Optional[RealtimePreprocessor] = None

        self._param_service = PreprocessParamService.get_instance()
        self._param_service.parametersChanged.connect(self._onPreprocessParamsChanged)

        rootLayout = QVBoxLayout(self)
        rootLayout.setContentsMargins(12, 12, 12, 12)
        rootLayout.setSpacing(6)

        self.fileCard = FileCard(self)
        rootLayout.addWidget(self.fileCard)

        self.playbackCard = PlaybackCard(self)
        rootLayout.addWidget(self.playbackCard)

        self.preprocessParamCard = PreprocessParamCard(self)
        rootLayout.addWidget(self.preprocessParamCard, 1)

        self.playbackTimer = QTimer(self)
        self.playbackTimer.timeout.connect(self.onPlaybackTick)

        self.fileCard.fileEdit.clicked.connect(self.onSelectFile)
        self.fileCard.browseButton.clicked.connect(self.onSelectFile)
        self.playbackCard.playPauseButton.clicked.connect(self.onPlayPauseClicked)
        self.playbackCard.fpsSpin.valueChanged.connect(self.onFpsChanged)
        self.playbackCard.playSlider.sliderPressed.connect(self.onSliderPressed)
        self.playbackCard.playSlider.sliderReleased.connect(self.onSliderReleased)
        self.playbackCard.playSlider.valueChanged.connect(self.onSliderValueChanged)

        self.onFpsChanged(self.playbackCard.fpsSpin.value())
        self._loadLastPlaybackFile()

    def _loadLastPlaybackFile(self):
        saved_path = SettingService.value(self.PLAYBACK_FILE_KEY, "", str)
        if not saved_path:
            return

        file_path = Path(saved_path)
        if file_path.exists() and file_path.is_file() and file_path.suffix.lower() == ".txt":
            self.loadPlaybackFile(str(file_path))

    def _onPreprocessParamsChanged(self, _params):
        if self._realtime_preprocess_enabled:
            self._preprocessor = RealtimePreprocessor(self._param_service.get_params_copy())

    def onRealtimePreprocessSwitchChanged(self, enabled: bool):
        self._realtime_preprocess_enabled = bool(enabled)
        if self._realtime_preprocess_enabled:
            self._preprocessor = RealtimePreprocessor(self._param_service.get_params_copy())
        else:
            self._preprocessor = None

    def onSelectFile(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择回放文件",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not file_path:
            return

        self.loadPlaybackFile(file_path)

    def loadPlaybackFile(self, file_path: str):
        try:
            frames, sample_depth = parsePlaybackTxtFile(file_path)
        except Exception as error:
            self.infoRequested.emit(f"加载回放文件失败: {error}", "error")
            return

        self._frames = frames
        self._sample_depth = sample_depth
        self._current_frame_index = 0
        self._play_direction = 1
        self._is_playing = False
        self.playbackTimer.stop()
        self.playbackCard.playPauseButton.setText("播放")
        self.fileCard.fileEdit.setText(file_path)
        self.playbackCard.sampleDepthEdit.setText(str(sample_depth))
        self.playbackCard.playSlider.setRange(0, max(0, len(self._frames) - 1))
        self._syncing_slider = True
        self.playbackCard.playSlider.setValue(0)
        self._syncing_slider = False
        self._updateFrameLabel()
        self._emitConfig()
        SettingService.set_value(self.PLAYBACK_FILE_KEY, str(Path(file_path)), sync=True)

    def onFpsChanged(self, value: int):
        self._fps = max(1, min(60, int(value)))
        interval = max(1, int(round(1000.0 / float(self._fps))))
        self.playbackTimer.setInterval(interval)
        self._emitConfig()

    def onPlayPauseClicked(self):
        if not self._frames:
            self.infoRequested.emit("请先选择回放文件", "warning")
            return

        if self._is_playing:
            self._stopPlayback()
        else:
            self._startPlayback()

    def onSliderPressed(self):
        if self._is_playing:
            self._was_playing_before_drag = True
            self.playbackTimer.stop()
        else:
            self._was_playing_before_drag = False
        self._is_slider_dragging = True

    def onSliderReleased(self):
        self._is_slider_dragging = False
        if self._was_playing_before_drag and self._frames:
            self._startPlayback()
        self._was_playing_before_drag = False

    def onSliderValueChanged(self, value: int):
        if self._syncing_slider or not self._frames:
            return

        new_index = max(0, min(int(value), len(self._frames) - 1))
        previous_index = self._current_frame_index
        if new_index != previous_index:
            self._play_direction = 1 if new_index >= previous_index else -1

        self._current_frame_index = new_index
        self._updateFrameLabel()

        if self._is_playing and self._is_slider_dragging:
            self.playbackTimer.stop()

    def onPlaybackTick(self):
        if not self._frames:
            self._stopPlayback()
            return

        if self._current_frame_index < 0 or self._current_frame_index >= len(self._frames):
            self._resetPlaybackToStart()
            return

        raw_frame = self._frames[self._current_frame_index]
        frame_to_plot = raw_frame

        if self._realtime_preprocess_enabled:
            self.rawFrameReady.emit(raw_frame)
            if self._preprocessor is None:
                self._preprocessor = RealtimePreprocessor(self._param_service.get_params_copy())
            try:
                frame_to_plot = self._preprocessor.process(raw_frame)
            except Exception:
                frame_to_plot = raw_frame

        self.frameReady.emit(frame_to_plot, self._play_direction)

        self._syncing_slider = True
        self.playbackCard.playSlider.setValue(self._current_frame_index)
        self._syncing_slider = False
        self._updateFrameLabel()

        next_index = self._current_frame_index + self._play_direction
        if next_index < 0 or next_index >= len(self._frames):
            self._resetPlaybackToStart()
            return

        self._current_frame_index = next_index

    def _startPlayback(self):
        if not self._frames:
            return
        # 用户点击“播放”后始终按正向播放。
        self._play_direction = 1
        self._is_playing = True
        self.playbackCard.playPauseButton.setText("暂停")
        self.playbackTimer.start()

    def _stopPlayback(self):
        self._is_playing = False
        self.playbackTimer.stop()
        self.playbackCard.playPauseButton.setText("播放")

    def _resetPlaybackToStart(self):
        self._stopPlayback()
        self._current_frame_index = 0
        self._syncing_slider = True
        self.playbackCard.playSlider.setValue(0)
        self._syncing_slider = False
        self._updateFrameLabel()

    def _updateFrameLabel(self):
        total = len(self._frames)
        if total <= 0:
            self.playbackCard.frameLabel.setText("0/0")
            return
        current = self._current_frame_index + 1
        if not self._is_playing and self._current_frame_index == 0 and self.playbackCard.playSlider.value() == 0:
            current = 0
        self.playbackCard.frameLabel.setText(f"{current}/{total}")

    def _emitConfig(self):
        if not self._frames:
            return
        self.configReady.emit({
            "scan_frequency_hz": int(self._fps),
            "sample_depth": int(self._sample_depth),
        })
