"""
Plotting and visualization: A-Mode (waveforms) and M-Mode (waterfall images).
"""
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QGridLayout
from qfluentwidgets import BodyLabel, SwitchButton

from utils.SettingService import SettingService


class PlotPanel(QWidget):
    preprocessToggled = pyqtSignal(bool)

    PREPROCESS_ENABLED_KEY = "Plot/RealtimePreprocessEnabled"

    def __init__(self, parent=None, m_mode_width: int = 512):
        super().__init__(parent)
        self._a_mode_x_min = 0.0
        self._a_mode_x_max = 600.0
        self._mModeSeconds = 3.0  # M 模式显示最近 3 秒的数据
        self._scanFreqHz = 20
        self._mModeDefaultDepth = 128
        self._plots: List[pg.PlotWidget] = []
        self._curves = []
        self._raw_curves = []
        self._m_mode_images = []
        self._m_mode_buffers: List[np.ndarray] = []
        self._m_mode_indices = [0] * 4  # 环形缓冲区写入位置
        self._m_mode_width = max(3, int(self._scanFreqHz * self._mModeSeconds))  # 根据扫描频率和显示时间计算宽度
        self._m_mode_plots = []
        self._current_sample_depths = [0] * 4
        self._m_mode_levels = [(0.0, 1024.0) for _ in range(4)] # M 超声的初始范围设置为 0-1024，后续根据数据动态调整
        self._mModeColorMaps = [
            self._buildColorMap(
                [0.0, 0.35, 0.7, 1.0],
                [[0, 0, 0, 255], [0, 70, 0, 255], [40, 170, 40, 255], [180, 255, 180, 255]],
            ),
            self._buildColorMap(
                [0.0, 0.35, 0.7, 1.0],
                [[0, 0, 0, 255], [0, 40, 90, 255], [0, 120, 200, 255], [170, 235, 255, 255]],
            ),
            self._buildColorMap(
                [0.0, 0.35, 0.7, 1.0],
                [[0, 0, 0, 255], [90, 35, 0, 255], [220, 110, 0, 255], [255, 220, 160, 255]],
            ),
            self._buildColorMap(
                [0.0, 0.35, 0.7, 1.0],
                [[0, 0, 0, 255], [90, 0, 0, 255], [210, 40, 40, 255], [255, 210, 210, 255]],
            ),
        ]
        self._a_mode_zero_lines = []
        self._a_mode_y0_lines = []
        self._a_mode_y_end_lines = []
        self._realtime_preprocess_enabled = False
        self._mModeTimeTicks = [[(0.0, "3s前"), (1.0, "2s前"), (2.0, "1s前"), (3.0, "0s当前")]]

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(8)

        preprocess_row = QHBoxLayout()
        preprocess_row.setContentsMargins(0, 0, 0, 0)
        preprocess_row.setSpacing(8)
        self._preprocessLabel = BodyLabel("实时预处理")
        self._preprocessSwitch = SwitchButton(self)
        self._loadRealtimePreprocessState()

        self._preprocessSwitch.checkedChanged.connect(self._onPreprocessSwitchChanged)
        preprocess_row.addWidget(self._preprocessLabel)
        preprocess_row.addWidget(self._preprocessSwitch)
        preprocess_row.addStretch()
        root_layout.addLayout(preprocess_row)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setSpacing(8)
        root_layout.addLayout(grid_layout, 1)

        channel_colors = ["#4e9a06", "#3465a4", "#c17d11", "#cc0000"]
        grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for idx in range(4):
            channelContainer = QWidget()
            channelContainerLayout = QVBoxLayout(channelContainer)
            channelContainerLayout.setContentsMargins(0, 0, 0, 0)
            channelContainerLayout.setSpacing(2)

            channelTitleLabel = QLabel(f"Channel {idx + 1}")
            channelTitleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            channelTitleLabel.setStyleSheet("color: #000000; font-size: 11pt;")
            channelContainerLayout.addWidget(
                channelTitleLabel,
                0,
                Qt.AlignmentFlag.AlignCenter,
            )

            channelContentWidget = QWidget()
            channelLayout = QHBoxLayout(channelContentWidget)
            channelLayout.setContentsMargins(0, 0, 0, 0)
            channelLayout.setSpacing(4)

            mModeWidget = pg.PlotWidget()
            mModeWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            mModeWidget.setMouseEnabled(x=False, y=False)
            mModeWidget.showGrid(x=True, y=True, alpha=0.15)
            mModeWidget.setLabel("bottom", "Time (s)", size="8pt")
            mModeWidget.setLabel("left", "Depth", size="8pt")
            mModeWidget.getViewBox().invertY(True)

            mModePlotItem = mModeWidget.getPlotItem()
            if mModePlotItem is not None:
                mModePlotItem.getAxis("bottom").setPen(pg.mkPen("#888888"))
                mModePlotItem.getAxis("left").setPen(pg.mkPen("#888888"))
                mModePlotItem.getAxis("bottom").setTicks(self._mModeTimeTicks)
                mModePlotItem.getAxis("left").setTicks([[(0, "0"), (self._mModeDefaultDepth, str(self._mModeDefaultDepth))]])

            mModeImage = pg.ImageItem()
            mModeImage.setLookupTable(self._mModeColorMaps[idx].getLookupTable(0.0, 1.0, 256))
            mModeWidget.addItem(mModeImage)

            mModeBuffer = np.zeros((self._mModeDefaultDepth, self._m_mode_width), dtype=np.float32)
            mModeImage.setImage(mModeBuffer.T, autoLevels=False, levels=self._m_mode_levels[idx])
            mModeImage.setRect(0.0, 0.0, self._mModeSeconds, self._mModeDefaultDepth)

            self._m_mode_plots.append(mModeWidget)
            self._m_mode_images.append(mModeImage)
            self._m_mode_buffers.append(mModeBuffer)

            plotWidget = pg.PlotWidget()
            plotWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            plotWidget.setMaximumWidth(200)
            plotWidget.setMinimumWidth(10)
            # plotWidget.setTitle(f"Channel {idx + 1}", color="#000000", size="11pt")
            plotWidget.showGrid(x=True, y=True, alpha=0.2)
            plotWidget.setLabel("bottom", "ADC Value", size="8pt")
            # plotWidget.setLabel("left", "Depth", size="8pt")
            plotWidget.getViewBox().invertY(True)

            plotItem = plotWidget.getPlotItem()
            if plotItem is not None:
                plotItem.getAxis("bottom").setPen(pg.mkPen("#888888"))
                plotItem.getAxis("left").setPen(pg.mkPen("#888888"))
                plotItem.getAxis("bottom").setTicks([[(100, "100"), (300, "300"), (500, "500")]])
                plotItem.getAxis("left").setTicks([[(0, "0"), (self._mModeDefaultDepth, str(self._mModeDefaultDepth))]])

            plotWidget.enableAutoRange(axis="x", enable=False)
            plotWidget.enableAutoRange(axis="y", enable=False)
            plotWidget.getViewBox().setDefaultPadding(0.0)
            plotWidget.setXRange(self._a_mode_x_min, self._a_mode_x_max)
            plotWidget.setYRange(0, self._mModeDefaultDepth)
            plotWidget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

            zeroLine = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen("#aaaaaa", width=1),
                movable=False,
                label="0",
                labelOpts={"position": 0.02, "color": "#666666"},
            )
            plotWidget.addItem(zeroLine)
            self._a_mode_zero_lines.append(zeroLine)

            y0_line = pg.InfiniteLine(
                pos=0,
                angle=0,
                pen=pg.mkPen("#bbbbbb", width=1),
                movable=False,
                label="0",
                labelOpts={"position": 0.95, "color": "#666666"},
            )
            plotWidget.addItem(y0_line)
            self._a_mode_y0_lines.append(y0_line)

            y_end_line = pg.InfiniteLine(
                pos=127,
                angle=0,
                pen=pg.mkPen("#999999", width=1),
                movable=False,
                label="127",
                labelOpts={"position": 0.95, "color": "#666666"},
            )
            plotWidget.addItem(y_end_line)
            self._a_mode_y_end_lines.append(y_end_line)

            curve = plotWidget.plot(pen=pg.mkPen(channel_colors[idx], width=1.5))
            raw_curve = plotWidget.plot(pen=pg.mkPen(channel_colors[idx], width=1.0, style=Qt.PenStyle.DashLine))
            raw_curve.setVisible(self._realtime_preprocess_enabled)

            channelLayout.addWidget(mModeWidget, 1)
            channelLayout.addWidget(plotWidget, 0)
            channelContainerLayout.addWidget(channelContentWidget, 1)
            self._plots.append(plotWidget)
            self._curves.append(curve)
            self._raw_curves.append(raw_curve)

            row, col = grid_positions[idx]
            grid_layout.addWidget(channelContainer, row, col)

        self._current_sample_depths = [self._mModeDefaultDepth] * 4
        

    def _onPreprocessSwitchChanged(self, enabled: bool):
        SettingService.set_value(self.PREPROCESS_ENABLED_KEY, bool(enabled), sync=True)
        self.setRealtimePreprocessEnabled(enabled, emit_signal=True)

    def _loadRealtimePreprocessState(self):
        enabled = SettingService.value(self.PREPROCESS_ENABLED_KEY, False, bool)
        self.setRealtimePreprocessEnabled(bool(enabled), emit_signal=False)

    def isRealtimePreprocessEnabled(self) -> bool:
        return self._realtime_preprocess_enabled

    def setRealtimePreprocessEnabled(self, enabled: bool, emit_signal: bool = False):
        '''
        切换是否启用实时预处理
        '''
        enabled = bool(enabled)
        if self._realtime_preprocess_enabled == enabled:
            return

        self._realtime_preprocess_enabled = enabled
        self._preprocessSwitch.blockSignals(True)
        self._preprocessSwitch.setChecked(enabled)
        self._preprocessSwitch.blockSignals(False)

        for raw_curve in self._raw_curves:
            raw_curve.setVisible(enabled)
            if not enabled:
                raw_curve.setData([], [])

        if emit_signal:
            self.preprocessToggled.emit(enabled)

    def updateRawOverlay(self, frames: List[List[float]]):
        if not self._realtime_preprocess_enabled:
            return
        if len(frames) != 4:
            return

        for idx, data in enumerate(frames):
            values = np.asarray(data, dtype=np.float32).reshape(-1)
            if values.size == 0:
                self._raw_curves[idx].setData([], [])
                continue
            sample_indices = list(range(int(values.shape[0])))
            self._raw_curves[idx].setData(values, sample_indices)

    def _buildColorMap(self, positions: List[float], colors: List[List[int]]) -> pg.ColorMap:
        positionArray = np.asarray(positions, dtype=np.float32)
        colorArray = np.asarray(colors, dtype=np.ubyte)
        return pg.ColorMap(positionArray, colorArray)

    def _setMModePlotRange(self, channelIdx: int, depth: int):
        mModePlot = self._m_mode_plots[channelIdx]
        plotItem = mModePlot.getPlotItem()
        if plotItem is not None:
            plotItem.getAxis("bottom").setTicks(self._mModeTimeTicks)
            plotItem.getAxis("left").setTicks([[(0, "0"), (depth, str(depth))]])
        mModePlot.setXRange(0.0, self._mModeSeconds)
        mModePlot.setYRange(0, depth)

    def _resetMModeBuffers(self, depth: int):
        depth = max(1, int(depth))
        self._current_sample_depths = [depth] * 4
        for idx in range(4):
            mModeBuffer = np.zeros((depth, self._m_mode_width), dtype=np.float32)
            self._m_mode_buffers[idx] = mModeBuffer
            self._m_mode_indices[idx] = 0
            self._m_mode_images[idx].setLookupTable(self._mModeColorMaps[idx].getLookupTable(0.0, 1.0, 256))
            self._m_mode_images[idx].setImage(mModeBuffer, autoLevels=False, levels=self._m_mode_levels[idx])
            self._m_mode_images[idx].setRect(0.0, 0.0, self._mModeSeconds, depth)
            self._setMModePlotRange(idx, depth)

    def onConfigReady(self, configDict: dict):
        scanFreqHz = int(configDict.get("scan_frequency_hz", self._scanFreqHz) or self._scanFreqHz)
        sampleDepth = int(configDict.get("sample_depth", self._mModeDefaultDepth) or self._mModeDefaultDepth)

        scanFreqHz = max(1, scanFreqHz)
        sampleDepth = max(1, sampleDepth)

        width = max(3, int(scanFreqHz * self._mModeSeconds))
        widthChanged = width != self._m_mode_width
        depthChanged = sampleDepth != self._current_sample_depths[0]

        self._scanFreqHz = scanFreqHz
        self._m_mode_width = width

        if widthChanged or depthChanged:
            self._resetMModeBuffers(sampleDepth)

    def _resize_m_mode_buffer(self, channel_idx: int, new_depth: int):
        """Dynamically adjust M-Mode buffer depth."""
        old_buffer = self._m_mode_buffers[channel_idx]
        old_depth = old_buffer.shape[0]

        if new_depth == old_depth:
            return

        new_buffer = np.zeros((new_depth, self._m_mode_width), dtype=np.float32)
        copy_depth = min(old_depth, new_depth)
        new_buffer[:copy_depth, :] = old_buffer[:copy_depth, :]

        self._m_mode_buffers[channel_idx] = new_buffer
        self._current_sample_depths[channel_idx] = new_depth
        self._m_mode_images[channel_idx].setImage(new_buffer.T, autoLevels=False, levels=self._m_mode_levels[channel_idx])
        self._m_mode_images[channel_idx].setRect(0.0, 0.0, self._mModeSeconds, new_depth)
        self._setMModePlotRange(channel_idx, new_depth)

    def _update_m_mode_levels(self, channel_idx: int):
        """Update color range based on robust percentiles to avoid saturation."""
        buffer = self._m_mode_buffers[channel_idx]
        values = buffer.reshape(-1)

        non_zero = values[values != 0]
        if non_zero.size < 64:
            return

        low = float(np.percentile(non_zero, 5))
        high = float(np.percentile(non_zero, 95))
        if high - low < 1e-6:
            return

        prev_low, prev_high = self._m_mode_levels[channel_idx]
        alpha = 0.2
        new_low = (1 - alpha) * prev_low + alpha * low
        new_high = (1 - alpha) * prev_high + alpha * high

        if new_high - new_low < 1.0:
            center = 0.5 * (new_low + new_high)
            new_low, new_high = center - 0.5, center + 0.5

        self._m_mode_levels[channel_idx] = (new_low, new_high)

    def updatePlots(self, frames: List[List[float]]):
        self._updatePlots(frames, direction=1)

    def updatePlaybackFrames(self, frames: List[List[float]], direction: int = 1):
        self._updatePlots(frames, direction=direction)

    def _updatePlots(self, frames: List[List[float]], direction: int = 1):
        if len(frames) != 4:
            return

        global_sample_depth = 1
        abs_frames: List[Optional[np.ndarray]] = [None] * 4

        for idx, data in enumerate(frames):
            values = np.asarray(data, dtype=np.float32).reshape(-1)
            if values.size > 0:
                abs_frames[idx] = values
                global_sample_depth = max(global_sample_depth, int(values.shape[0]))
                # self._plots[idx].setTitle(
                #     f"Channel {idx + 1}",
                #     color="#000000",
                #     size="10pt",
                # )
            # else:
                # self._plots[idx].setTitle(
                #     f"Channel {idx + 1} | N/A",
                #     color="#000000",
                #     size="10pt",
                # )

        for idx, abs_data in enumerate(abs_frames):
            y_depth = max(1, global_sample_depth)
            plotItem = self._plots[idx].getPlotItem()
            if plotItem is not None:
                plotItem.getAxis("bottom").setTicks([[(100, "100"), (300, "300"), (500, "500")]])
                plotItem.getAxis("left").setTicks([[(0, "0"), (y_depth, str(y_depth))]])

            if abs_data is None:
                self._plots[idx].setXRange(self._a_mode_x_min, self._a_mode_x_max)
                self._plots[idx].setYRange(0, y_depth)
                self._a_mode_y0_lines[idx].setPos(0)
                self._a_mode_y_end_lines[idx].setPos(y_depth)
                self._a_mode_y_end_lines[idx].label.setFormat(str(y_depth))
                continue

            data_len = int(abs_data.shape[0])
            sample_indices = list(range(data_len))
            self._curves[idx].setData(abs_data, sample_indices)

            self._plots[idx].setXRange(self._a_mode_x_min, self._a_mode_x_max)
            self._plots[idx].setYRange(0, y_depth)

            self._a_mode_y0_lines[idx].setPos(0)
            self._a_mode_y_end_lines[idx].setPos(y_depth)
            self._a_mode_y_end_lines[idx].label.setFormat(str(y_depth))

            if self._current_sample_depths[idx] != data_len:
                self._resize_m_mode_buffer(idx, data_len)

            mModeBuffer = self._m_mode_buffers[idx]
            if mModeBuffer.shape[1] != self._m_mode_width:
                self._resetMModeBuffers(data_len)
                mModeBuffer = self._m_mode_buffers[idx]

            if data_len > 0:
                if data_len != mModeBuffer.shape[0]:
                    continue

                if direction >= 0:
                    mModeBuffer = np.roll(mModeBuffer, -1, axis=1)
                    mModeBuffer[:, -1] = abs_data
                else:
                    mModeBuffer = np.roll(mModeBuffer, 1, axis=1)
                    mModeBuffer[:, 0] = abs_data
                self._m_mode_buffers[idx] = mModeBuffer
                self._m_mode_images[idx].setImage(mModeBuffer.T, autoLevels=False, levels=self._m_mode_levels[idx])
                self._m_mode_images[idx].setRect(0.0, 0.0, self._mModeSeconds, data_len)
                self._setMModePlotRange(idx, data_len)
                self._update_m_mode_levels(idx)
