"""
Plotting and visualization: A-Mode (waveforms) and M-Mode (waterfall images).
"""
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QGridLayout


class PlotPanel(QWidget):
    def __init__(self, parent=None, m_mode_width: int = 512):
        super().__init__(parent)
        self._a_mode_y_min = -497.0
        self._a_mode_y_max = 527.0
        self._plots: List[pg.PlotWidget] = []
        self._curves = []
        self._m_mode_images = []  # M-Mode image displays
        self._m_mode_buffers: List[np.ndarray] = []  # Ring buffers
        self._m_mode_indices = [0] * 4  # Write position per channel
        self._m_mode_width = m_mode_width  # M-Mode image width (time frames)
        self._m_mode_plots = []  # M-Mode plot references
        self._current_sample_depths = [0] * 4  # Current sample depth per channel
        self._m_mode_levels = [(-2000.0, 2000.0) for _ in range(4)]  # Display range per channel
        self._a_mode_zero_lines = []  # A-Mode y=0 reference lines
        self._a_mode_x0_lines = []  # A-Mode x=0 reference lines
        self._a_mode_x_end_lines = []  # A-Mode rightmost sample reference lines

        # Grid layout: 2x2
        grid_layout = QGridLayout(self)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setSpacing(8)

        channel_colors = ["#4e9a06", "#3465a4", "#c17d11", "#cc0000"]
        grid_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # top-left, top-right, bottom-left, bottom-right

        for idx in range(4):
            # Create container for each channel
            channel_container = QWidget()
            channel_layout = QVBoxLayout(channel_container)
            channel_layout.setContentsMargins(0, 0, 0, 0)
            channel_layout.setSpacing(4)

            # A-Mode (raw waveform line plot)
            plot_widget = pg.PlotWidget()
            plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            plot_widget.setTitle(f"Channel {idx + 1} A-Mode", color="#000000", size="11pt")
            plot_widget.showGrid(x=True, y=True, alpha=0.2)
            plot_widget.setLabel("bottom", "Sample Index", size="9pt")
            plot_widget.setLabel("left", "ADC Result", size="9pt")
            plot_item = plot_widget.getPlotItem()
            plot_item.getAxis('bottom').setPen(pg.mkPen('#888888'))
            plot_item.getAxis('left').setPen(pg.mkPen('#888888'))

            # Disable auto Y scaling; we'll set symmetric range around 0
            plot_widget.enableAutoRange(axis='y', enable=False)
            plot_widget.getViewBox().setDefaultPadding(0.0)
            plot_widget.setYRange(self._a_mode_y_min, self._a_mode_y_max)

            # Y=0 baseline
            zero_line = pg.InfiniteLine(
                pos=0,
                angle=0,
                pen=pg.mkPen('#aaaaaa', width=1),
                movable=False,
                label='0',
                labelOpts={'position': 0.02, 'color': '#666666'}
            )
            plot_widget.addItem(zero_line)
            self._a_mode_zero_lines.append(zero_line)

            # X=0 reference
            x0_line = pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen('#bbbbbb', width=1),
                movable=False,
                label='0',
                labelOpts={'position': 0.95, 'color': '#666666'}
            )
            plot_widget.addItem(x0_line)
            self._a_mode_x0_lines.append(x0_line)

            # Rightmost sample reference
            x_end_line = pg.InfiniteLine(
                pos=127,
                angle=90,
                pen=pg.mkPen('#999999', width=1),
                movable=False,
                label='127',
                labelOpts={'position': 0.95, 'color': '#666666'}
            )
            plot_widget.addItem(x_end_line)
            self._a_mode_x_end_lines.append(x_end_line)
            curve = plot_widget.plot(pen=pg.mkPen(channel_colors[idx], width=1.5))

            channel_layout.addWidget(plot_widget)
            self._plots.append(plot_widget)
            self._curves.append(curve)

            # # M-Mode (time-scrolling 2D image)
            # m_mode_plot = pg.PlotWidget()
            # m_mode_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # m_mode_plot.setTitle(f"Channel {idx + 1} M-Mode", color="#000000", size="11pt")
            # m_mode_plot.setLabel("bottom", "Time Frame", size="9pt")
            # m_mode_plot.setLabel("left", "Sample Index", size="9pt")

            # # Initialize ring buffer: (sample_depth, time_frames)
            # m_mode_buffer = np.zeros((128, m_mode_width), dtype=np.float32)
            # self._m_mode_buffers.append(m_mode_buffer)
            # self._current_sample_depths[idx] = 128

            # # Create image item
            # image_item = pg.ImageItem(image=m_mode_buffer)
            # # Color map (viridis)
            # colormap = pg.colormap.get('viridis')
            # lut = colormap.getLookupTable(start=0, stop=255)
            # image_item.setLookupTable(lut, update=True)
            # image_item.setLevels(self._m_mode_levels[idx])
            # m_mode_plot.addItem(image_item)

            # channel_layout.addWidget(m_mode_plot)
            # self._m_mode_images.append(image_item)
            # self._m_mode_plots.append(m_mode_plot)

            # Add to grid
            row, col = grid_positions[idx]
            grid_layout.addWidget(channel_container, row, col)

    def _resize_m_mode_buffer(self, channel_idx: int, new_depth: int):
        """Dynamically adjust M-Mode buffer depth."""
        old_buffer = self._m_mode_buffers[channel_idx]
        old_depth = old_buffer.shape[0]
        
        if new_depth == old_depth:
            return
        
        # Create new buffer
        new_buffer = np.zeros((new_depth, self._m_mode_width), dtype=np.float32)
        
        # Copy old data to smaller range
        copy_depth = min(old_depth, new_depth)
        new_buffer[:copy_depth, :] = old_buffer[:copy_depth, :]
        
        # Update
        self._m_mode_buffers[channel_idx] = new_buffer
        self._current_sample_depths[channel_idx] = new_depth
        self._m_mode_images[channel_idx].setImage(new_buffer, autoLevels=False, levels=self._m_mode_levels[channel_idx])

    def _update_m_mode_levels(self, channel_idx: int):
        """Update color range based on robust percentiles to avoid saturation."""
        buffer = self._m_mode_buffers[channel_idx]
        values = buffer.reshape(-1)

        # Ignore zeros from initialization
        non_zero = values[values != 0]
        if non_zero.size < 64:
            return

        low = float(np.percentile(non_zero, 5))
        high = float(np.percentile(non_zero, 95))
        if high - low < 1e-6:
            return

        prev_low, prev_high = self._m_mode_levels[channel_idx]
        alpha = 0.2  # Smoothing to reduce flicker
        new_low = (1 - alpha) * prev_low + alpha * low
        new_high = (1 - alpha) * prev_high + alpha * high

        if new_high - new_low < 1.0:
            center = 0.5 * (new_low + new_high)
            new_low, new_high = center - 0.5, center + 0.5

        self._m_mode_levels[channel_idx] = (new_low, new_high)

    def update_plots(self, frames: List[List[float]]):
        if len(frames) != 4:
            return

        # Calculate global scaling
        global_peak = 1.0
        global_right_edge = 1
        centered_frames: List[Optional[np.ndarray]] = [None] * 4
        channel_stats = []
        for idx, data in enumerate(frames):
            if data:
                raw_values = np.asarray(-np.array(data), dtype=np.float32)
                mean_value = float(np.mean(raw_values))
                centered_values = raw_values - mean_value
                centered_frames[idx] = centered_values

                max_value = float(np.max(raw_values))
                min_value = float(np.min(raw_values))
                diff_value = max_value - min_value
                local_peak = max(
                    abs(float(np.min(centered_values))),
                    abs(float(np.max(centered_values))),
                    1.0
                )
                global_peak = max(global_peak, local_peak)
                global_right_edge = max(global_right_edge, max(0, raw_values.size - 1))
                channel_stats.append(
                    f"CH{idx + 1}"
                    # f"CH{idx + 1}(mean={mean_value:.2f}, max={max_value:.2f}, min={min_value:.2f}, diff={diff_value:.2f})"
                )
                self._plots[idx].setTitle(
                    f"Channel {idx + 1} A-Mode",
                    # f"Channel {idx + 1} A-Mode | mean={mean_value:.2f}, max={max_value:.2f}, min={min_value:.2f}, diff={diff_value:.2f}",
                    color="#000000",
                    size="10pt"
                )
            else:
                channel_stats.append(f"CH{idx + 1}=N/A")
                self._plots[idx].setTitle(
                    f"Channel {idx + 1} A-Mode | N/A",
                    color="#000000",
                    size="10pt"
                )

        # Keep A-Mode Y-axis fixed for all channels
        global_right_edge = max(1, global_right_edge)

        for idx, centered_data in enumerate(centered_frames):
            if centered_data is None:
                self._plots[idx].setYRange(self._a_mode_y_min, self._a_mode_y_max)
                self._plots[idx].setXRange(0, global_right_edge)
                self._a_mode_x0_lines[idx].setPos(0)
                self._a_mode_x_end_lines[idx].setPos(0)
                self._a_mode_x_end_lines[idx].label.setFormat("0")
                continue
            
            data_len = int(centered_data.shape[0])
            
            # Update A-Mode
            x_values = list(range(data_len))
            self._curves[idx].setData(x_values, centered_data)

            self._plots[idx].setYRange(self._a_mode_y_min, self._a_mode_y_max)

            # X-axis markers
            farthest_x = max(0, data_len - 1)
            self._a_mode_x0_lines[idx].setPos(0)
            self._a_mode_x_end_lines[idx].setPos(farthest_x)
            self._a_mode_x_end_lines[idx].label.setFormat(str(farthest_x))

            self._plots[idx].setXRange(0, global_right_edge)
            
            # # Check if M-Mode buffer depth needs adjustment
            # if data_len != self._current_sample_depths[idx]:
            #     self._resize_m_mode_buffer(idx, data_len)
            
            # # Update M-Mode: write new data to ring buffer
            # raw_data = frames[idx]
            # buffer = self._m_mode_buffers[idx]
            # write_pos = self._m_mode_indices[idx]
            
            # # Write current frame to ring buffer column
            # buffer[:data_len, write_pos] = raw_data[:data_len]
            
            # # Advance write position (circular)
            # self._m_mode_indices[idx] = (write_pos + 1) % self._m_mode_width

            # # Update color range
            # self._update_m_mode_levels(idx)
            
            # # Update display
            # self._m_mode_images[idx].setImage(buffer, autoLevels=False, levels=self._m_mode_levels[idx])
