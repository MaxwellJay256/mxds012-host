import sys
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

import pyqtgraph as pg

from serial import Serial, SerialException
from serial.tools import list_ports

from PyQt5.QtCore import QThread, pyqtSignal

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy

from qfluentwidgets import (
    ComboBox,
    PrimaryPushButton,
    PushButton,
    InfoBar,
    InfoBarPosition,
    setTheme,
    Theme,
    setThemeColor
)


class SerialReader(QThread):
    frameReady = pyqtSignal(list)
    statusChanged = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, port: str, baudrate: int = 115200, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self._running = False

    def run(self):
        try:
            serial_port = Serial(self.port, self.baudrate, timeout=0.2)
        except SerialException as exc:
            self.errorOccurred.emit(f"无法打开串口 {self.port}: {exc}")
            self.stopped.emit()
            return

        self.statusChanged.emit(f"已连接 {self.port}")
        self._running = True

        try:
            while self._running:
                try:
                    raw_line = serial_port.readline()
                except SerialException as exc:
                    self.errorOccurred.emit(f"串口读取错误: {exc}")
                    break

                if not raw_line:
                    continue

                frame_text = raw_line.decode("utf-8", errors="ignore").strip()
                if not frame_text:
                    continue

                processed = self._process_frame(frame_text)
                if processed is None:
                    continue
                self.frameReady.emit(processed)
        finally:
            serial_port.close()
            self._running = False
            self.statusChanged.emit("连接已关闭")
            self.stopped.emit()

    def stop(self):
        self._running = False

    def _process_frame(self, frame_text: str) -> Optional[List[List[float]]]:
        channels_raw = [segment.strip() for segment in frame_text.split(',') if segment.strip()]
        if len(channels_raw) < 5:
            self.errorOccurred.emit("数据格式错误: 通道数量不足")
            return None

        channels: List[List[float]] = []
        for idx, raw_channel in enumerate(channels_raw[:5]):
            try:
                values = [float(token) for token in raw_channel.split() if token]
            except ValueError:
                self.errorOccurred.emit(f"数据解析失败: 通道 {idx + 1} 包含非法字符")
                return None
            channels.append(values)

        min_len = min(len(ch) for ch in channels)
        if min_len == 0:
            self.errorOccurred.emit("数据解析失败: 采样结果为空")
            return None

        if any(len(ch) != min_len for ch in channels):
            self.statusChanged.emit("警告: 通道采样长度不一致, 已截取最短长度")

        reference = channels[4][:min_len]
        processed = []
        for source_idx in range(4):
            processed.append([
                channels[source_idx][sample_idx] - reference[sample_idx]
                for sample_idx in range(min_len)
            ])
        return processed


class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plots: List[pg.PlotWidget] = []
        self._curves = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        channel_colors = ["#4e9a06", "#3465a4", "#c17d11", "#cc0000"]

        for idx in range(4):
            plot_widget = pg.PlotWidget()
            plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            plot_widget.setTitle(f"Channel {idx + 1}", color="#000000", size="12pt")
            plot_widget.showGrid(x=True, y=True, alpha=0.2)
            plot_widget.setLabel("bottom", "Sample Index")
            plot_widget.setLabel("left", "ADC Result")
            plot_item = plot_widget.getPlotItem()
            plot_item.getAxis('bottom').setPen(pg.mkPen('#888888'))
            plot_item.getAxis('left').setPen(pg.mkPen('#888888'))

            # 固定 Y 轴范围，并关闭自动缩放
            plot_widget.setYRange(-800, 800)
            plot_widget.enableAutoRange(False)

            # 在 y=0 位置画一条基线，确保 0 在中间视觉上明确
            zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#aaaaaa', width=1))
            plot_widget.addItem(zero_line)
            curve = plot_widget.plot(pen=pg.mkPen(channel_colors[idx], width=1.5))

            layout.addWidget(plot_widget)
            self._plots.append(plot_widget)
            self._curves.append(curve)

    def update_plots(self, frames: List[List[float]]):
        if len(frames) != 4:
            return

        for idx, data in enumerate(frames):
            if not data:
                continue
            x_values = list(range(1, len(data) + 1))
            self._curves[idx].setData(x_values, data)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MXDS012 Host")
        self.resize(1100, 800)

        setTheme(Theme.LIGHT)
        setThemeColor('#0078d4')

        pg.setConfigOptions(antialias=True, background=None, foreground='#f0f0f0')

        self.serial_thread: Optional[SerialReader] = None

        # FluentWindow may already provide a layout; use it if present, otherwise create a new layout.
        existing_layout = self.layout()
        if existing_layout is None:
            self._central_layout = QVBoxLayout()
            self._central_layout.setContentsMargins(24, 24, 24, 24)
            self._central_layout.setSpacing(18)
            self.setLayout(self._central_layout)
        else:
            # create a container widget so we don't overwrite or interfere with the window's internal layout
            container = QWidget()
            self._central_layout = QVBoxLayout(container)
            self._central_layout.setContentsMargins(24, 24, 24, 24)
            self._central_layout.setSpacing(18)
            existing_layout.addWidget(container)

        self._build_controls()
        self.plot_panel = PlotPanel(self)
        self._central_layout.addWidget(self.plot_panel)

        self._refresh_ports()

    def _build_controls(self):
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(12)

        self.port_combo = ComboBox()
        self.port_combo.setMinimumWidth(200)
        control_layout.addWidget(self.port_combo)

        self.refresh_button = PushButton("刷新串口")
        self.refresh_button.clicked.connect(self._refresh_ports)
        control_layout.addWidget(self.refresh_button)

        self.start_button = PrimaryPushButton("开始接收")
        self.start_button.clicked.connect(self._start_serial)
        control_layout.addWidget(self.start_button)

        self.stop_button = PushButton("停止")
        self.stop_button.clicked.connect(self._stop_serial)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        control_layout.addStretch()

        self._central_layout.addWidget(control_bar)

    def _refresh_ports(self):
        # 保存当前选中的 device（而不是 display text），以便刷新后恢复
        previous_device = self.port_combo.currentData()
        self.port_combo.clear()

        ports = list_ports.comports()
        for port_info in ports:
            display_text = f"{port_info.device} - {port_info.description}"
            # 第二个参数作为 userData 储存真实 device 名称
            self.port_combo.addItem(display_text, port_info.device)

        if not ports:
            self._show_info("未检测到串口设备", InfoBarPosition.TOP, "warning")
        else:
            # 如果之前保存的是 device 名称，尝试通过 userData 恢复
            if previous_device:
                idx = self.port_combo.findData(previous_device)
                if idx >= 0:
                    self.port_combo.setCurrentIndex(idx)

    def _start_serial(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self._show_info("串口已在运行", InfoBarPosition.TOP, "warning")
            return

        if self.port_combo.count() == 0:
            self._show_info("没有可用串口", InfoBarPosition.TOP, "warning")
            return

        # 获取 userData（device）为首选方法；若为空，则从显示文本解析（例如 'COM3 - ...'）
        port = self.port_combo.currentData()
        if not port:
            # 尝试从显示文本解析设备名（空格前的 token，例如 'COM3 - 描述'）
            display = self.port_combo.currentText().strip()
            if display:
                # display 可能为 'COM3 - 描述'，取第一个 token
                port_candidate = display.split()[0]
                if port_candidate:
                    port = port_candidate

        if not port:
            self._show_info("请选择串口", InfoBarPosition.TOP, "warning")
            return

        # 显示解析出的端口，便于调试和确认
        self._show_info(f"已选择串口: {port}", InfoBarPosition.TOP, "info")

        self.serial_thread = SerialReader(port)
        self.serial_thread.frameReady.connect(self.plot_panel.update_plots)
        self.serial_thread.statusChanged.connect(lambda msg: self._show_info(msg, InfoBarPosition.TOP))
        self.serial_thread.errorOccurred.connect(lambda msg: self._show_info(msg, InfoBarPosition.BOTTOM, "error"))
        self.serial_thread.stopped.connect(self._on_serial_stopped)

        self.serial_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _stop_serial(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait(2000)
        else:
            self._on_serial_stopped()

    def _on_serial_stopped(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.serial_thread = None

    def _show_info(self, message: str, position: InfoBarPosition, level: str = "success"):
        builders = {
            "success": InfoBar.success,
            "info": InfoBar.info,
            "warning": InfoBar.warning,
            "error": InfoBar.error
        }
        builder = builders.get(level, InfoBar.info)
        builder(title="", content=message, parent=self, position=position, duration=3000, isClosable=True)


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
