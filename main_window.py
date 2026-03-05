"""
Main application window and UI orchestration.
"""
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QApplication, QLineEdit, QPushButton, QLabel, QFileDialog
)
from PyQt5.QtCore import Qt
from qfluentwidgets import (
    ComboBox,
    PrimaryPushButton,
    PushButton,
    InfoBar,
    InfoBarPosition,
    setTheme,
    Theme,
    setThemeColor,
)
from serial.tools import list_ports

from serial_reader import SerialReader
from plot_panel import PlotPanel
from recording_service import RecordingService
from recording_utils import RecordingError, build_output_path, check_file_exists


class MainWindow(QWidget):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MXDS012 Host")
        self.resize(1100, 800)

        setTheme(Theme.LIGHT)
        setThemeColor('#0078d4')

        pg.setConfigOptions(antialias=True, background=None, foreground='#f0f0f0')

        self.serial_thread: Optional[SerialReader] = None
        
        # Recording state
        self.is_recording = False
        self.recording_service: Optional[RecordingService] = None
        self.recorded_frame_count = 0
        self.record_output_dir = ""
        self.record_frame_counter = 0

        # Layout
        existing_layout = self.layout()
        if existing_layout is None:
            self._central_layout = QVBoxLayout()
            self._central_layout.setContentsMargins(24, 24, 24, 24)
            self._central_layout.setSpacing(18)
            self.setLayout(self._central_layout)
        else:
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
        """Build control bar with serial and recording controls."""
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(12)

        # Serial port selection
        self.port_combo = ComboBox()
        self.port_combo.setMinimumWidth(200)
        control_layout.addWidget(self.port_combo)

        # Refresh ports
        self.refresh_button = PushButton("刷新串口")
        self.refresh_button.clicked.connect(self._refresh_ports)
        control_layout.addWidget(self.refresh_button)

        # Start/Stop serial
        self.start_button = PrimaryPushButton("开始接收")
        self.start_button.clicked.connect(self._start_serial)
        control_layout.addWidget(self.start_button)

        self.stop_button = PushButton("停止")
        self.stop_button.clicked.connect(self._stop_serial)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        # Separator
        control_layout.addSpacing(20)

        # Recording controls (Phase 6+)
        # Output directory selector
        self.record_dir_button = PushButton("选择录制目录")
        self.record_dir_button.clicked.connect(self._select_output_directory)
        control_layout.addWidget(self.record_dir_button)

        # Filename input
        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("输入文件名 (不需要 .txt)")
        self.filename_edit.setMinimumWidth(150)
        control_layout.addWidget(self.filename_edit)

        # Start/Stop recording
        self.record_start_button = PushButton("开始录制")
        self.record_start_button.clicked.connect(self._on_record_start)
        self.record_start_button.setEnabled(False)
        control_layout.addWidget(self.record_start_button)

        self.record_stop_button = PushButton("停止录制")
        self.record_stop_button.clicked.connect(self._on_record_stop)
        self.record_stop_button.setEnabled(False)
        control_layout.addWidget(self.record_stop_button)

        # Frame counter display
        self.frame_counter_label = QLabel("已录制: 0 帧")
        control_layout.addWidget(self.frame_counter_label)

        control_layout.addStretch()

        self._central_layout.addWidget(control_bar)

    def _refresh_ports(self):
        """Refresh available serial ports."""
        previous_device = self.port_combo.currentData()
        self.port_combo.clear()

        ports = list_ports.comports()
        for port_info in ports:
            display_text = f"{port_info.device} - {port_info.description}"
            self.port_combo.addItem(display_text, port_info.device)

        if not ports:
            self._show_info("未检测到串口设备", InfoBarPosition.TOP, "warning")
        else:
            if previous_device:
                idx = self.port_combo.findData(previous_device)
                if idx >= 0:
                    self.port_combo.setCurrentIndex(idx)

    def _start_serial(self):
        """Start serial connection."""
        if self.serial_thread and self.serial_thread.isRunning():
            self._show_info("串口已在运行", InfoBarPosition.TOP, "warning")
            return

        if self.port_combo.count() == 0:
            self._show_info("没有可用串口", InfoBarPosition.TOP, "warning")
            return

        port = self.port_combo.currentData()
        if not port:
            display = self.port_combo.currentText().strip()
            if display:
                port_candidate = display.split()[0]
                if port_candidate:
                    port = port_candidate

        if not port:
            self._show_info("请选择串口", InfoBarPosition.TOP, "warning")
            return

        self._show_info(f"已选择串口: {port}", InfoBarPosition.TOP, "info")

        self.serial_thread = SerialReader(port)
        self.serial_thread.frameReady.connect(self.plot_panel.update_plots)
        self.serial_thread.frameReady.connect(self._on_frame_received)  # For recording
        self.serial_thread.statusChanged.connect(lambda msg: self._show_info(msg, InfoBarPosition.TOP))
        self.serial_thread.errorOccurred.connect(lambda msg: self._show_info(msg, InfoBarPosition.BOTTOM, "error"))
        self.serial_thread.stopped.connect(self._on_serial_stopped)

        self.serial_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_start_button.setEnabled(True)

    def _stop_serial(self):
        """Stop serial connection."""
        # Force stop recording if active
        if self.is_recording:
            self._stop_recording()
        
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait(2000)
        else:
            self._on_serial_stopped()

    def _on_serial_stopped(self):
        """Handle serial thread stopped."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_start_button.setEnabled(False)
        self.record_stop_button.setEnabled(False)
        self.serial_thread = None

    def _select_output_directory(self):
        """Open directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择录制文件保存目录",
            ""
        )
        if directory:
            self.record_output_dir = directory
            self._show_info(f"录制目录: {directory}", InfoBarPosition.TOP, "info")
        else:
            self.record_output_dir = ""

    def _on_record_start(self):
        """Start recording."""
        if self.is_recording:
            self._show_info("录制已在进行中", InfoBarPosition.TOP, "warning")
            return
        
        if not self.serial_thread or not self.serial_thread.isRunning():
            self._show_info("请先连接串口", InfoBarPosition.TOP, "warning")
            return
        
        if not self.record_output_dir:
            self._show_info("请先选择录制目录", InfoBarPosition.TOP, "warning")
            return
        
        filename = self.filename_edit.text().strip()
        if not filename:
            self._show_info("请输入文件名", InfoBarPosition.TOP, "warning")
            return
        
        # Build full path
        full_path, error_msg = build_output_path(self.record_output_dir, filename)
        if error_msg:
            self._show_info(f"路径错误: {error_msg}", InfoBarPosition.TOP, "error")
            return
        
        # Check if file exists
        if check_file_exists(full_path):
            # For now, just overwrite with a warning
            self._show_info(f"文件已存在，将覆盖: {full_path}", InfoBarPosition.TOP, "warning")
        
        # Start recording
        try:
            self.recording_service = RecordingService(full_path)
            self.recording_service.start()
            self.is_recording = True
            self.recorded_frame_count = 0
            self.record_frame_counter = 0
            self._update_frame_counter()
            
            self.record_start_button.setEnabled(False)
            self.record_stop_button.setEnabled(True)
            self._show_info(f"开始录制: {full_path}", InfoBarPosition.TOP, "info")
        except RecordingError as e:
            self._show_info(f"录制启动失败: {e}", InfoBarPosition.BOTTOM, "error")
            self.is_recording = False

    def _on_record_stop(self):
        """Stop recording."""
        self._stop_recording()

    def _stop_recording(self):
        """Internal method to stop recording."""
        if not self.is_recording:
            return
        
        try:
            if self.recording_service:
                self.recording_service.stop()
                frame_count = self.recording_service.get_frame_count()
                self._show_info(
                    f"停止录制: {frame_count} 帧已保存",
                    InfoBarPosition.TOP,
                    "success"
                )
        except RecordingError as e:
            self._show_info(f"录制停止失败: {e}", InfoBarPosition.BOTTOM, "error")
        finally:
            self.is_recording = False
            self.recording_service = None
            self.record_start_button.setEnabled(True)
            self.record_stop_button.setEnabled(False)
            self.recorded_frame_count = 0
            self._update_frame_counter()

    def _on_frame_received(self, channels):
        """Handle incoming frame for both plotting and recording."""
        if self.is_recording and self.recording_service:
            try:
                self.recording_service.write_frame(self.record_frame_counter, channels)
                self.recorded_frame_count += 1
                self.record_frame_counter += 1
                self._update_frame_counter()
            except RecordingError as e:
                self._show_info(f"写入帧失败: {e}", InfoBarPosition.BOTTOM, "error")
                self._stop_recording()

    def _update_frame_counter(self):
        """Update frame counter label."""
        self.frame_counter_label.setText(f"已录制: {self.recorded_frame_count} 帧")

    def _show_info(self, message: str, position: InfoBarPosition, level: str = "success"):
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

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop recording if active
        if self.is_recording:
            self._stop_recording()
        
        # Stop serial if running
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait(2000)
        
        super().closeEvent(event)
