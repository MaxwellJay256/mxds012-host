"""
Main application window and UI orchestration.
"""
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QApplication, QLineEdit, QPushButton, QLabel, QFileDialog, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from qfluentwidgets import (
    ComboBox,
    PrimaryPushButton,
    PushButton,
    ProgressRing,
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

    FIXED_RECORD_FPS = 20
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MXDS012 Host")
        self.resize(1920, 1080)

        setTheme(Theme.LIGHT)
        setThemeColor('#0078d4')

        pg.setConfigOptions(antialias=True, background=None, foreground='#f0f0f0')

        self.serial_thread: Optional[SerialReader] = None
        
        # Recording state
        self.is_recording = False
        self.is_record_paused_pending_save = False
        self.recording_service: Optional[RecordingService] = None
        self.recorded_frame_count = 0
        self.record_output_dir = ""
        self.record_frame_counter = 0
        self.record_file_path = ""
        self.current_record_mode = None
        self.timed_target_frames = 0
        self.progress_ring_format = "0.0 s"

        self.timed_progress_timer = QTimer(self)
        self.timed_progress_timer.setInterval(100)
        self.timed_progress_timer.timeout.connect(self._on_timed_progress_tick)

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
        self.record_continuous_start_button = PushButton("开始连续录制")
        self.record_continuous_start_button.clicked.connect(self._on_record_start_continuous)
        self.record_continuous_start_button.setEnabled(False)
        control_layout.addWidget(self.record_continuous_start_button)

        self.record_duration_spin = QSpinBox()
        self.record_duration_spin.setRange(1, 60)
        self.record_duration_spin.setValue(5)
        self.record_duration_spin.setSuffix(" s")
        self.record_duration_spin.setToolTip("定时录制秒数，实际录制帧数=秒数x20")
        control_layout.addWidget(self.record_duration_spin)

        self.record_timed_start_button = PushButton("开始定时录制")
        self.record_timed_start_button.clicked.connect(self._on_record_start_timed)
        self.record_timed_start_button.setEnabled(False)
        control_layout.addWidget(self.record_timed_start_button)

        self.record_stop_button = PushButton("停止录制")
        self.record_stop_button.clicked.connect(self._on_record_stop)
        self.record_stop_button.setEnabled(False)
        control_layout.addWidget(self.record_stop_button)

        self.record_progress_ring = ProgressRing(self)
        self.record_progress_ring.setRange(0, 100)
        self.record_progress_ring.setValue(0)
        self.record_progress_ring.setTextVisible(True)
        self.record_progress_ring.setFixedSize(60, 60)
        self.record_progress_ring.setStrokeWidth(4)
        self.record_progress_ring.setFormat(self.progress_ring_format)
        control_layout.addWidget(self.record_progress_ring)

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
            self._show_info("未检测到串口设备", InfoBarPosition.BOTTOM, "warning")
        else:
            if previous_device:
                idx = self.port_combo.findData(previous_device)
                if idx >= 0:
                    self.port_combo.setCurrentIndex(idx)

    def _start_serial(self):
        """Start serial connection."""
        if self.serial_thread and self.serial_thread.isRunning():
            self._show_info("串口已在运行", InfoBarPosition.BOTTOM, "warning")
            return

        if self.port_combo.count() == 0:
            self._show_info("没有可用串口", InfoBarPosition.BOTTOM, "warning")
            return

        port = self.port_combo.currentData()
        if not port:
            display = self.port_combo.currentText().strip()
            if display:
                port_candidate = display.split()[0]
                if port_candidate:
                    port = port_candidate

        if not port:
            self._show_info("请选择串口", InfoBarPosition.BOTTOM, "warning")
            return

        self._show_info(f"已选择串口: {port}", InfoBarPosition.BOTTOM, "info")

        self.serial_thread = SerialReader(port)
        self.serial_thread.frameReady.connect(self.plot_panel.update_plots)
        self.serial_thread.frameReady.connect(self._on_frame_received)  # For recording
        self.serial_thread.statusChanged.connect(lambda msg: self._show_info(msg, InfoBarPosition.BOTTOM))
        self.serial_thread.errorOccurred.connect(lambda msg: self._show_info(msg, InfoBarPosition.BOTTOM, "error"))
        self.serial_thread.stopped.connect(self._on_serial_stopped)

        self.serial_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._update_recording_ui_state()

    def _stop_serial(self):
        """Stop serial connection."""
        # Finalize recording if a file is still open (active or paused)
        if self.recording_service and self.recording_service.get_state() != RecordingService.STATE_IDLE:
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
        self.serial_thread = None
        self._update_recording_ui_state()

    def _select_output_directory(self):
        """Open directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择录制文件保存目录",
            ""
        )
        if directory:
            self.record_output_dir = directory
            self._show_info(f"录制目录: {directory}", InfoBarPosition.BOTTOM, "info")
        else:
            self.record_output_dir = ""

    def _ensure_serial_running(self) -> bool:
        """Return whether serial receiving is currently running."""
        return bool(self.serial_thread and self.serial_thread.isRunning())

    def _get_recording_state(self) -> str:
        """Get current recording service state, defaulting to idle."""
        if not self.recording_service:
            return RecordingService.STATE_IDLE
        return self.recording_service.get_state()

    def _prepare_new_recording_service(self) -> bool:
        """Validate path inputs and create a new recording service instance."""
        if not self.record_output_dir:
            self._show_info("请先选择录制目录", InfoBarPosition.BOTTOM, "warning")
            return False

        filename = self.filename_edit.text().strip()
        if not filename:
            self._show_info("请输入文件名", InfoBarPosition.BOTTOM, "warning")
            return False

        full_path, error_msg = build_output_path(self.record_output_dir, filename)
        if error_msg:
            self._show_info(f"路径错误: {error_msg}", InfoBarPosition.BOTTOM, "error")
            return False

        if check_file_exists(full_path):
            self._show_info(f"文件已存在，将覆盖: {full_path}", InfoBarPosition.BOTTOM, "warning")

        self.record_file_path = full_path
        self.recording_service = RecordingService(full_path)
        self.recorded_frame_count = 0
        self.record_frame_counter = 0
        self._update_frame_counter()
        return True

    def _start_timed_progress(self, selected_seconds: int):
        """Initialize and start timed progress ring updates."""
        max_tenths = int(selected_seconds * 10)
        self.record_progress_ring.setRange(0, max_tenths)
        self.record_progress_ring.setValue(max_tenths)
        self._update_countdown_display(self.timed_target_frames)
        self.timed_progress_timer.start()

    def _update_countdown_display(self, remaining_frames: int):
        """Update ring value and text using remaining seconds."""
        remaining_seconds = max(0.0, remaining_frames / float(self.FIXED_RECORD_FPS))
        remaining_tenths = max(0, int(round(remaining_seconds * 10)))
        self.record_progress_ring.setValue(remaining_tenths)
        self.record_progress_ring.setFormat(f"{remaining_seconds:.1f} s")

    def _on_record_start_continuous(self):
        """Start continuous recording or resume from paused state."""
        if not self._ensure_serial_running():
            self._show_info("请先连接串口", InfoBarPosition.BOTTOM, "warning")
            return

        state = self._get_recording_state()
        if state == RecordingService.STATE_ACTIVE_CONTINUOUS:
            self._show_info("连续录制已在进行中", InfoBarPosition.BOTTOM, "warning")
            return
        if state == RecordingService.STATE_ACTIVE_TIMED:
            self._show_info("定时录制进行中，无法切换模式", InfoBarPosition.BOTTOM, "warning")
            return

        try:
            if state == RecordingService.STATE_PAUSED_PENDING_SAVE and self.recording_service:
                self.recording_service.start_continuous(new_session=False)
                self._show_info("已从暂停状态恢复为连续录制", InfoBarPosition.BOTTOM, "info")
            else:
                if not self._prepare_new_recording_service():
                    return
                service = self.recording_service
                if service is None:
                    return
                service.start_continuous(new_session=True)
                self._show_info(f"开始连续录制: {self.record_file_path}", InfoBarPosition.BOTTOM, "info")

            self.current_record_mode = "continuous"
            self.is_recording = True
            self.is_record_paused_pending_save = False
            self.timed_target_frames = 0
            self.timed_progress_timer.stop()
            self.record_progress_ring.setRange(0, 100)
            self.record_progress_ring.setValue(0)
            self.record_progress_ring.setFormat(self.progress_ring_format)
            self._update_recording_ui_state()
        except RecordingError as e:
            self._show_info(f"连续录制启动失败: {e}", InfoBarPosition.BOTTOM, "error")
            self.is_recording = False
            self._update_recording_ui_state()

    def _on_record_start_timed(self):
        """Start timed recording or resume from paused state."""
        if not self._ensure_serial_running():
            self._show_info("请先连接串口", InfoBarPosition.BOTTOM, "warning")
            return

        selected_seconds = int(self.record_duration_spin.value())
        target_frames = selected_seconds * self.FIXED_RECORD_FPS
        state = self._get_recording_state()

        if state == RecordingService.STATE_ACTIVE_TIMED:
            self._show_info("定时录制已在进行中", InfoBarPosition.BOTTOM, "warning")
            return
        if state == RecordingService.STATE_ACTIVE_CONTINUOUS:
            self._show_info("连续录制进行中，无法切换模式", InfoBarPosition.BOTTOM, "warning")
            return

        try:
            if state == RecordingService.STATE_PAUSED_PENDING_SAVE and self.recording_service:
                self.recording_service.start_timed(target_frames=target_frames, new_session=False)
                self._show_info(
                    f"恢复定时录制: {selected_seconds}s ({target_frames} 帧)",
                    InfoBarPosition.BOTTOM,
                    "info"
                )
            else:
                if not self._prepare_new_recording_service():
                    return
                service = self.recording_service
                if service is None:
                    return
                service.start_timed(target_frames=target_frames, new_session=True)
                self._show_info(
                    f"开始定时录制: {selected_seconds}s ({target_frames} 帧)",
                    InfoBarPosition.BOTTOM,
                    "info"
                )

            self.current_record_mode = "timed"
            self.timed_target_frames = target_frames
            self.is_recording = True
            self.is_record_paused_pending_save = False
            self._start_timed_progress(selected_seconds)
            self._update_recording_ui_state()
        except RecordingError as e:
            self._show_info(f"定时录制启动失败: {e}", InfoBarPosition.BOTTOM, "error")
            self.is_recording = False
            self._update_recording_ui_state()

    def _on_timed_progress_tick(self):
        """Refresh timed countdown display every 0.1s."""
        if not self.recording_service:
            self.timed_progress_timer.stop()
            return

        if self.recording_service.get_state() != RecordingService.STATE_ACTIVE_TIMED:
            self.timed_progress_timer.stop()
            return

        remaining_frames = self.recording_service.get_remaining_timed_frames()
        self._update_countdown_display(remaining_frames)

    def _on_record_stop(self):
        """Stop recording."""
        self._stop_recording()

    def _stop_recording(self):
        """Internal method to stop recording."""
        if not self.recording_service:
            return

        try:
            frame_count = self.recording_service.get_frame_count()
            self.recording_service.stop()
            self._show_info(
                f"停止录制: {frame_count} 帧已保存",
                InfoBarPosition.BOTTOM,
                "success"
            )
        except RecordingError as e:
            self._show_info(f"录制停止失败: {e}", InfoBarPosition.BOTTOM, "error")
        finally:
            self.is_recording = False
            self.is_record_paused_pending_save = False
            self.recording_service = None
            self.current_record_mode = None
            self.timed_target_frames = 0
            self.timed_progress_timer.stop()
            self.record_progress_ring.setRange(0, 100)
            self.record_progress_ring.setValue(0)
            self.record_progress_ring.setFormat(self.progress_ring_format)
            self.recorded_frame_count = 0
            self.record_frame_counter = 0
            self._update_frame_counter()
            self._update_recording_ui_state()

    def _on_frame_received(self, channels):
        """Handle incoming frame for both plotting and recording."""
        if not self.recording_service:
            return

        state = self.recording_service.get_state()
        if state not in (RecordingService.STATE_ACTIVE_CONTINUOUS, RecordingService.STATE_ACTIVE_TIMED):
            return

        try:
            self.recording_service.write_frame(self.record_frame_counter, channels)
            self.record_frame_counter += 1
            self.recorded_frame_count = self.recording_service.get_frame_count()
            self._update_frame_counter()

            if self.current_record_mode == "timed":
                remaining_frames = self.recording_service.get_remaining_timed_frames()
                self._update_countdown_display(remaining_frames)

                if self.recording_service.get_state() == RecordingService.STATE_PAUSED_PENDING_SAVE:
                    self.is_recording = False
                    self.is_record_paused_pending_save = True
                    self.timed_progress_timer.stop()
                    self._show_info("定时录制完成，已进入暂停状态（待停止后保存）", InfoBarPosition.BOTTOM, "success")
                    self._update_recording_ui_state()
            else:
                self.is_recording = True
                self.is_record_paused_pending_save = False
        except RecordingError as e:
            self._show_info(f"写入帧失败: {e}", InfoBarPosition.BOTTOM, "error")
            self._stop_recording()

    def _update_frame_counter(self):
        """Update frame counter label."""
        self.frame_counter_label.setText(f"已录制: {self.recorded_frame_count} 帧")

    def _update_recording_ui_state(self):
        """Update recording control enabled states based on serial and recording state."""
        serial_running = self._ensure_serial_running()
        state = self._get_recording_state()
        can_start = serial_running and state in (
            RecordingService.STATE_IDLE,
            RecordingService.STATE_PAUSED_PENDING_SAVE,
        )

        self.record_continuous_start_button.setEnabled(can_start)
        self.record_timed_start_button.setEnabled(can_start)
        self.record_duration_spin.setEnabled(state != RecordingService.STATE_ACTIVE_TIMED)
        self.record_stop_button.setEnabled(state != RecordingService.STATE_IDLE)

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
        # Finalize recording if file is open (active or paused)
        if self.recording_service and self.recording_service.get_state() != RecordingService.STATE_IDLE:
            self._stop_recording()
        
        # Stop serial if running
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
            self.serial_thread.wait(2000)
        
        super().closeEvent(event)
