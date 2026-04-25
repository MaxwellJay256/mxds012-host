"""
Serial port bidirectional communication service.

Reads binary data from the serial port, parses ADC frames, and emits signals.
Also supports outbound command transmission.
All parsing logic delegated to protocol module.
"""
import struct
from typing import List, Optional
from queue import Queue, Empty
import threading

from PyQt5.QtCore import QThread, pyqtSignal
from serial import Serial, SerialException
from serial.tools import list_ports

from utils.Protocol import (
    parseADCFrame,
    parsePowerFrame,
    parseConfigFrame,
    buildSetConfigCommand,
    buildAcqConfigCommand,
    PKT_TYPE_ADC,
    PKT_TYPE_POWER,
    PKT_TYPE_NOTIFY_CONFIG,
)
from utils.PreProcess import RealtimePreprocessor
from utils.PreprocessParamService import PreprocessParamService


class SerialReader(QThread):
    """Bidirectional serial port service for ADC data and command transmission.

    Manages both inbound ADC frame reception and outbound command transmission.
    All I/O and parsing occur inside this QThread so the GUI thread remains responsive.
    Parsed channel data are emitted via the ``frameReady`` signal.
    """
    frameReady = pyqtSignal(list)
    plotFrameReady = pyqtSignal(list)
    modelFrameReady = pyqtSignal(list)
    rawFrameReady = pyqtSignal(list)
    powerReady = pyqtSignal(dict)
    configReady = pyqtSignal(dict)
    statusChanged = pyqtSignal(str)
    errorOccurred = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(self, port: str, baudrate: int = 115200, parent=None, debug: bool = False):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self._running = False
        self.debug = debug
        # Statistics
        self.discarded_bytes = 0
        self.invalid_packets = 0
        self.checksum_errors = 0
        
        # Outbound command queue
        self._send_queue = Queue()
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._last_config = {
            'scan_frequency_hz': 0,
            'sample_depth': 0,
        }
        self._realtime_preprocess_enabled = False
        self._preprocessor: Optional[RealtimePreprocessor] = None
        self._preprocess_service = PreprocessParamService.get_instance()
        self._preprocess_service.parametersChanged.connect(self._onPreprocessParamsChanged)

    def setDebug(self, enabled: bool):
        """Enable/disable hex/raw logging."""
        self.debug = enabled

    def stats(self) -> dict:
        """Return statistics about parsing/bytes.

        keys: discarded_bytes, invalid_packets, checksum_errors
        """
        return {
            'discarded_bytes': self.discarded_bytes,
            'invalid_packets': self.invalid_packets,
            'checksum_errors': self.checksum_errors,
        }

    def sendSetConfig(self, scan_frequency_hz: int, sample_depth: int):
        """Queue a full configuration command for transmission.
        
        Args:
            scan_frequency_hz: Scan frequency in Hz (uint8 range).
            sample_depth: Sample depth (uint16 range).
        """
        try:
            packet = buildSetConfigCommand(scan_frequency_hz, sample_depth)
            self._send_queue.put(packet)
        except Exception as e:
            self.errorOccurred.emit(f"无法构造配置设置命令: {e}")

    def sendAcqConfig(self):
        """Queue a 'query configuration' command for transmission."""
        try:
            packet = buildAcqConfigCommand()
            self._send_queue.put(packet)
        except Exception as e:
            self.errorOccurred.emit(f"无法构造配置查询命令: {e}")

    def sendSetScanFreq(self, freq_hz: int):
        """Queue a configuration command while preserving the last known sample depth."""
        self.sendSetConfig(freq_hz, self._last_config['sample_depth'])

    def setRealtimePreprocessEnabled(self, enabled: bool):
        with self._state_lock:
            self._realtime_preprocess_enabled = bool(enabled)

    def isRealtimePreprocessEnabled(self) -> bool:
        with self._state_lock:
            return self._realtime_preprocess_enabled

    def _onPreprocessParamsChanged(self, params):
        if not self.isRealtimePreprocessEnabled():
            return
        try:
            self._preprocessor = RealtimePreprocessor(params)
        except Exception:
            self._preprocessor = None


    def run(self):
        try:
            serial_port = Serial(self.port, self.baudrate, timeout=0.2)
        except SerialException as exc:
            self.errorOccurred.emit(f"无法打开串口 {self.port}: {exc}")
            self.stopped.emit()
            return

        self.statusChanged.emit(f"已连接 {self.port}")
        self._running = True
        buffer = bytearray()

        try:
            while self._running:
                # Try to send any pending commands
                self._sendPending(serial_port)
                
                try:
                    chunk = serial_port.read(256)
                except SerialException as exc:
                    self.errorOccurred.emit(f"串口读取错误: {exc}")
                    break

                if not chunk:
                    continue

                if self.debug:
                    print(f"[DEBUG] recv {chunk.hex()}")

                buffer.extend(chunk)

                # Parsing loop: extract complete frames from buffer
                while len(buffer) >= 9:
                    header_idx = buffer.find(b'\xAA\x55')
                    if header_idx == -1:
                        # Keep last 0xAA in case it's start of next header
                        if buffer and buffer[-1] == 0xAA:
                            self.discarded_bytes += len(buffer) - 1
                            buffer[:] = buffer[-1:]
                        else:
                            self.discarded_bytes += len(buffer)
                            buffer.clear()
                        if self.debug:
                            print(f"[DEBUG] 丢弃无效字节, total discards={self.discarded_bytes}")
                        break

                    if header_idx > 0:
                        self.discarded_bytes += header_idx
                        del buffer[:header_idx]
                        if self.debug:
                            print(f"[DEBUG] 丢弃 {header_idx} 个垃圾字节")

                    if len(buffer) < 4:
                        break

                    # Read length field (little-endian)
                    payload_len = struct.unpack('<H', buffer[2:4])[0]
                    
                    # Validate legality (ADC frames must have payload_len % 8 == 0; commands may not)
                    if payload_len > 16384:
                        self.invalid_packets += 1
                        self.errorOccurred.emit(f"非法 payload_len={payload_len}")
                        # Discard header and continue searching
                        del buffer[0:2]
                        if self.debug:
                            print("[DEBUG] payload_len 非法, 丢弃头字节继续")
                        continue

                    total_len = 2 + 2 + 1 + payload_len + 2 + 2
                    if len(buffer) < total_len:
                        break

                    packet = bytes(buffer[:total_len])
                    del buffer[:total_len]

                    processed = self._processFrame(packet)
                    if processed is not None:
                        self.frameReady.emit(processed)
        finally:
            serial_port.close()
            self._running = False
            self.statusChanged.emit("连接已关闭")
            self.stopped.emit()

    def stop(self):
        self._running = False

    def _sendPending(self, serial_port: Serial):
        """Send any pending commands from the queue.
        
        Args:
            serial_port: Open serial port object.
        """
        while not self._send_queue.empty():
            try:
                packet = self._send_queue.get_nowait()
                try:
                    with self._send_lock:
                        serial_port.write(packet)
                    if self.debug:
                        print(f"[DEBUG] sent {packet.hex()}")
                except SerialException as e:
                    self.errorOccurred.emit(f"串口写入错误: {e}")
            except Empty:
                break

    def _processFrame(self, packet: bytes) -> Optional[List[List[float]]]:
        """
        Parse binary frame using protocol module.
        
        Returns None on error (error already emitted via errorOccurred signal).
        Emits ADC data via frameReady, power data via powerReady, and config
        notifications via configReady.
        """
        try:
            if len(packet) < 5:
                raise ValueError("Packet too short for type field")
            
            pkt_type = packet[4]

            if pkt_type == PKT_TYPE_ADC:
                channels = parseADCFrame(packet)
                if channels is None:
                    raise ValueError("ADC frame parse returned no data")
                raw_channels = [[float(abs(int(sample))) for sample in channel] for channel in channels]
                self.rawFrameReady.emit(raw_channels)

                if not self.isRealtimePreprocessEnabled():
                    self.modelFrameReady.emit(raw_channels)
                    self.plotFrameReady.emit(raw_channels)
                    return raw_channels

                if self._preprocessor is None:
                    self._preprocessor = RealtimePreprocessor(self._preprocess_service.get_params_copy())

                try:
                    model_channels, output_channels, display_channels = self._preprocessor.processForModelOutputAndDisplay(raw_channels)
                    self.modelFrameReady.emit(model_channels)
                    self.plotFrameReady.emit(display_channels)
                    return output_channels
                except Exception as preprocess_error:
                    self.errorOccurred.emit(f"实时预处理失败，已回退原始数据: {preprocess_error}")
                    self.modelFrameReady.emit(raw_channels)
                    self.plotFrameReady.emit(raw_channels)
                    return raw_channels

            if pkt_type == PKT_TYPE_POWER:
                power_data = parsePowerFrame(packet)
                self.powerReady.emit(power_data)
                return None

            if pkt_type == PKT_TYPE_NOTIFY_CONFIG:
                config_data = parseConfigFrame(packet)
                self._last_config.update(config_data)
                self.configReady.emit(config_data)
                return None

            raise ValueError(f"Unsupported packet type: 0x{pkt_type:02X}")
        except ValueError as e:
            self.invalid_packets += 1
            self.errorOccurred.emit(f"数据格式错误: {str(e)}")
            return None
        except Exception as e:
            self.invalid_packets += 1
            self.errorOccurred.emit(f"未预期的错误: {str(e)}")
            return None
