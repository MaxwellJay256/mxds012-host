"""
Serial port reading thread with frame parsing.

Reads binary data from serial port, parses ADC frames, and emits signals.
All parsing logic delegated to protocol module.
"""
import struct
from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal
from serial import Serial, SerialException
from serial.tools import list_ports

from protocol import parse_adc_frame


class SerialReader(QThread):
    """Thread responsible for reading from the serial port and parsing frames.

    All I/O and parsing occur inside this QThread so the GUI thread remains
    responsive even at high packet rates. Parsed channel data are emitted via
    the ``frameReady`` signal and consumed by the UI.
    """
    frameReady = pyqtSignal(list)
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

    def set_debug(self, enabled: bool):
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

                # Parsing loop
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
                    
                    # Validate legality
                    if payload_len > 16384 or (payload_len % 8) != 0:
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

                    processed = self._process_frame(packet)
                    if processed is not None:
                        self.frameReady.emit(processed)
        finally:
            serial_port.close()
            self._running = False
            self.statusChanged.emit("连接已关闭")
            self.stopped.emit()

    def stop(self):
        self._running = False

    def _process_frame(self, packet: bytes) -> Optional[List[List[int]]]:
        """
        Parse binary ADC frame using protocol module.
        
        Returns None on error (error already emitted via errorOccurred signal).
        """
        try:
            channels = parse_adc_frame(packet)
            return channels
        except ValueError as e:
            self.invalid_packets += 1
            self.errorOccurred.emit(f"数据格式错误: {str(e)}")
            return None
        except Exception as e:
            self.invalid_packets += 1
            self.errorOccurred.emit(f"未预期的错误: {str(e)}")
            return None
