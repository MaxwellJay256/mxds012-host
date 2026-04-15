"""
Binary ADC protocol parsing and constants.

Frame structure: Header(2) + Length(2) + Type(1) + Data(payload_len) + Checksum(2) + Tail(2)
All I/O and checksum logic isolated here for testing and reusability.
"""
import struct
from typing import List, Optional


# Protocol constants
FRAME_HEADER = b'\xAA\x55'
FRAME_TAIL = b'\x0D\x0A'

# Packet types
PKT_TYPE_ADC = 0x01                    # ADC data (downlink from STM32)
PKT_TYPE_POWER = 0x02                  # Power data (downlink from STM32)
PKT_TYPE_NOTIFY_CONFIG = 0x04          # Notify config change (downlink from STM32)

PKT_TYPE_SET_CONFIG = 0x10          # Set scan frequency (uplink to STM32)
PKT_TYPE_ACQ_CONFIG = 0x11          # Query scan frequency (uplink to STM32)

MAX_PAYLOAD = 16384  # bytes; must be multiple of 8 (4ch×2bytes×samples) for ADC frames


def validateFrameHeader(packet: bytes) -> bool:
    """Check if packet starts with correct frame header."""
    return len(packet) >= 2 and packet[0:2] == FRAME_HEADER


def validateFrameTail(packet: bytes, payload_len: int) -> bool:
    """Check if packet ends with correct frame tail at the right position."""
    tail_pos = 7 + payload_len
    return (
        len(packet) >= tail_pos + 2
        and packet[tail_pos:tail_pos + 2] == FRAME_TAIL
    )


def _computeChecksum(data_bytes: bytes) -> int:
    """Compute XOR checksum over bytes."""
    checksum = 0
    for byte in data_bytes:
        checksum ^= byte
    return checksum


def _parseCommonFrame(packet: bytes, expected_type: int) -> tuple[int, bytes]:
    """Validate a frame and return its payload length and payload bytes."""
    if not validateFrameHeader(packet):
        raise ValueError("Invalid frame header: expected 0xAA55")

    if len(packet) < 4:
        raise ValueError("Packet too short for length field")

    payload_len = struct.unpack('<H', packet[2:4])[0]

    if payload_len > MAX_PAYLOAD:
        raise ValueError(f"Invalid payload_len={payload_len}")

    total_len = 2 + 2 + 1 + payload_len + 2 + 2
    if len(packet) < total_len:
        raise ValueError(f"Incomplete packet: expected {total_len}, got {len(packet)}")

    pkt_type = packet[4]
    if pkt_type != expected_type:
        raise ValueError(f"Invalid packet type: {pkt_type}, expected {expected_type}")

    payload = packet[5 : 5 + payload_len]
    if len(payload) != payload_len:
        raise ValueError("Payload length mismatch")

    checksum_received = struct.unpack('<H', packet[5 + payload_len : 7 + payload_len])[0]
    calc_checksum = _computeChecksum(packet[2 : 5 + payload_len])

    if calc_checksum != checksum_received:
        raise ValueError(
            f"Checksum mismatch: expected {checksum_received:04X}, got {calc_checksum:04X}"
        )

    if not validateFrameTail(packet, payload_len):
        raise ValueError("Invalid frame tail: expected 0x0D0A")

    return payload_len, payload


def buildFrame(pkt_type: int, payload: bytes) -> bytes:
    """
    Build a complete binary frame with header, length, type, payload, checksum, and tail.
    
    Frame structure: Header(2) + Length(2) + Type(1) + Data(payload_len) + Checksum(2) + Tail(2)
    
    Args:
        pkt_type: Packet type (0x01 for ADC, 0x10-0x13 for commands).
        payload: Raw payload bytes.
    
    Returns:
        Complete binary frame ready to send over serial.
    
    Raises:
        ValueError if payload exceeds MAX_PAYLOAD.
    """
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"Payload too large: {len(payload)} > {MAX_PAYLOAD}")
    
    payload_len = len(payload)
    
    # Build frame: header + length + type + payload (checksum will be calculated over length..payload)
    frame = b""
    frame += FRAME_HEADER
    frame += struct.pack('<H', payload_len)
    frame += bytes([pkt_type])
    frame += payload
    
    # Compute checksum over length field through payload end
    checksum_data = frame[2:]  # from length field onwards
    checksum = _computeChecksum(checksum_data)
    frame += struct.pack('<H', checksum)
    
    # Add tail
    frame += FRAME_TAIL
    
    return frame


def parseADCFrame(packet: bytes) -> Optional[List[List[int]]]:
    """
    Parse binary ADC data packet into 4-channel sample lists.
    
    Frame structure: Header(2) + Length(2) + Type(1) + Data(payload_len) + Checksum(2) + Tail(2)
    
    Args:
        packet: Raw binary packet bytes.
    
    Returns:
        List of 4 channel lists (each containing int16 samples) on success, None on error.
        Raises ValueError on protocol violations.
    """
    try:
        payload_len, adc_data_bytes = _parseCommonFrame(packet, PKT_TYPE_ADC)

        if (payload_len % 8) != 0:
            raise ValueError(f"Invalid payload_len={payload_len}")

        # Extract ADC data (little-endian int16)
        # Unpack as little-endian signed 16-bit integers
        num_samples = payload_len // 8  # 4 channels × 2 bytes per int16
        adc_raw = struct.unpack(f'<{payload_len // 2}h', adc_data_bytes)

        # Reorganize into 4 channels from interleaved samples:
        # CH1[0], CH2[0], CH3[0], CH4[0], CH1[1], ...
        channels = []
        for ch in range(4):
            start = ch * num_samples
            end = (ch + 1) * num_samples
            channels.append(list(adc_raw[start:end]))
        
        return channels
    
    except struct.error as e:
        raise ValueError(f"Struct unpack error: {e}")


def parsePowerFrame(packet: bytes) -> dict:
    """Parse a power frame into bus/current/power values."""
    try:
        payload_len, payload = _parseCommonFrame(packet, PKT_TYPE_POWER)
        if payload_len != 6:
            raise ValueError(f"Invalid payload_len={payload_len}")

        bus_mv, current_ma, power_mw = struct.unpack('<HHH', payload)
        return {
            'bus_mv': bus_mv,
            'current_ma': current_ma,
            'power_mw': power_mw,
        }
    except struct.error as e:
        raise ValueError(f"Struct unpack error: {e}")


def parseConfigFrame(packet: bytes) -> dict:
    """Parse a configuration notification frame into scan frequency and sample depth."""
    try:
        payload_len, payload = _parseCommonFrame(packet, PKT_TYPE_NOTIFY_CONFIG)
        if payload_len != 3:
            raise ValueError(f"Invalid payload_len={payload_len}")

        scan_frequency_hz, sample_depth = struct.unpack('<BH', payload)
        return {
            'scan_frequency_hz': scan_frequency_hz,
            'sample_depth': sample_depth,
        }
    except struct.error as e:
        raise ValueError(f"Struct unpack error: {e}")


# Command packet builders (uplink to STM32)


def buildSetConfigCommand(scan_frequency_hz: int, sample_depth: int) -> bytes:
    """Build a full configuration command frame.

    STM32 constraints:
    - scan_frequency_hz: 1..60
    - sample_depth: 1..1024
    """
    if not (1 <= scan_frequency_hz <= 60):
        raise ValueError(f"Scan frequency out of range [1, 60]: {scan_frequency_hz}")
    if not (1 <= sample_depth <= 1024):
        raise ValueError(f"Sample depth out of range [1, 1024]: {sample_depth}")

    payload = struct.pack('<BH', scan_frequency_hz, sample_depth)
    return buildFrame(PKT_TYPE_SET_CONFIG, payload)


def buildAcqConfigCommand() -> bytes:
    """Build a configuration query command frame."""
    return buildFrame(PKT_TYPE_ACQ_CONFIG, b"")
