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
PKT_TYPE_ADC = 0x01
MAX_PAYLOAD = 16384  # bytes; must be multiple of 8 (4ch×2bytes×samples)


def validate_frame_header(packet: bytes) -> bool:
    """Check if packet starts with correct frame header."""
    return len(packet) >= 2 and packet[0:2] == FRAME_HEADER


def validate_frame_tail(packet: bytes, payload_len: int) -> bool:
    """Check if packet ends with correct frame tail at the right position."""
    tail_pos = 7 + payload_len
    return (
        len(packet) >= tail_pos + 2
        and packet[tail_pos:tail_pos + 2] == FRAME_TAIL
    )


def parse_adc_frame(packet: bytes) -> Optional[List[List[int]]]:
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
        # 1. Verify frame header
        if not validate_frame_header(packet):
            raise ValueError("Invalid frame header: expected 0xAA55")
        
        if len(packet) < 4:
            raise ValueError("Packet too short for length field")
        
        # 2. Read length (little-endian uint16)
        payload_len = struct.unpack('<H', packet[2:4])[0]
        
        # 3. Validate payload length
        if payload_len > MAX_PAYLOAD or (payload_len % 8) != 0:
            raise ValueError(f"Invalid payload_len={payload_len}")
        
        # 4. Check total packet length
        total_len = 2 + 2 + 1 + payload_len + 2 + 2
        if len(packet) < total_len:
            raise ValueError(f"Incomplete packet: expected {total_len}, got {len(packet)}")
        
        # 5. Verify packet type
        pkt_type = packet[4]
        if pkt_type != PKT_TYPE_ADC:
            raise ValueError(f"Invalid packet type: {pkt_type}, expected {PKT_TYPE_ADC}")
        
        # 6. Extract ADC data (little-endian int16)
        adc_data_bytes = packet[5 : 5 + payload_len]
        if len(adc_data_bytes) != payload_len:
            raise ValueError("ADC data length mismatch")
        
        # Unpack as little-endian signed 16-bit integers
        num_samples = payload_len // 8  # 4 channels × 2 bytes per int16
        adc_raw = struct.unpack(f'<{payload_len // 2}h', adc_data_bytes)
        
        # 7. Validate checksum (XOR over length field through data)
        checksum_received = struct.unpack('<H', packet[5 + payload_len : 7 + payload_len])[0]
        calc_checksum = 0
        for byte in packet[2 : 5 + payload_len]:
            calc_checksum ^= byte
        
        if calc_checksum != checksum_received:
            raise ValueError(
                f"Checksum mismatch: expected {checksum_received:04X}, got {calc_checksum:04X}"
            )
        
        # 8. Verify frame tail
        if not validate_frame_tail(packet, payload_len):
            raise ValueError("Invalid frame tail: expected 0x0D0A")
        
        # 9. Reorganize into 4 channels
        channels = []
        for ch in range(4):
            start = ch * num_samples
            end = (ch + 1) * num_samples
            channels.append(list(adc_raw[start:end]))
        
        return channels
    
    except struct.error as e:
        raise ValueError(f"Struct unpack error: {e}")
