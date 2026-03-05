"""
Recording service: real-time TXT file writer for ADC frames.

Handles file I/O, frame formatting, and frame counting.
"""
from typing import List, Optional

from recording_utils import RecordingError


class RecordingService:
    """Service for recording ADC frame data to TXT file in real-time."""
    
    def __init__(self, output_path: str):
        """
        Initialize recording service.
        
        Args:
            output_path: Full path to output TXT file.
        """
        self.output_path = output_path
        self.file: Optional[object] = None
        self.frame_count = 0
        self.is_active = False
    
    def start(self) -> bool:
        """
        Open file for writing.
        
        Returns:
            True if successful, False otherwise (error reason in last_error).
        """
        try:
            self.file = open(self.output_path, 'w', encoding='utf-8')
            self.frame_count = 0
            self.is_active = True
            return True
        except IOError as e:
            self.is_active = False
            raise RecordingError(f"无法打开文件 {self.output_path}: {e}")
    
    def write_frame(self, frame_idx: int, channels: List[List[float]]) -> bool:
        """
        Write one frame of ADC data to file in 4-line format.
        
        Each frame produces 4 lines:
        - Line 0: Frame N, Channel 1 data (space-separated samples)
        - Line 1: Frame N, Channel 2 data
        - Line 2: Frame N, Channel 3 data
        - Line 3: Frame N, Channel 4 data
        
        Args:
            frame_idx: Frame sequence number.
            channels: List of 4 channel lists, each containing samples.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.is_active or not self.file:
            raise RecordingError("Recording service not active")
        
        if len(channels) != 4:
            raise RecordingError(f"Expected 4 channels, got {len(channels)}")
        
        try:
            for ch_idx, data in enumerate(channels):
                # Format: "F000123 CH1 val1 val2 val3 ..."
                prefix = f"F{frame_idx:06d} CH{ch_idx + 1}"
                values_str = " ".join(f"{v:.1f}" for v in data)
                line = f"{prefix} {values_str}\n"
                self.file.write(line)
            
            self.frame_count += 1
            self.file.flush()  # Immediate disk write
            return True
        except Exception as e:
            raise RecordingError(f"Error writing frame {frame_idx}: {e}")
    
    def stop(self) -> bool:
        """
        Close file handle and stop recording.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.file:
                self.file.flush()
                self.file.close()
                self.file = None
            self.is_active = False
            return True
        except Exception as e:
            raise RecordingError(f"Error closing file: {e}")
    
    def get_frame_count(self) -> int:
        """Return number of frames written in current session."""
        return self.frame_count
    
    def __del__(self):
        """Ensure file is closed on object destruction."""
        try:
            if self.file:
                self.file.close()
        except:
            pass
