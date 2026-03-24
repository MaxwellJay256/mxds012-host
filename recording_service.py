"""
Recording service: real-time TXT file writer for ADC frames.

Handles file I/O, frame formatting, and frame counting.
"""
from typing import List, Optional, TextIO

from recording_utils import RecordingError


class RecordingService:
    """Service for recording ADC frame data to TXT file in real-time."""

    STATE_IDLE = "idle"
    STATE_ACTIVE_CONTINUOUS = "active_continuous"
    STATE_ACTIVE_TIMED = "active_timed"
    STATE_PAUSED_PENDING_SAVE = "paused_pending_save"
    
    def __init__(self, output_path: str):
        """
        Initialize recording service.
        
        Args:
            output_path: Full path to output TXT file.
        """
        self.output_path = output_path
        self.file: Optional[TextIO] = None
        self.frame_count = 0
        self.is_active = False
        self.state = self.STATE_IDLE
        self.timed_target_frames = 0
        self.timed_written_frames = 0

    def _open_new_file(self) -> None:
        """Open a new file and reset counters for a new recording session."""
        self.file = open(self.output_path, 'w', encoding='utf-8')
        self.frame_count = 0
        self.timed_target_frames = 0
        self.timed_written_frames = 0
    
    def start(self) -> bool:
        """
        Backward-compatible API: start a new continuous recording session.
        
        Returns:
            True if successful, False otherwise (error reason in last_error).
        """
        return self.start_continuous(new_session=True)

    def start_continuous(self, new_session: bool = False) -> bool:
        """Start or resume continuous recording."""
        try:
            if new_session or self.state == self.STATE_IDLE or self.file is None:
                if self.file:
                    self.file.close()
                    self.file = None
                self._open_new_file()

            self.is_active = True
            self.state = self.STATE_ACTIVE_CONTINUOUS
            return True
        except IOError as e:
            self.is_active = False
            self.state = self.STATE_IDLE
            raise RecordingError(f"无法打开文件 {self.output_path}: {e}")

    def start_timed(self, target_frames: int, new_session: bool = False) -> bool:
        """
        Start or resume timed recording.

        Args:
            target_frames: Number of frames to record in this timed round.
            new_session: Whether to start with a new file from frame 0.
        """
        if target_frames <= 0:
            raise RecordingError("target_frames must be positive")

        try:
            if new_session or self.state == self.STATE_IDLE or self.file is None:
                if self.file:
                    self.file.close()
                    self.file = None
                self._open_new_file()

            self.timed_target_frames = int(target_frames)
            self.timed_written_frames = 0
            self.is_active = True
            self.state = self.STATE_ACTIVE_TIMED
            return True
        except IOError as e:
            self.is_active = False
            self.state = self.STATE_IDLE
            raise RecordingError(f"无法打开文件 {self.output_path}: {e}")

    def pause_pending_save(self) -> bool:
        """Pause recording while keeping file handle open for potential resume."""
        if self.state in (self.STATE_ACTIVE_CONTINUOUS, self.STATE_ACTIVE_TIMED):
            self.is_active = False
            self.state = self.STATE_PAUSED_PENDING_SAVE
            return True

        if self.state == self.STATE_PAUSED_PENDING_SAVE:
            return True

        raise RecordingError("Recording service is not active")
    
    def write_frame(self, frame_idx: int, channels: List[List[int]]) -> bool:
        """
        Write one frame of ADC data to file in 4-line format.
        
        Each frame produces 4 lines:
        - Line 0: Frame N, Channel 1 data (space-separated samples)
        - Line 1: Frame N, Channel 2 data
        - Line 2: Frame N, Channel 3 data
        - Line 3: Frame N, Channel 4 data
        
        Args:
            frame_idx: Frame sequence number.
            channels: List of 4 channel lists, each containing integer samples.
        
        Returns:
            True if successful, False otherwise.
        """
        if self.state not in (self.STATE_ACTIVE_CONTINUOUS, self.STATE_ACTIVE_TIMED) or not self.file:
            raise RecordingError("Recording service not active")
        
        if len(channels) != 4:
            raise RecordingError(f"Expected 4 channels, got {len(channels)}")
        
        try:
            for ch_idx, data in enumerate(channels):
                # Format: "F000123 CH1 val1 val2 val3 ..."
                prefix = f"F{frame_idx:06d} CH{ch_idx + 1}"
                values_str = " ".join(str(int(v)) for v in data)
                line = f"{prefix} {values_str}\n"
                self.file.write(line)
            
            self.frame_count += 1

            if self.state == self.STATE_ACTIVE_TIMED:
                self.timed_written_frames += 1
                if self.timed_written_frames >= self.timed_target_frames:
                    self.is_active = False
                    self.state = self.STATE_PAUSED_PENDING_SAVE

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
            self.state = self.STATE_IDLE
            self.timed_target_frames = 0
            self.timed_written_frames = 0
            return True
        except Exception as e:
            raise RecordingError(f"Error closing file: {e}")
    
    def get_frame_count(self) -> int:
        """Return number of frames written in current session."""
        return self.frame_count

    def get_state(self) -> str:
        """Return current recording state."""
        return self.state

    def get_remaining_timed_frames(self) -> int:
        """Return remaining frames for the current timed round."""
        if self.state != self.STATE_ACTIVE_TIMED:
            return 0

        remaining = self.timed_target_frames - self.timed_written_frames
        return max(0, remaining)
    
    def __del__(self):
        """Ensure file is closed on object destruction."""
        try:
            if self.file:
                self.file.close()
        except:
            pass
