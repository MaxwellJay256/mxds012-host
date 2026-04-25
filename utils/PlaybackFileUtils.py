"""Utilities for loading playback text frames from local files."""

from pathlib import Path
from typing import List, Tuple


def validatePlaybackFile(file_path: str) -> Tuple[bool, str]:
    path = Path(file_path)
    if not file_path or not file_path.strip():
        return False, "文件路径不能为空"
    if not path.exists():
        return False, f"文件不存在: {file_path}"
    if path.suffix.lower() != ".txt":
        return False, "只支持 txt 文件"
    if not path.is_file():
        return False, f"路径不是文件: {file_path}"
    return True, ""


def _read_text_file(path: Path) -> str:
    encodings = ("utf-8-sig", "utf-8", "gb18030", "cp936")
    last_error = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return path.read_text(encoding="utf-8")


def parsePlaybackTxtFile(file_path: str) -> Tuple[List[List[List[int]]], int]:
    ok, message = validatePlaybackFile(file_path)
    if not ok:
        raise ValueError(message)

    path = Path(file_path)
    text = _read_text_file(path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("文件内容为空")

    if len(lines) % 4 != 0:
        raise ValueError(f"文件行数必须是 4 的倍数，当前为 {len(lines)}")

    frames: List[List[List[int]]] = []
    sample_depth = 0

    for frame_start in range(0, len(lines), 4):
        frame_channels: List[List[int]] = []
        for channel_offset in range(4):
            line = lines[frame_start + channel_offset]
            tokens = line.split()
            if len(tokens) < 3:
                raise ValueError(f"第 {frame_start + channel_offset + 1} 行格式无效")

            channel_token = tokens[1].upper()
            expected_channel = f"CH{channel_offset + 1}"
            if channel_token != expected_channel:
                raise ValueError(
                    f"第 {frame_start + channel_offset + 1} 行通道标记无效: {channel_token}, 期望 {expected_channel}"
                )

            try:
                values = [int(value) for value in tokens[2:]]
            except ValueError as error:
                raise ValueError(f"第 {frame_start + channel_offset + 1} 行包含非法数字") from error

            if not values:
                raise ValueError(f"第 {frame_start + channel_offset + 1} 行没有采样值")

            if sample_depth == 0:
                sample_depth = len(values)
            elif len(values) != sample_depth:
                raise ValueError(
                    f"采样深度不一致: 第 {frame_start + channel_offset + 1} 行长度为 {len(values)}，期望 {sample_depth}"
                )

            frame_channels.append(values)

        frames.append(frame_channels)

    return frames, sample_depth