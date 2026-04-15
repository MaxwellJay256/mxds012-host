"""
Recording file utilities: validation, path construction, and error handling.
"""
import os
from pathlib import Path
from typing import Tuple


class RecordingError(Exception):
    """Recording service exceptions."""
    pass


def validateFilename(filename: str) -> Tuple[bool, str]:
    """
    Validate filename for legality and safety.
    
    Args:
        filename: Proposed filename (with or without .txt extension).
    
    Returns:
        Tuple of (is_valid: bool, error_message: str).
        If valid, error_message is empty string.
    """
    if not filename or not filename.strip():
        return False, "文件名不能为空"
    
    filename = filename.strip()
    
    # Windows forbidden characters
    forbidden_chars = {'<', '>', ':', '"', '/', '\\', '|', '?', '*'}
    found_chars = [c for c in filename if c in forbidden_chars]
    if found_chars:
        return False, f"文件名包含非法字符: {', '.join(set(found_chars))}"
    
    # Reserved Windows names
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
    }
    name_upper = filename.upper()
    for reserved in reserved_names:
        if name_upper == reserved or name_upper.startswith(reserved + '.'):
            return False, f"'{filename}' 是 Windows 保留名称，不能使用"
    
    # Warn if user added .txt extension
    if filename.endswith('.txt') and filename[:-4].endswith('.txt'):
        return False, "文件名不应有重复的 .txt 扩展名"
    
    return True, ""


def ensureOutputDir(directory: str) -> Tuple[bool, str]:
    """
    Check if directory exists, or create it recursively.
    
    Args:
        directory: Directory path.
    
    Returns:
        Tuple of (success: bool, error_message: str).
    """
    if not directory or not directory.strip():
        return False, "输出目录不能为空"
    
    try:
        dir_path = Path(directory)
        if dir_path.exists() and not dir_path.is_dir():
            return False, f"路径已存在但不是目录: {directory}"
        
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return True, ""
    except Exception as e:
        return False, f"无法创建或访问目录: {e}"


def buildOutputPath(output_dir: str, filename: str) -> Tuple[str, str]:
    """
    Build full output file path with .txt extension.
    
    Args:
        output_dir: Directory path.
        filename: Base filename (with or without .txt).
    
    Returns:
        Tuple of (full_path: str, error_message: str).
        If error_message is non-empty, full_path is None.
    """
    # Validate filename
    valid, err_msg = validateFilename(filename)
    if not valid:
        return None, err_msg
    
    # Ensure directory exists
    valid, err_msg = ensureOutputDir(output_dir)
    if not valid:
        return None, err_msg
    
    # Add .txt extension if not present
    if not filename.endswith('.txt'):
        filename = filename + '.txt'
    
    try:
        full_path = str(Path(output_dir) / filename)
        return full_path, ""
    except Exception as e:
        return None, f"无法构建输出路径: {e}"


def checkFileExists(file_path: str) -> bool:
    """Check if file already exists."""
    return Path(file_path).exists()


def safeOpenFile(file_path: str, mode: str = 'w') -> Tuple[object, str]:
    """
    Safely open file for writing, with error handling.
    
    Args:
        file_path: Full file path.
        mode: File open mode (default 'w' = write, overwrites).
    
    Returns:
        Tuple of (file_object, error_message).
        If error_message is non-empty, file_object is None.
    """
    try:
        file_obj = open(file_path, mode, encoding='utf-8')
        return file_obj, ""
    except Exception as e:
        return None, f"无法打开文件 {file_path}: {e}"
