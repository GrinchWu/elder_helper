"""数据验证器"""

from __future__ import annotations

from ..config import config


class ValidationError(Exception):
    """验证错误"""
    pass


def validate_audio_size(audio_data: bytes) -> None:
    """验证音频大小"""
    max_size = config.security.max_audio_duration * 16000 * 2  # 16kHz, 16bit
    
    if len(audio_data) > max_size:
        raise ValidationError(
            f"音频数据过大: {len(audio_data)} bytes, "
            f"最大允许: {max_size} bytes"
        )


def validate_screenshot_size(screenshot_data: bytes) -> None:
    """验证截图大小"""
    max_size = config.security.max_screenshot_size_mb * 1024 * 1024
    
    if len(screenshot_data) > max_size:
        raise ValidationError(
            f"截图数据过大: {len(screenshot_data)} bytes, "
            f"最大允许: {max_size} bytes"
        )


def validate_text_input(text: str, max_length: int = 1000) -> str:
    """验证文本输入"""
    if not text:
        raise ValidationError("文本不能为空")
    
    text = text.strip()
    
    if len(text) > max_length:
        raise ValidationError(f"文本过长: {len(text)}, 最大允许: {max_length}")
    
    return text


def sanitize_user_input(text: str) -> str:
    """清理用户输入"""
    # 移除潜在的危险字符
    dangerous_chars = ["<", ">", "&", '"', "'", "\\", "\x00"]
    
    result = text
    for char in dangerous_chars:
        result = result.replace(char, "")
    
    return result.strip()
