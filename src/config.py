"""配置管理模块"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class APIConfig:
    """API端点配置"""
    qwen_vl_url: str = field(default_factory=lambda: os.getenv("QWEN_VL_API_URL", "http://localhost:8000/v1"))
    qwen_llm_url: str = field(default_factory=lambda: os.getenv("QWEN_LLM_API_URL", "http://localhost:8001/v1"))
    fun_asr_url: str = field(default_factory=lambda: os.getenv("FUN_ASR_API_URL", "http://localhost:8002"))
    cosyvoice_url: str = field(default_factory=lambda: os.getenv("COSYVOICE_API_URL", "http://localhost:8003"))
    bge_m3_url: str = field(default_factory=lambda: os.getenv("BGE_M3_API_URL", "http://localhost:8004"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("QWEN_API_KEY"))


@dataclass(frozen=True)
class RedisConfig:
    """Redis配置"""
    url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))


@dataclass(frozen=True)
class SecurityConfig:
    """安全配置"""
    max_audio_duration: int = field(default_factory=lambda: int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "60")))
    max_screenshot_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_SCREENSHOT_SIZE_MB", "10")))
    rate_limit_rpm: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "30")))
    
    # 安全关键词黑名单
    scam_keywords: tuple[str, ...] = (
        "转账", "汇款", "验证码", "中奖", "退款", "客服电话",
        "银行卡", "密码", "身份证", "公安局", "法院传票",
        "投资理财", "高额回报", "刷单", "兼职赚钱",
    )
    
    # 敏感操作需要确认
    sensitive_operations: tuple[str, ...] = (
        "支付", "转账", "删除", "卸载", "授权", "登录", "注册",
    )


@dataclass(frozen=True)
class LogConfig:
    """日志配置"""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: Path = field(default_factory=lambda: Path(os.getenv("LOG_FILE", "logs/agent.log")))


@dataclass(frozen=True)
class AppConfig:
    """应用总配置"""
    api: APIConfig = field(default_factory=APIConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    log: LogConfig = field(default_factory=LogConfig)


# 全局配置实例
config = AppConfig()
