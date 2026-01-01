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
    # Sophnet API 配置 (OpenAI兼容格式)
    sophnet_base_url: str = "https://www.sophnet.com/api/open-apis/v1"
    
    # 纯语言模型
    llm_model: str = field(default_factory=lambda: os.getenv(
        "LLM_MODEL", "Qwen2.5-72B-Instruct"
    ))
    
    # 快速规划模型 (用于ReAct循环中的快速决策)
    planner_model_fast: str = field(default_factory=lambda: os.getenv(
        "PLANNER_MODEL_FAST", "Qwen3-14B"
    ))
    
    # 多模态模型 - 轻量级 (用于页面状态分析)
    vl_model_light: str = field(default_factory=lambda: os.getenv(
        "VL_MODEL_LIGHT", "Qwen2.5-VL-72B-Instruct"
    ))
    
    # 多模态模型 - 重量级 (用于精确元素定位)
    vl_model_heavy: str = field(default_factory=lambda: os.getenv(
        "VL_MODEL_HEAVY", "Qwen3-VL-235B-A22B-Instruct"
    ))
    
    # 兼容旧配置
    vl_model: str = field(default_factory=lambda: os.getenv(
        "VL_MODEL", "Qwen3-VL-235B-A22B-Instruct"
    ))
    
    # 统一 API Key
    api_key: str = field(default_factory=lambda: os.getenv(
        "SOPHNET_API_KEY",
        "CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ"
    ))
    
    # 其他服务
    cosyvoice_url: str = field(default_factory=lambda: os.getenv("COSYVOICE_API_URL", "http://localhost:8003"))
    bge_m3_url: str = field(default_factory=lambda: os.getenv("BGE_M3_API_URL", "http://localhost:8004"))


@dataclass(frozen=True)
class ASRConfig:
    """ASR语音识别配置 - Sophnet WebSocket API"""
    project_id: str = field(default_factory=lambda: os.getenv("ASR_PROJECT_ID", "4EygjiMQCjGljeZ8tFJlZD"))
    easyllm_id: str = field(default_factory=lambda: os.getenv("ASR_EASYLLM_ID", "7asJ6QtG2wmknC3iBH7l4B"))
    api_key: str = field(default_factory=lambda: os.getenv(
        "ASR_API_KEY",
        "CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ"
    ))
    format: str = field(default_factory=lambda: os.getenv("ASR_FORMAT", "pcm"))
    sample_rate: int = field(default_factory=lambda: int(os.getenv("ASR_SAMPLE_RATE", "16000")))
    heartbeat: bool = True


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
    asr: ASRConfig = field(default_factory=ASRConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    log: LogConfig = field(default_factory=LogConfig)


# 全局配置实例
config = AppConfig()
