"""用户意图模型"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class IntentType(str, Enum):
    """意图类型枚举"""
    COMMUNICATION = "communication"      # 联系他人
    ENTERTAINMENT = "entertainment"      # 娱乐
    INFORMATION = "information"          # 查询信息
    SHOPPING = "shopping"                # 购物
    PAYMENT = "payment"                  # 支付
    SETTINGS = "settings"                # 设置
    HELP = "help"                        # 求助
    NAVIGATION = "navigation"            # 导航
    UNKNOWN = "unknown"                  # 未知


@dataclass(frozen=True)
class Confidence:
    """置信度"""
    value: float  # 0.0 - 1.0
    
    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"置信度必须在0-1之间，当前值: {self.value}")
    
    @property
    def is_high(self) -> bool:
        return self.value >= 0.8
    
    @property
    def is_medium(self) -> bool:
        return 0.5 <= self.value < 0.8
    
    @property
    def is_low(self) -> bool:
        return self.value < 0.5


@dataclass
class Intent:
    """用户意图"""
    id: UUID = field(default_factory=uuid4)
    raw_text: str = ""                              # 原始用户输入
    normalized_text: str = ""                       # 标准化后的文本
    intent_type: IntentType = IntentType.UNKNOWN
    confidence: Confidence = field(default_factory=lambda: Confidence(0.0))
    target_app: Optional[str] = None                # 目标应用
    target_contact: Optional[str] = None            # 目标联系人
    target_state: str = ""                          # 目标完成状态描述（如"浏览器显示新闻网站首页"）
    success_criteria: list[str] = field(default_factory=list)  # 成功判断条件列表
    parameters: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    # 老年人语言映射示例
    ELDERLY_LANGUAGE_MAP: dict[str, str] = field(default_factory=lambda: {
        "手机吃钱": "流量超标或扣费",
        "屏幕上有脏东西关不掉": "悬浮窗广告",
        "我家老二": "第二个孩子",
        "那个绿色的": "微信",
        "那个蓝色的": "支付宝或QQ",
        "打字的地方": "输入框",
        "小红点": "通知提醒",
        "手机发烫": "后台程序过多",
        "手机变慢了": "内存不足或缓存过多",
        "照片找不到了": "相册或文件管理",
    })
    
    def normalize_elderly_language(self, text: str) -> str:
        """将老年人语言转换为标准表述"""
        result = text
        for elderly_term, standard_term in self.ELDERLY_LANGUAGE_MAP.items():
            if elderly_term in result:
                result = result.replace(elderly_term, standard_term)
        return result
