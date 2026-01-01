"""安全服务 - 防诈骗、隐私保护"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from ..config import config


class RiskLevel(str, Enum):
    """风险等级"""
    SAFE = "safe"              # 安全
    LOW = "low"                # 低风险
    MEDIUM = "medium"          # 中风险
    HIGH = "high"              # 高风险
    CRITICAL = "critical"      # 严重风险


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    is_safe: bool
    risk_level: RiskLevel = RiskLevel.SAFE
    warnings: list[str] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    suggestions: list[str] = field(default_factory=list)
    
    @classmethod
    def safe(cls) -> SafetyCheckResult:
        return cls(is_safe=True, risk_level=RiskLevel.SAFE)
    
    @classmethod
    def blocked(cls, reason: str) -> SafetyCheckResult:
        return cls(
            is_safe=False,
            risk_level=RiskLevel.CRITICAL,
            blocked_reason=reason,
        )


class SafetyService:
    """安全服务"""
    
    def __init__(self) -> None:
        self._scam_keywords = config.security.scam_keywords
        self._sensitive_operations = config.security.sensitive_operations
        
        # 诈骗模式
        self._scam_patterns = [
            # 冒充公检法
            ("公安局", "法院", "检察院", "传票", "涉嫌犯罪"),
            # 冒充客服
            ("客服", "退款", "订单异常", "账户冻结"),
            # 投资诈骗
            ("高额回报", "稳赚不赔", "内部消息", "投资理财"),
            # 中奖诈骗
            ("中奖", "领奖", "幸运用户", "免费领取"),
            # 刷单诈骗
            ("刷单", "兼职", "日赚", "轻松赚钱"),
            # 冒充亲友
            ("急用钱", "出事了", "不要告诉"),
        ]
        
        # 敏感信息模式
        self._sensitive_info_patterns = [
            "身份证",
            "银行卡",
            "密码",
            "验证码",
            "支付密码",
            "CVV",
            "有效期",
        ]
    
    def check_text_safety(self, text: str) -> SafetyCheckResult:
        """检查文本安全性"""
        if not text:
            return SafetyCheckResult.safe()
        
        text_lower = text.lower()
        warnings: list[str] = []
        risk_level = RiskLevel.SAFE
        
        # 检查诈骗关键词
        for keyword in self._scam_keywords:
            if keyword in text:
                warnings.append(f"检测到可疑词汇：{keyword}")
                risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: list(RiskLevel).index(x))
        
        # 检查诈骗模式
        for pattern in self._scam_patterns:
            matches = sum(1 for word in pattern if word in text)
            if matches >= 2:
                warnings.append(f"检测到可疑诈骗模式")
                risk_level = RiskLevel.HIGH
                break
        
        # 检查敏感信息请求
        for pattern in self._sensitive_info_patterns:
            if pattern in text:
                warnings.append(f"涉及敏感信息：{pattern}")
                risk_level = max(risk_level, RiskLevel.MEDIUM, key=lambda x: list(RiskLevel).index(x))
        
        is_safe = risk_level in (RiskLevel.SAFE, RiskLevel.LOW)
        
        suggestions = []
        if not is_safe:
            suggestions = [
                "请不要向陌生人透露个人信息",
                "如有疑问，请先与家人商量",
                "可以拨打110核实情况",
            ]
        
        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=risk_level,
            warnings=warnings,
            suggestions=suggestions,
        )
    
    def check_operation_safety(self, operation: str) -> SafetyCheckResult:
        """检查操作安全性"""
        warnings: list[str] = []
        suggestions: list[str] = []
        
        # 检查是否是敏感操作
        is_sensitive = any(
            op in operation for op in self._sensitive_operations
        )
        
        if is_sensitive:
            warnings.append(f"这是一个敏感操作，请确认是否继续")
            suggestions = [
                "请仔细确认操作内容",
                "如果不确定，可以先问问家人",
                "不要在不熟悉的网站进行支付",
            ]
            
            return SafetyCheckResult(
                is_safe=True,  # 不阻止，但需要确认
                risk_level=RiskLevel.MEDIUM,
                warnings=warnings,
                suggestions=suggestions,
            )
        
        return SafetyCheckResult.safe()
    
    def check_url_safety(self, url: str) -> SafetyCheckResult:
        """检查URL安全性"""
        if not url:
            return SafetyCheckResult.safe()
        
        warnings: list[str] = []
        risk_level = RiskLevel.SAFE
        
        # 检查是否是HTTP（不安全）
        if url.startswith("http://") and not url.startswith("http://localhost"):
            warnings.append("这个网站没有加密，可能不安全")
            risk_level = RiskLevel.LOW
        
        # 检查可疑域名模式
        suspicious_patterns = [
            "login", "verify", "secure", "account", "update",
            "confirm", "banking", "paypal", "alipay", "wechat",
        ]
        
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            # 检查是否是仿冒网站（域名中包含但不是官方域名）
            if pattern in url_lower:
                # 简单检查是否是官方域名
                official_domains = [
                    "alipay.com", "weixin.qq.com", "wechat.com",
                    "taobao.com", "jd.com", "baidu.com",
                ]
                is_official = any(domain in url_lower for domain in official_domains)
                
                if not is_official:
                    warnings.append(f"这个网站可能是仿冒网站，请小心")
                    risk_level = RiskLevel.MEDIUM
                    break
        
        is_safe = risk_level in (RiskLevel.SAFE, RiskLevel.LOW)
        
        suggestions = []
        if not is_safe:
            suggestions = [
                "请确认这是官方网站",
                "不要在可疑网站输入密码",
                "如有疑问，直接打开官方APP",
            ]
        
        return SafetyCheckResult(
            is_safe=is_safe,
            risk_level=risk_level,
            warnings=warnings,
            suggestions=suggestions,
        )
    
    def check_screen_content(
        self,
        screen_text: str,
        detected_elements: list[str],
    ) -> SafetyCheckResult:
        """检查屏幕内容安全性"""
        # 合并所有文本
        all_text = screen_text + " " + " ".join(detected_elements)
        
        # 进行文本安全检查
        text_result = self.check_text_safety(all_text)
        
        # 额外检查弹窗和广告
        popup_keywords = ["中奖", "恭喜", "领取", "红包", "优惠", "限时"]
        popup_count = sum(1 for kw in popup_keywords if kw in all_text)
        
        if popup_count >= 2:
            text_result.warnings.append("检测到可能的广告弹窗")
            text_result.suggestions.append("这可能是广告，建议关闭")
        
        return text_result
    
    def generate_safety_warning(
        self,
        check_result: SafetyCheckResult,
    ) -> str:
        """生成安全警告语音文本"""
        if check_result.is_safe and not check_result.warnings:
            return ""
        
        parts = []
        
        # 根据风险等级选择语气
        if check_result.risk_level == RiskLevel.CRITICAL:
            parts.append("请注意！这可能是诈骗！")
        elif check_result.risk_level == RiskLevel.HIGH:
            parts.append("请小心，这里有些可疑。")
        elif check_result.risk_level == RiskLevel.MEDIUM:
            parts.append("提醒您注意一下。")
        
        # 添加具体警告
        if check_result.warnings:
            parts.append(check_result.warnings[0])
        
        # 添加建议
        if check_result.suggestions:
            parts.append(check_result.suggestions[0])
        
        return "".join(parts)
    
    def should_require_confirmation(
        self,
        operation: str,
        context: str = "",
    ) -> tuple[bool, str]:
        """判断是否需要用户确认"""
        # 检查操作安全性
        op_result = self.check_operation_safety(operation)
        
        if op_result.risk_level >= RiskLevel.MEDIUM:
            confirmation_message = (
                f"您确定要{operation}吗？"
                f"{'，'.join(op_result.warnings)}"
            )
            return True, confirmation_message
        
        # 检查上下文安全性
        if context:
            ctx_result = self.check_text_safety(context)
            if ctx_result.risk_level >= RiskLevel.MEDIUM:
                return True, f"检测到一些可疑内容，您确定要继续吗？"
        
        return False, ""
