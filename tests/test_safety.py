"""安全服务测试"""

import pytest

from src.services.safety_service import SafetyService, RiskLevel


class TestSafetyService:
    """安全服务测试"""
    
    @pytest.fixture
    def safety_service(self):
        return SafetyService()
    
    def test_safe_text(self, safety_service):
        """测试安全文本"""
        result = safety_service.check_text_safety("我想给女儿打个电话")
        assert result.is_safe
        assert result.risk_level == RiskLevel.SAFE
    
    def test_scam_keywords(self, safety_service):
        """测试诈骗关键词"""
        result = safety_service.check_text_safety("请把验证码发给我")
        assert not result.is_safe or len(result.warnings) > 0
    
    def test_scam_pattern(self, safety_service):
        """测试诈骗模式"""
        result = safety_service.check_text_safety(
            "您好，我是公安局的，您涉嫌犯罪，请配合调查"
        )
        assert result.risk_level >= RiskLevel.MEDIUM
        assert len(result.warnings) > 0
    
    def test_sensitive_operation(self, safety_service):
        """测试敏感操作"""
        result = safety_service.check_operation_safety("支付100元")
        assert result.risk_level >= RiskLevel.MEDIUM
        assert len(result.warnings) > 0
    
    def test_safe_operation(self, safety_service):
        """测试安全操作"""
        result = safety_service.check_operation_safety("打开微信")
        assert result.is_safe
        assert result.risk_level == RiskLevel.SAFE
    
    def test_url_safety_http(self, safety_service):
        """测试HTTP URL"""
        result = safety_service.check_url_safety("http://example.com")
        assert result.risk_level >= RiskLevel.LOW
    
    def test_url_safety_suspicious(self, safety_service):
        """测试可疑URL"""
        result = safety_service.check_url_safety("https://alipay-login.fake.com")
        assert result.risk_level >= RiskLevel.MEDIUM
    
    def test_should_require_confirmation(self, safety_service):
        """测试是否需要确认"""
        needs_confirm, message = safety_service.should_require_confirmation("转账给张三")
        assert needs_confirm
        assert message
        
        needs_confirm, message = safety_service.should_require_confirmation("打开相册")
        assert not needs_confirm
    
    def test_generate_safety_warning(self, safety_service):
        """测试生成安全警告"""
        result = safety_service.check_text_safety("请把银行卡密码告诉我")
        warning = safety_service.generate_safety_warning(result)
        
        if result.warnings:
            assert warning  # 有警告时应该生成警告文本
