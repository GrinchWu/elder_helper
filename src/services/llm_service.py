"""大语言模型服务 - 使用Qwen进行意图理解和对话"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from ..config import config
from ..models.intent import Intent, IntentType, Confidence
from ..models.session import UserProfile


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    tokens_used: int = 0
    finish_reason: str = ""


class LLMService:
    """大语言模型服务"""
    
    def __init__(self) -> None:
        self._api_url = config.api.qwen_llm_url
        self._client: Optional[httpx.AsyncClient] = None
        self._system_prompt = self._build_system_prompt()
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"LLM服务初始化完成，API地址: {self._api_url}")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专门帮助老年人使用电脑的AI助手。你的特点是：

1. 耐心和温和：永远不要表现出不耐烦，用温暖的语气交流
2. 简单易懂：避免使用技术术语，用生活化的语言解释
3. 无过错假设：任何问题都是系统的问题，不是用户的错
4. 安全意识：警惕诈骗信息，保护用户隐私和财产安全

你需要：
- 理解老年人的模糊表述（如"手机吃钱"="流量超标"）
- 将复杂操作分解为简单步骤
- 每一步都给予鼓励和确认
- 发现可疑内容时立即警告

回复时请：
- 使用简短的句子
- 一次只说一件事
- 多用"我们"而不是"你"
- 避免使用英文和专业术语"""
    
    async def understand_intent(
        self,
        user_input: str,
        user_profile: Optional[UserProfile] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> Intent:
        """理解用户意图"""
        if not self._client:
            raise RuntimeError("LLM服务未初始化")
        
        # 构建上下文
        context_parts = []
        
        if user_profile:
            context_parts.append(f"用户名称：{user_profile.name}")
            if user_profile.family_mapping:
                context_parts.append(f"家庭成员：{user_profile.family_mapping}")
            if user_profile.frequent_contacts:
                context_parts.append(f"常用联系人：{user_profile.frequent_contacts}")
        
        context = "\n".join(context_parts) if context_parts else ""
        
        prompt = f"""请分析用户的意图。

用户说："{user_input}"

{f"用户信息：{context}" if context else ""}

请以JSON格式返回：
{{
    "intent_type": "communication/entertainment/information/shopping/payment/settings/help/navigation/unknown",
    "normalized_text": "标准化后的表述",
    "target_app": "目标应用名称（如有）",
    "target_contact": "目标联系人（如有）",
    "confidence": 0.0-1.0,
    "parameters": {{}}
}}

注意理解老年人的特殊表述：
- "我家老二" = 第二个孩子
- "那个绿色的" = 微信
- "手机吃钱" = 流量超标
- "屏幕上有脏东西" = 广告弹窗"""
        
        try:
            messages = [{"role": "system", "content": self._system_prompt}]
            
            if conversation_history:
                messages.extend(conversation_history[-5:])  # 最近5轮对话
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_intent(user_input, content, user_profile)
            
        except httpx.HTTPError as e:
            logger.error(f"意图理解请求失败: {e}")
            return Intent(raw_text=user_input, intent_type=IntentType.UNKNOWN)
    
    def _parse_intent(
        self,
        raw_text: str,
        llm_response: str,
        user_profile: Optional[UserProfile],
    ) -> Intent:
        """解析意图"""
        import json
        
        intent = Intent(raw_text=raw_text)
        
        try:
            # 提取JSON
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(llm_response[start:end])
                
                intent_type_str = data.get("intent_type", "unknown")
                try:
                    intent.intent_type = IntentType(intent_type_str)
                except ValueError:
                    intent.intent_type = IntentType.UNKNOWN
                
                intent.normalized_text = data.get("normalized_text", raw_text)
                intent.target_app = data.get("target_app")
                intent.confidence = Confidence(data.get("confidence", 0.5))
                intent.parameters = data.get("parameters", {})
                
                # 解析联系人
                target_contact = data.get("target_contact")
                if target_contact and user_profile:
                    # 尝试解析家庭称呼
                    resolved = user_profile.resolve_family_reference(target_contact)
                    intent.target_contact = resolved or target_contact
                else:
                    intent.target_contact = target_contact
                    
        except json.JSONDecodeError:
            logger.warning("无法解析意图JSON")
        
        return intent
    
    async def generate_response(
        self,
        user_input: str,
        context: str = "",
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> LLMResponse:
        """生成对话响应"""
        if not self._client:
            raise RuntimeError("LLM服务未初始化")
        
        try:
            messages = [{"role": "system", "content": self._system_prompt}]
            
            if conversation_history:
                messages.extend(conversation_history[-5:])
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"当前上下文：{context}"
                })
            
            messages.append({"role": "user", "content": user_input})
            
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": messages,
                    "max_tokens": 500,
                    "temperature": 0.7,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            choice = result["choices"][0]
            
            return LLMResponse(
                content=choice["message"]["content"],
                tokens_used=result.get("usage", {}).get("total_tokens", 0),
                finish_reason=choice.get("finish_reason", ""),
            )
            
        except httpx.HTTPError as e:
            logger.error(f"生成响应失败: {e}")
            return LLMResponse(content="抱歉，我现在有点问题，请稍后再试。")
    
    async def translate_elderly_language(self, text: str) -> str:
        """翻译老年人语言为标准表述"""
        if not self._client:
            raise RuntimeError("LLM服务未初始化")
        
        prompt = f"""请将以下老年人的表述翻译为标准的技术描述。

老年人说："{text}"

常见映射：
- "手机吃钱" = 流量超标或扣费
- "屏幕上有脏东西关不掉" = 悬浮窗广告
- "那个绿色的" = 微信
- "那个蓝色的" = 支付宝或QQ
- "小红点" = 通知提醒
- "手机发烫" = 后台程序过多
- "手机变慢了" = 内存不足

请直接返回翻译后的标准表述，不要解释。"""
        
        try:
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except httpx.HTTPError as e:
            logger.error(f"翻译失败: {e}")
            return text
    
    async def generate_friendly_instruction(
        self,
        technical_instruction: str,
    ) -> str:
        """将技术指令转换为老年人友好的表述"""
        if not self._client:
            raise RuntimeError("LLM服务未初始化")
        
        prompt = f"""请将以下技术操作指令转换为老年人能理解的简单表述。

技术指令："{technical_instruction}"

要求：
1. 使用生活化的语言
2. 避免英文和专业术语
3. 描述要具体，比如"点击左上角的绿色按钮"
4. 语气温和鼓励

请直接返回转换后的指令。"""
        
        try:
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.5,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except httpx.HTTPError as e:
            logger.error(f"生成友好指令失败: {e}")
            return technical_instruction
