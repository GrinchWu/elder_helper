"""视觉服务 - 使用Qwen-VL进行屏幕理解"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Optional

import httpx
from loguru import logger
from PIL import Image

from ..config import config


@dataclass
class ScreenElement:
    """屏幕元素"""
    element_type: str = ""           # button, text, input, icon, etc.
    text: str = ""                   # 元素文本
    description: str = ""            # 元素描述
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    confidence: float = 0.0
    is_clickable: bool = False
    is_input: bool = False


@dataclass
class ScreenAnalysis:
    """屏幕分析结果"""
    app_name: str = ""               # 当前应用名称
    screen_type: str = ""            # 屏幕类型 (home, app, dialog, etc.)
    elements: list[ScreenElement] = field(default_factory=list)
    description: str = ""            # 屏幕整体描述
    suggested_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # 安全警告


class VisionService:
    """视觉服务"""
    
    def __init__(self) -> None:
        self._api_url = config.api.qwen_vl_url
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"Vision服务初始化完成，API地址: {self._api_url}")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def capture_screen(self) -> bytes:
        """截取屏幕"""
        try:
            import mss
            
            with mss.mss() as sct:
                # 截取主显示器
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                
                # 转换为PNG
                img = Image.frombytes(
                    "RGB",
                    screenshot.size,
                    screenshot.bgra,
                    "raw",
                    "BGRX",
                )
                
                # 压缩图片以减少传输大小
                img = self._resize_if_needed(img)
                
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                return buffer.getvalue()
                
        except Exception as e:
            logger.error(f"截屏失败: {e}")
            return b""
    
    def _resize_if_needed(
        self,
        img: Image.Image,
        max_size: int = 1920,
    ) -> Image.Image:
        """如果图片太大则缩放"""
        width, height = img.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img
    
    async def analyze_screen(
        self,
        screenshot: bytes,
        user_intent: str = "",
    ) -> ScreenAnalysis:
        """分析屏幕内容"""
        if not self._client:
            raise RuntimeError("Vision服务未初始化")
        
        if not screenshot:
            return ScreenAnalysis()
        
        try:
            # 将图片转为base64
            image_base64 = base64.b64encode(screenshot).decode("utf-8")
            
            # 构建提示词
            prompt = self._build_analysis_prompt(user_intent)
            
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen-vl",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    "max_tokens": 2000,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_analysis_result(content)
            
        except httpx.HTTPError as e:
            logger.error(f"屏幕分析请求失败: {e}")
            return ScreenAnalysis()
    
    def _build_analysis_prompt(self, user_intent: str) -> str:
        """构建分析提示词"""
        base_prompt = """请分析这个屏幕截图，用于帮助老年人操作电脑。

请提供以下信息（JSON格式）：
1. app_name: 当前应用名称
2. screen_type: 屏幕类型（home/app/dialog/settings/browser等）
3. elements: 可交互元素列表，每个元素包含：
   - element_type: 类型（button/text/input/icon/link）
   - text: 元素文本
   - description: 用老年人能理解的语言描述
   - bbox: [x, y, width, height]
   - is_clickable: 是否可点击
   - is_input: 是否是输入框
4. description: 用简单的语言描述当前屏幕
5. suggested_actions: 建议的操作
6. warnings: 安全警告（如发现可疑内容、诈骗信息等）

请特别注意识别：
- 广告和弹窗
- 可能的诈骗信息
- 需要输入敏感信息的地方
- 付款或转账相关内容
"""
        
        if user_intent:
            base_prompt += f"\n用户想要：{user_intent}\n请重点关注与此相关的元素。"
        
        return base_prompt
    
    def _parse_analysis_result(self, content: str) -> ScreenAnalysis:
        """解析分析结果"""
        import json
        
        try:
            # 尝试提取JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                
                elements = []
                for elem_data in data.get("elements", []):
                    bbox = elem_data.get("bbox", [0, 0, 0, 0])
                    elements.append(ScreenElement(
                        element_type=elem_data.get("element_type", ""),
                        text=elem_data.get("text", ""),
                        description=elem_data.get("description", ""),
                        bbox=tuple(bbox) if len(bbox) == 4 else (0, 0, 0, 0),
                        is_clickable=elem_data.get("is_clickable", False),
                        is_input=elem_data.get("is_input", False),
                    ))
                
                return ScreenAnalysis(
                    app_name=data.get("app_name", ""),
                    screen_type=data.get("screen_type", ""),
                    elements=elements,
                    description=data.get("description", ""),
                    suggested_actions=data.get("suggested_actions", []),
                    warnings=data.get("warnings", []),
                )
        except json.JSONDecodeError:
            logger.warning("无法解析屏幕分析结果为JSON")
        
        return ScreenAnalysis(description=content)
    
    async def find_element(
        self,
        screenshot: bytes,
        element_description: str,
    ) -> Optional[ScreenElement]:
        """查找特定元素"""
        analysis = await self.analyze_screen(screenshot, element_description)
        
        # 查找最匹配的元素
        best_match: Optional[ScreenElement] = None
        best_score = 0.0
        
        for element in analysis.elements:
            score = self._calculate_match_score(element, element_description)
            if score > best_score:
                best_score = score
                best_match = element
        
        return best_match if best_score > 0.5 else None
    
    def _calculate_match_score(
        self,
        element: ScreenElement,
        description: str,
    ) -> float:
        """计算元素匹配分数"""
        description_lower = description.lower()
        score = 0.0
        
        if element.text and element.text.lower() in description_lower:
            score += 0.5
        if element.description and description_lower in element.description.lower():
            score += 0.3
        if element.element_type in description_lower:
            score += 0.2
        
        return min(score, 1.0)
    
    async def verify_action_result(
        self,
        before_screenshot: bytes,
        after_screenshot: bytes,
        expected_change: str,
    ) -> tuple[bool, str]:
        """验证操作结果"""
        if not self._client:
            raise RuntimeError("Vision服务未初始化")
        
        try:
            before_b64 = base64.b64encode(before_screenshot).decode("utf-8")
            after_b64 = base64.b64encode(after_screenshot).decode("utf-8")
            
            prompt = f"""比较这两张屏幕截图（前后对比），判断操作是否成功。

预期变化：{expected_change}

请回答：
1. 操作是否成功？（是/否）
2. 实际发生了什么变化？
3. 如果失败，可能的原因是什么？

用简单的语言回答，适合老年人理解。"""
            
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen-vl",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ],
                    "max_tokens": 500,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 简单判断是否成功
            success = "是" in content[:50] or "成功" in content[:50]
            
            return success, content
            
        except httpx.HTTPError as e:
            logger.error(f"验证操作结果失败: {e}")
            return False, "无法验证操作结果"
