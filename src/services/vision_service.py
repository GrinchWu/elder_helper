"""视觉服务 - 两层架构：页面状态分析 + 精确元素定位"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx
from loguru import logger
from PIL import Image

from ..config import config


class PageStatus(str, Enum):
    """页面状态"""
    NORMAL = "normal"          # 正常
    LOADING = "loading"        # 加载中
    ERROR = "error"            # 错误
    DIALOG = "dialog"          # 弹窗
    LOGIN = "login"            # 登录页
    UNKNOWN = "unknown"        # 未知


@dataclass
class VLConfig:
    """多模态模型配置"""
    api_key: str = ""
    base_url: str = "https://www.sophnet.com/api/open-apis/v1"
    # 轻量级模型 - 用于页面状态分析
    model_light: str = "Qwen2.5-VL-72B-Instruct"
    # 重量级模型 - 用于精确元素定位
    model_heavy: str = "Qwen3-VL-235B-A22B-Instruct"
    # OCR 模型 [新增]
    model_ocr: str = "deepseek-ocr"
    # 兼容旧配置
    model: str = ""


@dataclass
class ScreenStateAnalysis:
    """第一层：页面状态分析结果（不含坐标）"""
    app_name: str = ""                          # 当前应用名称
    screen_state: str = ""                      # 页面状态描述
    page_status: PageStatus = PageStatus.NORMAL # 页面状态枚举
    description: str = ""                       # 详细描述
    available_elements: list[str] = field(default_factory=list)  # 可用元素列表（仅名称）
    element_locations: dict[str, str] = field(default_factory=dict)  # 元素大致位置描述
    suggested_action: str = ""                  # 建议的下一步操作
    warnings: list[str] = field(default_factory=list)  # 警告信息
    # 新增字段
    is_desktop: bool = False                    # 是否是桌面
    has_open_window: bool = False               # 是否有打开的窗口
    foreground_app: str = ""                    # 当前最前面的应用


@dataclass
class ScreenElement:
    """屏幕元素（含精确坐标）"""
    element_type: str = ""           # button, text, input, icon, etc.
    text: str = ""                   # 元素文本
    description: str = ""            # 元素描述
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    confidence: float = 0.0
    is_clickable: bool = False
    is_input: bool = False
    
    def get_center(self) -> tuple[int, int]:
        """获取元素中心点坐标"""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass
class ScreenAnalysis:
    """第二层：完整屏幕分析结果（含坐标）- 兼容旧接口"""
    app_name: str = ""
    screen_type: str = ""
    elements: list[ScreenElement] = field(default_factory=list)
    description: str = ""
    suggested_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class VisionService:
    """视觉服务 - 两层架构"""
    
    def __init__(self, vl_config: Optional[VLConfig] = None) -> None:
        if vl_config:
            self._config = vl_config
            # 如果只传了 model，自动设置为 heavy
            if vl_config.model and not vl_config.model_heavy:
                self._config.model_heavy = vl_config.model
        else:
            self._config = VLConfig(
                api_key=config.api.api_key,
                model_light=config.api.vl_model_light,
                model_heavy=config.api.vl_model_heavy,
                
                # 🟢 [修改这里] 确保参数名是 model_ocr，而不是 ocr_model
                model_ocr="DeepSeek-OCR"  
            )
        
        self._client: Optional[httpx.AsyncClient] = None
        
        self._client: Optional[httpx.AsyncClient] = None
    
    def _build_api_url(self) -> str:
        """构建API URL"""
        return f"{self._config.base_url}/chat/completions"
    
    def _build_ocr_url(self) -> str:
        """构建 OCR API URL [新增]"""
        return f"{self._config.base_url}/ocr"
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=300.0)
        logger.info("Vision服务初始化完成（两层架构 + OCR）")
        logger.info(f"  - VL API URL: {self._build_api_url()}")
        logger.info(f"  - OCR API URL: {self._build_ocr_url()}")
        logger.info(f"  - 轻量模型: {self._config.model_light}")
        logger.info(f"  - 重量模型: {self._config.model_heavy}")
        logger.info(f"  - OCR模型: {self._config.model_ocr}")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None

            # ==================== OCR 识别 [新增] ====================

    async def extract_text_from_bytes(self, image_bytes: bytes) -> str:
        """
        使用 DeepSeek-OCR 模型通过 Chat 接口提取文字
        """
        if not self._client or not image_bytes:
            return ""
            
        try:
            # 1. 编码图片
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # 2. 构造标准 Chat 消息 (符合你的 curl 结构)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "ocr"}, # 简单的 Prompt 触发 OCR
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }]
            
            logger.info(f"正在调用 OCR (Model: {self._config.model_ocr})...")
            
            # 3. 复用 _call_vl_api (它已经处理了 URL、Header 和 Response 解析)
            content = await self._call_vl_api(
                messages,
                model=self._config.model_ocr, # 使用 DeepSeek-OCR
                max_tokens=2000
            )
            
            logger.info(f"OCR 识别成功，长度: {len(content)}")
            return content

        except Exception as e:
            logger.error(f"OCR 调用失败: {e}")
            return ""

    # ==================== 通用方法 ====================
    
    async def _call_vl_api(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 2000,
        max_retries: int = 3,
    ) -> str:
        """调用VL API（带重试机制）"""
        if not self._client:
            raise RuntimeError("Vision服务未初始化")
        
        url = self._build_api_url()
        last_error = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"调用VL API: model={model}, 尝试 {attempt + 1}/{max_retries}")
                
                response = await self._client.post(
                    url,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                    },
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    logger.debug(f"VL API响应长度: {len(content)}")
                    return content
                
                logger.warning(f"未知的API响应格式: {result}")
                return str(result)
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"VL API HTTP错误: {e.response.status_code}")
                logger.error(f"响应内容: {e.response.text}")
                last_error = e
                # HTTP错误不重试（如401、403等）
                break
            except (httpx.ReadError, httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"VL API网络错误 (尝试 {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                last_error = e
                if attempt < max_retries - 1:
                    # 等待后重试
                    import asyncio
                    await asyncio.sleep(2 ** attempt)  # 指数退避: 1s, 2s, 4s
                    continue
            except httpx.HTTPError as e:
                logger.error(f"VL API调用失败: {type(e).__name__}: {e}")
                last_error = e
                break
        
        # 所有重试都失败
        raise last_error or RuntimeError("VL API调用失败")
    
    async def capture_screen(self) -> tuple[bytes, tuple[int, int]]:
        """截取屏幕，返回(图片数据, 原始尺寸)"""
        try:
            import mss
            
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                
                img = Image.frombytes(
                    "RGB",
                    screenshot.size,
                    screenshot.bgra,
                    "raw",
                    "BGRX",
                )
                
                original_size = img.size
                img = self._resize_if_needed(img)
                
                buffer = io.BytesIO()
                img.save(buffer, format="PNG", optimize=True)
                return buffer.getvalue(), original_size
                
        except Exception as e:
            logger.error(f"截屏失败: {e}")
            return b"", (0, 0)
    
    def _resize_if_needed(self, img: Image.Image, max_size: int = 1280) -> Image.Image:
        """如果图片太大则缩放"""
        width, height = img.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            logger.debug(f"缩放图片: {width}x{height} -> {new_size[0]}x{new_size[1]}")
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img
    
    def _get_scale_ratio(self, original_size: tuple[int, int], max_size: int = 1280) -> float:
        """计算缩放比例"""
        width, height = original_size
        if width > max_size or height > max_size:
            return min(max_size / width, max_size / height)
        return 1.0
    
    def _scale_bbox(self, bbox: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
        """将bbox坐标从缩放图片映射回原始屏幕"""
        if scale == 1.0 or scale == 0:
            return bbox
        x, y, w, h = bbox
        return (int(x / scale), int(y / scale), int(w / scale), int(h / scale))

    # ==================== 第一层：页面状态分析 ====================
    
    async def analyze_screen_state(
        self,
        screenshot: bytes,
        user_intent: str = "",
    ) -> ScreenStateAnalysis:
        """
        第一层分析：页面状态分析（轻量级）
        
        使用 Qwen2.5-VL-72B-Instruct，不返回坐标，只分析：
        - 当前是什么应用
        - 页面处于什么状态
        - 有哪些可用元素（仅名称和大致位置描述）
        - 建议的下一步操作
        """
        if not self._client or not screenshot:
            return ScreenStateAnalysis()
        
        try:
            image_base64 = base64.b64encode(screenshot).decode("utf-8")
            prompt = self._build_state_analysis_prompt(user_intent)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            content = await self._call_vl_api(
                messages,
                model=self._config.model_light,
                max_tokens=1500,
            )
            
            return self._parse_state_analysis(content)
            
        except Exception as e:
            logger.error(f"页面状态分析失败: {e}")
            return ScreenStateAnalysis()
    
    def _build_state_analysis_prompt(self, user_intent: str) -> str:
        """构建页面状态分析提示词"""
        prompt = """分析这个屏幕截图，准确描述当前页面状态。只返回JSON，不要其他文字。

【重要】你必须准确识别当前屏幕显示的是什么：
- 如果是 Windows 桌面（显示桌面图标、任务栏、壁纸），app_name 应该是 "Windows桌面"
- 如果是某个应用程序窗口，app_name 应该是该应用的名称
- 如果是浏览器，要区分是浏览器本身还是网页内容

格式：
{
  "app_name": "应用名称（如：Windows桌面、微信、Chrome浏览器、文件资源管理器、系统设置）",
  "screen_state": "页面状态简述（如：桌面、聊天界面、登录页面、主页、文件列表）",
  "page_status": "normal/loading/error/dialog/login",
  "is_desktop": true或false,
  "has_open_window": true或false,
  "foreground_app": "当前最前面的应用名称，如果是桌面则为空",
  "description": "用简单语言描述当前屏幕显示的内容，适合老年人理解",
  "available_elements": ["元素1名称", "元素2名称", "..."],
  "element_locations": {
    "元素名称": "大致位置描述（如：屏幕底部中间、右上角、左侧列表）"
  },
  "suggested_action": "建议用户下一步做什么",
  "warnings": ["如果有安全风险或异常，在这里提醒"]
}

【判断规则】
1. 如果看到桌面壁纸和桌面图标，is_desktop = true
2. 如果有应用窗口覆盖在桌面上，has_open_window = true
3. 如果是全屏应用或最大化窗口，is_desktop = false
4. foreground_app 是当前用户正在操作的应用

注意：
1. 不需要返回精确坐标，只需要描述元素的大致位置
2. available_elements 只列出可交互的元素（按钮、输入框、链接等）
3. 用老年人能理解的语言描述"""
        
        if user_intent:
            prompt += f"\n\n用户想要：{user_intent}\n请重点关注与用户意图相关的元素，并判断当前屏幕状态是否已经满足用户需求。"
        
        return prompt
    
    def _parse_state_analysis(self, content: str) -> ScreenStateAnalysis:
        """解析页面状态分析结果"""
        try:
            json_str = self._extract_json(content)
            if json_str:
                data = json.loads(json_str)
                
                # 解析页面状态枚举
                status_str = data.get("page_status", "normal").lower()
                status_map = {
                    "normal": PageStatus.NORMAL,
                    "loading": PageStatus.LOADING,
                    "error": PageStatus.ERROR,
                    "dialog": PageStatus.DIALOG,
                    "login": PageStatus.LOGIN,
                }
                page_status = status_map.get(status_str, PageStatus.UNKNOWN)
                
                return ScreenStateAnalysis(
                    app_name=data.get("app_name", ""),
                    screen_state=data.get("screen_state", ""),
                    page_status=page_status,
                    description=data.get("description", ""),
                    available_elements=data.get("available_elements", []),
                    element_locations=data.get("element_locations", {}),
                    suggested_action=data.get("suggested_action", ""),
                    warnings=data.get("warnings", []),
                    is_desktop=data.get("is_desktop", False),
                    has_open_window=data.get("has_open_window", False),
                    foreground_app=data.get("foreground_app", ""),
                )
        except Exception as e:
            logger.warning(f"解析页面状态失败: {e}")
        
        return ScreenStateAnalysis(description=content)

    # ==================== 第二层：精确元素定位 ====================
    
    async def locate_element(
        self,
        screenshot: bytes,
        element_description: str,
        original_size: tuple[int, int] = (0, 0),
    ) -> Optional[ScreenElement]:
        """
        第二层分析：精确定位单个元素
        
        使用 Qwen3-VL-235B-A22B-Instruct，返回精确坐标
        仅在需要知道元素具体位置时调用
        """
        if not self._client or not screenshot:
            return None
        
        try:
            image_base64 = base64.b64encode(screenshot).decode("utf-8")
            prompt = self._build_locate_element_prompt(element_description)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            content = await self._call_vl_api(
                messages,
                model=self._config.model_heavy,
                max_tokens=500,
            )
            
            element = self._parse_single_element(content)
            
            # 坐标映射回原始屏幕
            if element and original_size != (0, 0):
                scale = self._get_scale_ratio(original_size)
                if scale != 1.0:
                    element.bbox = self._scale_bbox(element.bbox, scale)
            
            return element
            
        except Exception as e:
            logger.error(f"元素定位失败: {e}")
            return None
    
    def _build_locate_element_prompt(self, element_description: str) -> str:
        """构建元素定位提示词"""
        return f"""在截图中找到"{element_description}"这个元素，返回它的精确位置。

只返回JSON，格式：
{{
  "found": true,
  "element_type": "button/input/link/icon/text",
  "text": "元素上的文字",
  "description": "元素描述",
  "bbox_2d": [左上角x, 左上角y, 右下角x, 右下角y],
  "confidence": 0.95,
  "is_clickable": true,
  "is_input": false
}}

如果找不到，返回：
{{"found": false, "reason": "找不到的原因"}}

注意：bbox_2d 是像素坐标，格式为 [x1, y1, x2, y2]"""
    
    def _parse_single_element(self, content: str) -> Optional[ScreenElement]:
        """解析单个元素定位结果"""
        try:
            json_str = self._extract_json(content)
            if json_str:
                data = json.loads(json_str)
                
                if not data.get("found", False):
                    logger.info(f"未找到元素: {data.get('reason', '未知原因')}")
                    return None
                
                bbox = data.get("bbox_2d", [0, 0, 0, 0])
                bbox_xywh = self._convert_bbox_to_xywh(bbox)
                
                return ScreenElement(
                    element_type=data.get("element_type", ""),
                    text=data.get("text", ""),
                    description=data.get("description", ""),
                    bbox=bbox_xywh,
                    confidence=data.get("confidence", 0.0),
                    is_clickable=data.get("is_clickable", False),
                    is_input=data.get("is_input", False),
                )
        except Exception as e:
            logger.warning(f"解析元素定位结果失败: {e}")
        
        return None

    # ==================== 兼容旧接口 ====================
    
    async def analyze_screen(
        self,
        screenshot: bytes,
        user_intent: str = "",
        original_size: tuple[int, int] = (0, 0),
    ) -> ScreenAnalysis:
        """
        完整屏幕分析（兼容旧接口）
        
        使用重量级模型，返回所有元素的精确坐标
        注意：这个方法较慢，建议优先使用 analyze_screen_state + locate_element
        """
        if not self._client or not screenshot:
            return ScreenAnalysis()
        
        try:
            image_base64 = base64.b64encode(screenshot).decode("utf-8")
            prompt = self._build_full_analysis_prompt(user_intent)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            content = await self._call_vl_api(
                messages,
                model=self._config.model_heavy,
                max_tokens=4000,
            )
            
            analysis = self._parse_full_analysis(content)
            
            # 坐标映射
            if original_size != (0, 0):
                scale = self._get_scale_ratio(original_size)
                if scale != 1.0:
                    for elem in analysis.elements:
                        elem.bbox = self._scale_bbox(elem.bbox, scale)
            
            return analysis
            
        except Exception as e:
            logger.error(f"完整屏幕分析失败: {e}")
            return ScreenAnalysis()
    
    def _build_full_analysis_prompt(self, user_intent: str) -> str:
        """构建完整分析提示词"""
        prompt = """分析截图中的UI元素，只返回JSON，不要其他文字。

格式：
{
  "app_name": "应用名称",
  "screen_type": "app",
  "elements": [
    {"element_type": "button", "text": "文字", "description": "描述", "bbox_2d": [x1, y1, x2, y2], "is_clickable": true, "is_input": false}
  ],
  "description": "屏幕描述",
  "suggested_actions": ["建议操作"],
  "warnings": []
}

说明：
- element_type: button/input/link/menu/tab/icon/text
- bbox_2d: [左上角x, 左上角y, 右下角x, 右下角y] 像素坐标
- 请识别所有可见的按钮、输入框、链接、菜单等可交互元素"""
        
        if user_intent:
            prompt += f"\n用户想要：{user_intent}"
        
        return prompt
    
    def _parse_full_analysis(self, content: str) -> ScreenAnalysis:
        """解析完整分析结果"""
        try:
            json_str = self._extract_json(content)
            if json_str:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    data = self._try_fix_truncated_json(json_str)
                
                if data:
                    elements = []
                    for elem_data in data.get("elements", []):
                        bbox = elem_data.get("bbox_2d") or elem_data.get("bbox", [0, 0, 0, 0])
                        bbox_xywh = self._convert_bbox_to_xywh(bbox)
                        
                        elements.append(ScreenElement(
                            element_type=elem_data.get("element_type", ""),
                            text=elem_data.get("text", ""),
                            description=elem_data.get("description", ""),
                            bbox=bbox_xywh,
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
        except Exception as e:
            logger.warning(f"解析完整分析结果失败: {e}")
        
        return ScreenAnalysis(description=content)

    # ==================== 验证方法 ====================
    
    async def verify_step_completion(
        self,
        before_screenshot: bytes,
        after_screenshot: bytes,
        step_description: str,
        expected_result: str,
    ) -> tuple[bool, str, str]:
        """
        使用 VL 模型判断步骤是否完成
        
        Args:
            before_screenshot: 操作前截图
            after_screenshot: 操作后截图
            step_description: 步骤描述（如"点击发送按钮"）
            expected_result: 预期结果（如"消息发送成功"）
        
        Returns:
            (是否成功, 变化描述, 判断理由)
        """
        if not self._client:
            raise RuntimeError("Vision服务未初始化")
        
        try:
            before_b64 = base64.b64encode(before_screenshot).decode("utf-8")
            after_b64 = base64.b64encode(after_screenshot).decode("utf-8")
            
            prompt = f"""比较这两张截图（第一张是操作前，第二张是操作后），判断用户的操作是否成功完成。

操作描述：{step_description}
预期结果：{expected_result}

请分析：
1. 两张截图之间发生了什么变化？
2. 这些变化是否表明操作成功？
3. 是否达到了预期结果？

只返回JSON，格式：
{{
  "success": true或false,
  "changes": "页面发生的具体变化",
  "matches_expected": true或false,
  "reason": "判断成功或失败的理由，用简单语言描述"
}}

注意：
- 如果页面没有明显变化，可能操作没有生效
- 如果出现错误提示、加载失败等，应判断为失败
- 用老年人能理解的语言描述"""
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            content = await self._call_vl_api(
                messages,
                model=self._config.model_light,  # 使用轻量模型
                max_tokens=800,
            )
            
            return self._parse_step_verification(content)
            
        except Exception as e:
            logger.error(f"步骤验证失败: {e}")
            return False, "", "无法验证操作结果"
    
    def _parse_step_verification(self, content: str) -> tuple[bool, str, str]:
        """解析步骤验证结果"""
        try:
            json_str = self._extract_json(content)
            if json_str:
                data = json.loads(json_str)
                success = data.get("success", False) and data.get("matches_expected", False)
                changes = data.get("changes", "")
                reason = data.get("reason", "")
                return success, changes, reason
        except Exception as e:
            logger.warning(f"解析步骤验证结果失败: {e}")
        
        # 解析失败时，尝试从文本判断
        success = "成功" in content[:100] or "是" in content[:50]
        return success, "", content[:200]
    
    async def verify_action_result(
        self,
        before_screenshot: bytes,
        after_screenshot: bytes,
        expected_change: str,
    ) -> tuple[bool, str]:
        """验证操作结果（使用轻量模型）"""
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
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            content = await self._call_vl_api(
                messages,
                model=self._config.model_light,  # 使用轻量模型
                max_tokens=500,
            )
            
            success = "是" in content[:50] or "成功" in content[:50]
            return success, content
            
        except Exception as e:
            logger.error(f"验证操作结果失败: {e}")
            return False, "无法验证操作结果"
    
    async def find_element(
        self,
        screenshot: bytes,
        element_description: str,
        original_size: tuple[int, int] = (0, 0),
    ) -> Optional[ScreenElement]:
        """查找特定元素（兼容旧接口，内部调用 locate_element）"""
        return await self.locate_element(screenshot, element_description, original_size)

    # ==================== 工具方法 ====================
    
    def _extract_json(self, content: str) -> Optional[str]:
        """从响应中提取JSON字符串"""
        # 移除markdown代码块标记
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                return content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                return content[start:end].strip()
        
        # 直接查找JSON对象
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return content[start:end]
        
        return None
    
    def _convert_bbox_to_xywh(self, bbox: list) -> tuple[int, int, int, int]:
        """
        将bbox转换为统一的 (x, y, width, height) 格式
        
        支持两种输入格式：
        1. [x1, y1, x2, y2] - 左上角和右下角坐标
        2. [x, y, w, h] - 左上角坐标和宽高
        """
        if len(bbox) != 4:
            return (0, 0, 0, 0)
        
        v0, v1, v2, v3 = [int(v) for v in bbox]
        
        # 判断是否为 [x1, y1, x2, y2] 格式
        if v2 > v0 and v3 > v1:
            width = v2 - v0
            height = v3 - v1
            if width > 5 or height > 5:
                return (v0, v1, width, height)
        
        return (v0, v1, v2, v3)
    
    def _try_fix_truncated_json(self, json_str: str) -> Optional[dict]:
        """尝试修复被截断的JSON"""
        elements_start = json_str.find('"elements"')
        if elements_start == -1:
            return None
        
        array_start = json_str.find('[', elements_start)
        if array_start == -1:
            return None
        
        # 找最后一个完整的元素对象
        last_complete = -1
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(array_start, len(json_str)):
            char = json_str[i]
            
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_complete = i
        
        if last_complete > array_start:
            fixed_json = json_str[:last_complete + 1] + ']}'
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass
        
        # 尝试正则提取
        try:
            import re
            app_name_match = re.search(r'"app_name"\s*:\s*"([^"]*)"', json_str)
            screen_type_match = re.search(r'"screen_type"\s*:\s*"([^"]*)"', json_str)
            
            elements = []
            element_pattern = r'\{\s*"element_type"\s*:\s*"([^"]*)"\s*,\s*"text"\s*:\s*"([^"]*)"\s*,\s*"description"\s*:\s*"([^"]*)"\s*,\s*"bbox_2d"\s*:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
            
            for match in re.finditer(element_pattern, json_str):
                elements.append({
                    "element_type": match.group(1),
                    "text": match.group(2),
                    "description": match.group(3),
                    "bbox_2d": [int(match.group(4)), int(match.group(5)), int(match.group(6)), int(match.group(7))],
                    "is_clickable": True,
                    "is_input": False,
                })
            
            if elements:
                logger.info(f"通过正则修复JSON，提取到 {len(elements)} 个元素")
                return {
                    "app_name": app_name_match.group(1) if app_name_match else "",
                    "screen_type": screen_type_match.group(1) if screen_type_match else "",
                    "elements": elements,
                    "description": "",
                    "suggested_actions": [],
                    "warnings": [],
                }
        except Exception as e:
            logger.debug(f"正则修复失败: {e}")
        
        return None
