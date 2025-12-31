"""大语言模型服务 - 使用Sophnet API + SimToM意图理解"""

from __future__ import annotations

import asyncio
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional

import httpx
from loguru import logger

from ..config import config
from ..models.intent import Intent, IntentType, Confidence
from ..models.session import UserProfile


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str = ""
    base_url: str = "https://www.sophnet.com/api/open-apis/v1"
    model: str = "Qwen2.5-72B-Instruct"  # 默认使用Qwen2.5-72B
    vl_model: str = "Qwen2.5-VL-72B-Instruct"  # 多模态模型


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    tokens_used: int = 0
    finish_reason: str = ""


class LLMService:
    """大语言模型服务 - 使用Sophnet API"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None) -> None:
        # 使用传入配置或从全局配置获取
        if llm_config:
            self._config = llm_config
        else:
            self._config = LLMConfig(
                api_key=config.api.api_key,
            )
        
        self._client: Optional[httpx.AsyncClient] = None
        self._installed_apps: list[str] = []  # 缓存已安装应用列表
        self._system_prompt = self._build_system_prompt()
    
    def _build_api_url(self) -> str:
        """构建API URL - 使用OpenAI兼容格式"""
        return f"{self._config.base_url}/chat/completions"
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=60.0)
        
        # 获取已安装应用列表
        self._installed_apps = await self._get_installed_apps()
        
        logger.info(f"LLM服务初始化完成")
        logger.info(f"  - API URL: {self._build_api_url()}")
        logger.info(f"  - 模型: {self._config.model}")
        logger.info(f"  - 检测到 {len(self._installed_apps)} 个已安装应用")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _get_installed_apps(self) -> list[str]:
        """获取系统已安装的应用程序列表"""
        apps = []
        
        try:
            # Windows: 从开始菜单和注册表获取应用
            # 方法1: 获取开始菜单快捷方式
            result = subprocess.run(
                ['powershell', '-Command', 
                 'Get-ChildItem "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs" -Recurse -Filter *.lnk | Select-Object -ExpandProperty BaseName'],
                capture_output=True, text=True, timeout=10, encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0:
                apps.extend([app.strip() for app in result.stdout.split('\n') if app.strip()])
            
            # 方法2: 获取用户开始菜单
            result2 = subprocess.run(
                ['powershell', '-Command',
                 'Get-ChildItem "$env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs" -Recurse -Filter *.lnk | Select-Object -ExpandProperty BaseName'],
                capture_output=True, text=True, timeout=10, encoding='utf-8', errors='ignore'
            )
            if result2.returncode == 0:
                apps.extend([app.strip() for app in result2.stdout.split('\n') if app.strip()])
            
            # 去重并过滤
            apps = list(set(apps))
            apps = [app for app in apps if len(app) > 1 and not app.startswith('Uninstall')]
            
            logger.debug(f"检测到已安装应用: {apps[:20]}...")  # 只打印前20个
            
        except Exception as e:
            logger.warning(f"获取已安装应用列表失败: {e}")
            # 返回常见应用作为后备
            apps = [
                "微信", "WeChat", "QQ", "钉钉", "企业微信",
                "Chrome", "Edge", "Firefox", "360浏览器",
                "Word", "Excel", "PowerPoint", "WPS",
                "记事本", "计算器", "画图", "文件资源管理器",
                "支付宝", "淘宝", "京东", "拼多多",
                "爱奇艺", "优酷", "腾讯视频", "哔哩哔哩",
                "网易云音乐", "QQ音乐", "酷狗音乐",
            ]
        
        return apps
    
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

    def _build_simtom_prompt(
        self,
        user_input: str,
        user_profile: Optional[UserProfile] = None,
    ) -> str:
        """
        构建基于SimToM和BDI模型的心智理论提示词
        
        参考论文: "Think Twice: Perspective-Taking Improves Large Language Models' 
        Theory-of-Mind Capabilities" (Wilf et al., 2023)
        
        SimToM两阶段框架:
        Stage 1 - Perspective Taking (视角转换): 过滤信息，只保留该角色知道的内容
        Stage 2 - Question Answering (问题回答): 基于过滤后的视角回答问题
        
        结合BDI模型 (Belief-Desire-Intention):
        - Beliefs: 用户对世界的认知（知道什么、不知道什么、错误认知）
        - Desires: 用户的愿望和目标（表层愿望、深层需求）
        - Intentions: 用户计划采取的行动（基于beliefs和desires推理）
        """
        
        # ========== 构建用户画像上下文 ==========
        profile_context = self._build_profile_context(user_profile)
        
        # ========== 构建系统环境上下文 ==========
        apps_context = self._build_apps_context()
        
        # ========== 构建完整的SimToM提示词 ==========
        prompt = f"""# SimToM心智理论意图理解任务

你的任务是使用心智理论(Theory of Mind)来理解一位老年电脑用户的真实意图。
请严格按照以下两阶段框架进行分析。

---

## 第一阶段：视角转换 (Perspective-Taking)

### 1.1 角色代入
想象你现在就是这位老年用户。你需要"穿上他们的鞋子"，从他们的视角看世界。

**老年用户的典型特征：**
- 年龄：60-80岁，退休人员
- 教育背景：可能没有接受过计算机教育
- 认知特点：晶体智力（经验知识）强，流体智力（处理新信息）下降
- 心理状态：对电脑操作有一定焦虑，害怕"按错键把电脑弄坏"
- 语言习惯：使用生活化、具象化的语言描述技术概念

### 1.2 BDI心智状态建模

**Beliefs（信念）- 用户知道什么、不知道什么、可能的错误认知：**

用户可能知道的（Known）：
- 电脑可以"写东西"（文档编辑）
- 电脑可以"发信"（电子邮件）
- 电脑可以"上网看新闻"（浏览器）
- 应用的视觉特征（颜色、图标形状、位置）
- 家人的称呼和关系

用户可能不知道的（Unknown）：
- 技术术语：浏览器、URL、附件、格式、保存、另存为、云端、同步
- 应用的官方名称和版本
- 文件系统的概念（文件夹、路径、扩展名）
- 操作的可逆性（很多操作可以撤销）

用户可能的错误认知（False Beliefs）：
- "关闭窗口会丢失所有内容"
- "点错按钮会弄坏电脑"
- "电脑里的东西删了就永远找不回来了"
- "红色的提示都是危险的"

**Desires（愿望）- 用户想要达成什么：**

表层愿望（Surface Desire）：用户字面上说想做什么
深层愿望（Deep Desire）：用户真正想要达成的目标
情感需求（Emotional Need）：用户的情感状态（焦虑、困惑、挫败、期待）
社交需求（Social Need）：是否涉及与他人的联系

**Intentions（意图）- 基于Beliefs和Desires推理用户的行动意图：**

目标应用：用户需要使用哪个应用程序
具体操作：需要执行什么操作
操作路径：从当前状态到目标的步骤序列
潜在障碍：用户可能遇到什么困难

---

## 第二阶段：意图推理与回答 (Question-Answering)

基于第一阶段的视角转换，现在从用户的视角回答：用户真正想做什么？

---

## 用户输入
"{user_input}"

## 用户画像信息
{profile_context if profile_context else "暂无详细用户画像，请基于典型老年用户特征推理"}

## 电脑环境信息
{apps_context if apps_context else "暂无应用信息，请基于常见Windows电脑环境推理"}

---

## 推理示例

### 示例1：发邮件场景

**用户说：** "我想给老同事发个信"

**第一阶段：视角转换**

我现在是这位老年用户。我想联系我的老同事，我知道可以用电脑"发信"。

**BDI分析：**

**Beliefs（信念）：**
- Known：我知道电脑可以发信给别人；我知道老同事的名字
- Unknown：我不知道"电子邮件"这个词；我不知道邮箱地址的格式；我不知道Outlook/QQ邮箱的区别
- False Beliefs：我可能以为发信就像寄信一样，需要写地址

**Desires（愿望）：**
- Surface Desire：发一封信
- Deep Desire：与老同事保持联系，可能是正式的沟通（不是微信聊天那种随意的）
- Emotional Need：可能有些紧张，担心操作错误
- Social Need：维系老朋友关系

**Intentions（意图推理）：**
- 目标应用：Outlook / 网页邮箱 / QQ邮箱
- 具体操作：撰写并发送电子邮件
- 操作路径：打开邮箱应用 → 点击"写信/新建邮件" → 填写收件人 → 写内容 → 点击发送
- 潜在障碍：可能不记得收件人邮箱地址；可能不会添加附件；可能找不到"发送"按钮

**第二阶段：意图推理结果**
```json
{{
    "intent_type": "communication",
    "normalized_text": "使用电子邮件客户端给老同事发送邮件",
    "target_app": "Outlook/邮箱",
    "specific_action": "撰写并发送电子邮件",
    "confidence": 0.85,
    "parameters": {{
        "communication_type": "email",
        "formality": "formal"
    }}
}}
```

---

### 示例2：Word使用问题

**用户说：** "我写的东西找不到了，刚才还在的"

**第一阶段：视角转换**

我现在是这位老年用户。我刚才在电脑上写了一些东西，但现在找不到了，我很着急。

**BDI分析：**

**Beliefs（信念）：**
- Known：我刚才确实在写东西；东西"消失"了
- Unknown：我不知道"保存"的概念；我不知道文件存在哪里；我不知道Word有自动恢复功能
- False Beliefs：我可能以为关闭窗口后内容会自动保存；我可能以为内容永远丢失了

**Desires（愿望）：**
- Surface Desire：找回刚才写的内容
- Deep Desire：不想失去自己的劳动成果
- Emotional Need：焦虑、着急、害怕，可能有挫败感
- Social Need：无

**Intentions（意图推理）：**
- 目标应用：Microsoft Word / WPS文字
- 具体操作：恢复未保存的文档 或 查找最近的文件
- 操作路径：
  1. 检查Word是否有自动恢复提示
  2. 查看"文件" → "最近使用的文档"
  3. 搜索临时文件夹
- 潜在障碍：如果没有保存且没有自动恢复，内容可能确实丢失；用户可能不知道去哪里找

**第二阶段：意图推理结果**
```json
{{
    "intent_type": "help",
    "normalized_text": "恢复Word中未保存或意外关闭的文档",
    "target_app": "Microsoft Word/WPS",
    "specific_action": "查找并恢复最近编辑的文档",
    "confidence": 0.80,
    "parameters": {{
        "problem_type": "document_recovery",
        "urgency": "high",
        "user_emotion": "anxious"
    }}
}}
```

---

### 示例3：上网看新闻

**用户说：** "我想看看人民网上有什么新消息"

**第一阶段：视角转换**

我现在是这位老年用户。我想了解国家大事，我知道人民网是个看新闻的地方。

**BDI分析：**

**Beliefs（信念）：**
- Known：人民网是一个可以看新闻的地方；新闻很重要
- Unknown：我不知道"浏览器"是什么；我不知道怎么输入网址；我不知道www.people.com.cn
- False Beliefs：我可能以为人民网是一个"软件"需要安装；我可能分不清浏览器和搜索引擎

**Desires（愿望）：**
- Surface Desire：访问人民网看新闻
- Deep Desire：了解时事新闻，关心国家大事，保持与社会的连接
- Emotional Need：求知欲，想要获取信息
- Social Need：可能想和家人朋友讨论新闻

**Intentions（意图推理）：**
- 目标应用：Microsoft Edge / Chrome / 360浏览器
- 具体操作：打开浏览器，访问人民网
- 操作路径：
  1. 找到并打开浏览器（蓝色的e图标或圆形彩色图标）
  2. 在地址栏或搜索框输入"人民网"
  3. 点击搜索结果中的人民网链接
  4. 浏览新闻内容
- 潜在障碍：可能找不到浏览器图标；可能不会输入；可能被广告页面干扰

**第二阶段：意图推理结果**
```json
{{
    "intent_type": "information",
    "normalized_text": "使用浏览器访问人民网(www.people.com.cn)查看新闻",
    "target_app": "浏览器(Edge/Chrome/360)",
    "specific_action": "打开人民网首页浏览新闻",
    "confidence": 0.90,
    "parameters": {{
        "target_url": "www.people.com.cn",
        "content_type": "news",
        "website_name": "人民网"
    }}
}}
```

---

## 请分析当前用户输入

请严格按照上述两阶段框架分析用户输入"{user_input}"，然后以JSON格式返回结果：

```json
{{
    "simtom_analysis": {{
        "perspective_taking": {{
            "role_description": "我现在是这位老年用户，我的状态是...",
            "beliefs": {{
                "known": ["用户知道的信息"],
                "unknown": ["用户不知道的信息"],
                "false_beliefs": ["用户可能的错误认知"]
            }},
            "desires": {{
                "surface_desire": "表层愿望",
                "deep_desire": "深层愿望",
                "emotional_need": "情感需求",
                "social_need": "社交需求"
            }},
            "intentions": {{
                "target_app": "目标应用",
                "specific_action": "具体操作",
                "operation_path": ["操作步骤1", "操作步骤2"],
                "potential_obstacles": ["潜在障碍1", "潜在障碍2"]
            }}
        }}
    }},
    "intent_type": "communication/entertainment/information/shopping/payment/settings/help/navigation/unknown",
    "normalized_text": "标准化后的技术表述",
    "target_app": "目标应用名称",
    "target_contact": "目标联系人（如有）",
    "specific_action": "具体要执行的操作",
    "confidence": 0.0-1.0,
    "parameters": {{}},
    "clarification_needed": false,
    "clarification_question": "如果需要澄清，这里是要问用户的问题"
}}
```"""
        
        return prompt
    
    def _build_profile_context(self, user_profile: Optional[UserProfile]) -> str:
        """构建用户画像上下文"""
        if not user_profile:
            return ""
        
        parts = []
        
        # 基础信息
        if user_profile.name:
            parts.append(f"姓名：{user_profile.name}")
        if user_profile.age:
            parts.append(f"年龄：{user_profile.age}岁")
        
        # 技术水平
        if hasattr(user_profile, 'tech_level'):
            level = user_profile.tech_level
            if hasattr(level, 'value'):
                level = level.value
            level_desc = {
                "novice": "完全不会电脑，需要手把手教",
                "beginner": "会基本操作（开关机、打字），但容易忘记步骤",
                "intermediate": "能独立完成简单任务，复杂操作需要帮助",
                "advanced": "较熟练，偶尔遇到问题需要帮助",
            }
            parts.append(f"技术水平：{level_desc.get(str(level), str(level))}")
        
        # 家庭成员
        if user_profile.family_mapping:
            family_str = "、".join([f"{k}是{v}" for k, v in user_profile.family_mapping.items()])
            parts.append(f"家庭成员：{family_str}")
        
        # 常用应用
        if user_profile.frequent_apps:
            parts.append(f"常用应用：{', '.join(user_profile.frequent_apps)}")
        
        # 兴趣爱好
        if hasattr(user_profile, 'interests') and user_profile.interests:
            parts.append(f"兴趣爱好：{', '.join(user_profile.interests)}")
        
        # 心理状态
        if hasattr(user_profile, 'anxiety_index'):
            anxiety_desc = "低" if user_profile.anxiety_index < 0.4 else ("中" if user_profile.anxiety_index < 0.7 else "高")
            parts.append(f"焦虑程度：{anxiety_desc}")
        
        if hasattr(user_profile, 'self_efficacy'):
            efficacy_desc = "低" if user_profile.self_efficacy < 0.4 else ("中" if user_profile.self_efficacy < 0.7 else "高")
            parts.append(f"自信程度：{efficacy_desc}")
        
        return "\n".join(parts)
    
    def _build_apps_context(self) -> str:
        """构建应用环境上下文"""
        if not self._installed_apps:
            return ""
        
        # 分类应用
        categories = {
            "办公软件": [],
            "浏览器": [],
            "通讯软件": [],
            "邮件客户端": [],
            "其他常用": [],
        }
        
        for app in self._installed_apps:
            app_lower = app.lower()
            if any(kw in app_lower for kw in ['word', 'excel', 'powerpoint', 'wps', 'office']):
                categories["办公软件"].append(app)
            elif any(kw in app_lower for kw in ['chrome', 'edge', 'firefox', '浏览器', 'browser', '360']):
                categories["浏览器"].append(app)
            elif any(kw in app_lower for kw in ['微信', 'wechat', 'qq', '钉钉', 'teams']):
                categories["通讯软件"].append(app)
            elif any(kw in app_lower for kw in ['outlook', 'mail', '邮件', '邮箱', 'foxmail']):
                categories["邮件客户端"].append(app)
        
        parts = []
        for category, apps in categories.items():
            if apps:
                parts.append(f"{category}：{', '.join(apps[:5])}")
        
        return "\n".join(parts) if parts else f"已安装应用：{', '.join(self._installed_apps[:15])}"

    async def _call_llm(
        self,
        messages: list[dict],
        max_tokens: int = 1000,
        temperature: float = 0.3,
        model: Optional[str] = None,
    ) -> str:
        """调用Sophnet LLM API (OpenAI兼容格式)"""
        if not self._client:
            raise RuntimeError("LLM服务未初始化")
        
        url = self._build_api_url()
        use_model = model or self._config.model
        
        try:
            response = await self._client.post(
                url,
                json={
                    "model": use_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            
            result = response.json()
            
            # OpenAI兼容格式响应
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logger.warning(f"未知的API响应格式: {result}")
                return str(result)
                
        except httpx.HTTPError as e:
            logger.error(f"LLM API调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"响应内容: {e.response.text}")
            raise
    
    async def understand_intent(
        self,
        user_input: str,
        user_profile: Optional[UserProfile] = None,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> Intent:
        """
        使用SimToM理解用户意图
        
        SimToM (Simulation Theory of Mind) 通过模拟老年人的心智状态来理解意图：
        1. 视角转换：站在老年人角度理解表述
        2. 知识过滤：只考虑老年人可能知道的信息
        3. 意图推理：基于有限信息推理真实意图
        """
        if not self._client:
            raise RuntimeError("LLM服务未初始化")
        
        # 构建SimToM提示词
        simtom_prompt = self._build_simtom_prompt(user_input, user_profile)
        
        # 构建消息
        messages = [{"role": "system", "content": self._system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-5:])  # 最近5轮对话
        
        messages.append({"role": "user", "content": simtom_prompt})
        
        try:
            content = await self._call_llm(messages, max_tokens=1500, temperature=0.3)
            return self._parse_intent(user_input, content, user_profile)
            
        except Exception as e:
            logger.error(f"意图理解请求失败: {e}")
            return Intent(raw_text=user_input, intent_type=IntentType.UNKNOWN)
    
    def _parse_intent(
        self,
        raw_text: str,
        llm_response: str,
        user_profile: Optional[UserProfile],
    ) -> Intent:
        """解析意图"""
        intent = Intent(raw_text=raw_text)
        
        try:
            # 提取JSON
            start = llm_response.find("{")
            end = llm_response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(llm_response[start:end])
                
                # 记录SimToM分析过程（用于调试）
                if "simtom_analysis" in data:
                    logger.debug(f"SimToM分析: {data['simtom_analysis']}")
                
                intent_type_str = data.get("intent_type", "unknown")
                try:
                    intent.intent_type = IntentType(intent_type_str)
                except ValueError:
                    intent.intent_type = IntentType.UNKNOWN
                
                intent.normalized_text = data.get("normalized_text", raw_text)
                intent.target_app = data.get("target_app")
                intent.confidence = Confidence(data.get("confidence", 0.5))
                intent.parameters = data.get("parameters", {})
                
                # 添加具体操作到参数
                if data.get("specific_action"):
                    intent.parameters["specific_action"] = data["specific_action"]
                
                # 解析联系人
                target_contact = data.get("target_contact")
                if target_contact and user_profile:
                    # 尝试解析家庭称呼
                    resolved = user_profile.resolve_family_reference(target_contact)
                    intent.target_contact = resolved or target_contact
                else:
                    intent.target_contact = target_contact
                
                # 检查是否需要澄清
                if data.get("clarification_needed"):
                    intent.parameters["clarification_question"] = data.get("clarification_question", "")
                    intent.confidence = Confidence(0.3)  # 降低置信度
                    
        except json.JSONDecodeError:
            logger.warning("无法解析意图JSON，尝试文本解析")
            intent = self._fallback_parse_intent(raw_text, llm_response)
        
        return intent
    
    def _fallback_parse_intent(self, raw_text: str, llm_response: str) -> Intent:
        """后备意图解析（当JSON解析失败时）"""
        intent = Intent(raw_text=raw_text)
        
        # 简单的关键词匹配
        text_lower = raw_text.lower()
        
        if any(kw in text_lower for kw in ["微信", "wechat", "聊天", "视频", "打电话", "联系"]):
            intent.intent_type = IntentType.COMMUNICATION
            intent.target_app = "微信"
        elif any(kw in text_lower for kw in ["看视频", "电视", "电影", "追剧"]):
            intent.intent_type = IntentType.ENTERTAINMENT
        elif any(kw in text_lower for kw in ["买", "购物", "淘宝", "京东"]):
            intent.intent_type = IntentType.SHOPPING
        elif any(kw in text_lower for kw in ["设置", "关闭", "打开", "调"]):
            intent.intent_type = IntentType.SETTINGS
        else:
            intent.intent_type = IntentType.UNKNOWN
        
        intent.normalized_text = raw_text
        intent.confidence = Confidence(0.4)
        
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
            
            content = await self._call_llm(messages, max_tokens=500, temperature=0.7)
            
            return LLMResponse(content=content)
            
        except Exception as e:
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
- "我家老大/老二" = 第一个/第二个孩子
- "那个能买东西的" = 淘宝/京东/拼多多
- "看电视的" = 视频播放软件

请直接返回翻译后的标准表述，不要解释。"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            content = await self._call_llm(messages, max_tokens=100, temperature=0.1)
            return content.strip()
            
        except Exception as e:
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
5. 如果涉及图标，用颜色和形状描述

请直接返回转换后的指令。"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            content = await self._call_llm(messages, max_tokens=200, temperature=0.5)
            return content.strip()
            
        except Exception as e:
            logger.error(f"生成友好指令失败: {e}")
            return technical_instruction
    
    async def refresh_installed_apps(self) -> list[str]:
        """刷新已安装应用列表"""
        self._installed_apps = await self._get_installed_apps()
        return self._installed_apps
    
    @property
    def installed_apps(self) -> list[str]:
        """获取已安装应用列表"""
        return self._installed_apps
