"""任务规划服务 - ReAct风格的规划器（使用标准化 Skill Set）"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

import httpx
from loguru import logger

from ..config import config
from ..models.intent import Intent
from ..models.action import Action, ActionType, ActionStatus
from ..models.task import Task, TaskStep, TaskPlan, TaskStatus
from ..models.knowledge import KnowledgeGraph, OperationGuide
from .vision_service import ScreenAnalysis

if TYPE_CHECKING:
    from ..knowledge.rag_service import RAGService


class PlannerState(str, Enum):
    """规划器状态"""
    THINKING = "thinking"        # 思考中
    ACTING = "acting"            # 执行中
    OBSERVING = "observing"      # 观察中
    REPLANNING = "replanning"    # 重新规划
    COMPLETED = "completed"      # 完成
    FAILED = "failed"            # 失败


@dataclass
class ReActStep:
    """ReAct步骤"""
    thought: str = ""            # 思考
    action: Optional[Action] = None
    observation: str = ""        # 观察结果
    state: PlannerState = PlannerState.THINKING


@dataclass
class PlannerContext:
    """规划器上下文"""
    intent: Intent
    current_screen: Optional[ScreenAnalysis] = None
    knowledge_context: str = ""
    history: list[ReActStep] = field(default_factory=list)
    max_steps: int = 20
    current_step: int = 0


class PlannerService:
    """任务规划服务 - 使用ReAct模式"""
    
    def __init__(self) -> None:
        self._base_url = config.api.sophnet_base_url
        self._api_key = config.api.api_key
        self._model = config.api.llm_model
        self._model_fast = config.api.planner_model_fast  # 快速模型
        self._client: Optional[httpx.AsyncClient] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._rag_service: Optional["RAGService"] = None
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=120.0)
        self._knowledge_graph = KnowledgeGraph()
        logger.info("Planner服务初始化完成")
        logger.info(f"  - API URL: {self._base_url}/chat/completions")
        logger.info(f"  - 主模型: {self._model}")
        logger.info(f"  - 快速模型: {self._model_fast}")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """设置知识图谱（兼容旧接口）"""
        self._knowledge_graph = kg
    
    def set_rag_service(self, rag_service: "RAGService") -> None:
        """设置RAG服务（推荐使用）"""
        self._rag_service = rag_service
        logger.info("Planner已关联RAG服务")
    
    async def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000, use_fast_model: bool = False) -> str:
        """调用LLM API
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            max_tokens: 最大token数
            use_fast_model: 是否使用快速模型(Qwen3-14B)
        """
        if not self._client:
            raise RuntimeError("Planner服务未初始化")
        
        model = self._model_fast if use_fast_model else self._model
        
        try:
            logger.debug(f"调用LLM API: {self._base_url}/chat/completions, 模型: {model}")
            
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"LLM API响应结构: {list(result.keys())}")
            
            # 兼容不同的响应格式
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    msg = choice["message"]
                    # Qwen3 模型可能返回 reasoning_content 或 content
                    if "content" in msg and msg["content"]:
                        content = msg["content"]
                    elif "reasoning_content" in msg and msg["reasoning_content"]:
                        # Qwen3 的推理内容，需要从中提取有用信息
                        content = msg["reasoning_content"]
                    else:
                        logger.warning(f"message中无content: {msg.keys()}")
                        content = str(msg)
                elif "text" in choice:
                    content = choice["text"]
                else:
                    logger.error(f"未知的choice格式: {list(choice.keys())}")
                    content = str(choice)
            elif "content" in result:
                content = result["content"]
            elif "text" in result:
                content = result["text"]
            else:
                logger.error(f"未知的响应格式: {result}")
                content = str(result)
            
            logger.debug(f"LLM响应长度: {len(content)}")
            return content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API HTTP错误: {e.response.status_code}")
            logger.error(f"响应内容: {e.response.text}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"LLM API调用失败: {type(e).__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM API解析失败: {type(e).__name__}: {e}")
            raise
    
    async def create_plan(
        self,
        intent: Intent,
        screen_analysis: Optional[ScreenAnalysis] = None,
    ) -> TaskPlan:
        """创建任务计划"""
        # 获取相关知识
        knowledge_context = await self._get_relevant_knowledge(intent)
        
        # 构建规划提示
        prompt = self._build_planning_prompt(
            intent=intent,
            screen_analysis=screen_analysis,
            knowledge_context=knowledge_context,
        )
        
        try:
            content = await self._call_llm(
                system_prompt=self._get_system_prompt(),
                user_prompt=prompt,
                max_tokens=2000,
            )
            return self._parse_plan(content, intent)
            
        except Exception as e:
            logger.error(f"创建计划失败: {e}")
            return TaskPlan(intent=intent)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示（使用标准化 Skill Set）"""
        return """你是一个帮助老年人操作电脑的AI规划器。

你的任务是将用户的意图分解为简单、清晰的操作步骤。

═══════════════════════════════════════════════════════════════
                    【重要限制】输出动作必须严格限制在以下技能集内
═══════════════════════════════════════════════════════════════

【技能集 Skill Set - 你只能使用以下 12 种原子操作，不允许使用任何其他操作】

┌─────────────────────────────────────────────────────────────┐
│ 鼠标操作                                                      │
├─────────────────────────────────────────────────────────────┤
│ 1. 单击{目标}        - 用鼠标左键点击一次                        │
│ 2. 双击{目标}        - 用鼠标左键快速点击两次                     │
│ 3. 右键单击{目标}    - 用鼠标右键点击一次                        │
│ 4. 拖动{对象}至{目标位置} - 按住鼠标左键拖动到目标位置             │
├─────────────────────────────────────────────────────────────┤
│ 滚动操作                                                      │
├─────────────────────────────────────────────────────────────┤
│ 5. 向上滚动{区域}    - 在指定区域向上滚动鼠标滚轮                 │
│ 6. 向下滚动{区域}    - 在指定区域向下滚动鼠标滚轮                 │
├─────────────────────────────────────────────────────────────┤
│ 键盘操作                                                      │
├─────────────────────────────────────────────────────────────┤
│ 7. 输入{文本内容}    - 用键盘输入指定文字                        │
│ 8. 按下{按键}        - 按下单个按键（如回车键、F2键）             │
│ 9. 组合键{按键组合}  - 同时按下多个按键（如Ctrl+C）               │
├─────────────────────────────────────────────────────────────┤
│ 等待操作                                                      │
├─────────────────────────────────────────────────────────────┤
│ 10. 等待{秒数}       - 等待指定时间（如等待{3秒}）                │
│ 11. 等待{元素}出现   - 等待某个元素出现在屏幕上                   │
├─────────────────────────────────────────────────────────────┤
│ 完成操作                                                      │
├─────────────────────────────────────────────────────────────┤
│ 12. 完成            - 任务已完成，无需更多操作                    │
└─────────────────────────────────────────────────────────────┘

【合法的 skill_type 值 - 只能是以下 12 个之一】
单击、双击、右键单击、拖动、向上滚动、向下滚动、输入、按下、组合键、等待、等待出现、完成

【系统元素集 - {目标}可以使用以下标准元素名称】

┌─────────────────────────────────────────────────────────────┐
│ 窗口控制按钮（位于窗口右上角）                                   │
├─────────────────────────────────────────────────────────────┤
│ • 关闭按钮      - 窗口右上角的 × 按钮                           │
│ • 最小化按钮    - 窗口右上角的 - 按钮                           │
│ • 最大化按钮    - 窗口右上角的 □ 按钮                           │
├─────────────────────────────────────────────────────────────┤
│ 任务栏元素（位于屏幕底部）                                       │
├─────────────────────────────────────────────────────────────┤
│ • 开始按钮      - 屏幕左下角的 Windows 图标                     │
│ • 搜索框        - 开始按钮旁边的搜索输入框                       │
│ • 任务栏        - 屏幕底部的横条                                │
│ • 系统托盘      - 屏幕右下角，显示时间的区域旁边                  │
├─────────────────────────────────────────────────────────────┤
│ 通用对话框按钮                                                  │
├─────────────────────────────────────────────────────────────┤
│ • 确定按钮、取消按钮、是按钮、否按钮、应用按钮                    │
├─────────────────────────────────────────────────────────────┤
│ 通用输入控件                                                    │
├─────────────────────────────────────────────────────────────┤
│ • 文本输入框、密码输入框、下拉菜单、复选框、单选按钮              │
├─────────────────────────────────────────────────────────────┤
│ 导航元素                                                        │
├─────────────────────────────────────────────────────────────┤
│ • 返回按钮、前进按钮、刷新按钮、主页按钮                         │
│ • 滚动条、菜单栏、标题栏、状态栏                                 │
└─────────────────────────────────────────────────────────────┘

【按键名称 - {按键}可以使用以下名称】

• 功能键：回车键、Esc键、Tab键、退格键、删除键、空格键
• 方向键：上箭头、下箭头、左箭头、右箭头
• 修饰键：Ctrl键、Alt键、Shift键、Windows键
• 功能键：F1键、F2键、F3键、F4键、F5键、F11键、F12键
• 其他键：Home键、End键、PageUp键、PageDown键

【常用组合键 - {按键组合}可以使用以下组合】

• Ctrl+C（复制）、Ctrl+V（粘贴）、Ctrl+X（剪切）
• Ctrl+Z（撤销）、Ctrl+Y（重做）、Ctrl+S（保存）
• Ctrl+A（全选）、Ctrl+F（查找）
• Alt+F4（关闭窗口）、Alt+Tab（切换应用）
• Win+D（显示桌面）、Win+E（打开文件资源管理器）
• Win+I（打开设置）、Win+L（锁屏）
• F2（重命名）

═══════════════════════════════════════════════════════════════
                         【严格限制规则】
═══════════════════════════════════════════════════════════════

1. ⛔ 禁止使用技能集以外的任何操作
2. ⛔ 禁止发明新的动作类型（如"点击"应该写成"单击"）
3. ⛔ 禁止使用模糊描述（如"找到某个按钮"、"打开应用"）
4. ⛔ skill_type 只能是以下12个值之一：单击、双击、右键单击、拖动、向上滚动、向下滚动、输入、按下、组合键、等待、等待出现、完成
5. ✅ 每个步骤必须是上述 12 种操作之一
6. ✅ 目标必须具体明确，优先使用系统元素集中的名称
7. ✅ 必须提供 visual_hint 告诉用户目标在屏幕的什么位置
8. ✅ 必须提供 expected_result 说明操作后应该看到什么
9. ✅ 如果任务已经完成或当前屏幕已经是目标状态，使用"完成"操作

═══════════════════════════════════════════════════════════════
                    【重要：根据当前屏幕状态规划】
═══════════════════════════════════════════════════════════════

你必须仔细阅读"当前屏幕"信息，根据实际屏幕状态来规划：
- 如果用户已经在桌面上，不要让用户点击浏览器的返回按钮
- 如果目标应用已经打开，不需要再打开它
- 如果任务已经完成，直接返回"完成"操作
- 只规划从当前状态到目标状态需要的步骤

═══════════════════════════════════════════════════════════════
                           【输出格式】
═══════════════════════════════════════════════════════════════

严格按照以下 JSON 格式输出：
{
    "steps": [
        {
            "step_number": 1,
            "skill_type": "单击/双击/右键单击/拖动/向上滚动/向下滚动/输入/按下/组合键/等待/等待出现/完成",
            "target": "目标元素（使用系统元素名称或具体描述）",
            "target_position": "目标位置（仅拖动操作需要）",
            "text": "输入的文本（仅输入操作需要）",
            "key": "按键名称（仅按下操作需要）",
            "hotkey": "组合键（仅组合键操作需要）",
            "wait_seconds": 0,
            "visual_hint": "视觉提示，告诉用户在屏幕哪里找到目标",
            "expected_result": "操作后应该看到什么",
            "friendly_description": "用老年人能理解的语言描述这一步"
        }
    ]
}

═══════════════════════════════════════════════════════════════
                             【示例】
═══════════════════════════════════════════════════════════════

用户想要：打开微信
当前屏幕：Windows桌面
{
    "steps": [
        {
            "step_number": 1,
            "skill_type": "单击",
            "target": "开始按钮",
            "visual_hint": "屏幕左下角的Windows图标（四个方块组成的图案）",
            "expected_result": "开始菜单弹出，显示应用列表和搜索框",
            "friendly_description": "请点击屏幕左下角的Windows图标（开始按钮）"
        },
        {
            "step_number": 2,
            "skill_type": "输入",
            "text": "微信",
            "visual_hint": "开始菜单弹出后，顶部会自动出现搜索框",
            "expected_result": "搜索结果中显示微信应用图标",
            "friendly_description": "直接用键盘输入"微信"两个字"
        },
        {
            "step_number": 3,
            "skill_type": "单击",
            "target": "微信应用图标",
            "visual_hint": "搜索结果列表中，绿色的微信图标（一个对话气泡的图案）",
            "expected_result": "微信应用启动，显示微信登录或主界面",
            "friendly_description": "点击搜索结果中显示的绿色微信图标"
        }
    ]
}

用户想要：打开微信
当前屏幕：微信主界面已经打开
{
    "steps": [
        {
            "step_number": 1,
            "skill_type": "完成",
            "target": "",
            "visual_hint": "微信已经在屏幕上显示",
            "expected_result": "微信已打开",
            "friendly_description": "微信已经打开了，不需要其他操作"
        }
    ]
}

用户想要：关闭当前窗口
当前屏幕：Windows桌面（没有打开的窗口）
{
    "steps": [
        {
            "step_number": 1,
            "skill_type": "完成",
            "target": "",
            "visual_hint": "当前已经是桌面，没有需要关闭的窗口",
            "expected_result": "无需操作",
            "friendly_description": "您已经在桌面上了，没有需要关闭的窗口"
        }
    ]
}"""
    
    def _build_planning_prompt(
        self,
        intent: Intent,
        screen_analysis: Optional[ScreenAnalysis],
        knowledge_context: str,
    ) -> str:
        """构建规划提示"""
        parts = [f"用户想要：{intent.normalized_text or intent.raw_text}"]
        
        if intent.target_app:
            parts.append(f"目标应用：{intent.target_app}")
        
        if intent.target_contact:
            parts.append(f"目标联系人：{intent.target_contact}")
        
        if screen_analysis:
            parts.append(f"\n═══════════════════════════════════════")
            parts.append(f"【当前屏幕状态 - 请仔细阅读】")
            parts.append(f"═══════════════════════════════════════")
            parts.append(f"屏幕描述：{screen_analysis.description}")
            parts.append(f"当前应用：{screen_analysis.app_name}")
            parts.append(f"页面类型：{screen_analysis.screen_type}")
            
            # 检测是否是桌面状态
            is_desktop = "桌面" in screen_analysis.app_name or "桌面" in screen_analysis.description
            if is_desktop:
                parts.append(f"\n⚠️ 重要：用户当前在Windows桌面上！")
                parts.append(f"   - 不要让用户点击浏览器的返回按钮（桌面上没有浏览器）")
                parts.append(f"   - 不要让用户关闭窗口（桌面上可能没有打开的窗口）")
                parts.append(f"   - 如果用户想打开应用，应该从开始菜单或桌面图标开始")
            
            # 使用建议的操作
            if screen_analysis.suggested_actions:
                parts.append(f"\n建议操作：{', '.join(screen_analysis.suggested_actions)}")
            
            # 如果有警告，提醒规划器
            if screen_analysis.warnings:
                parts.append(f"\n⚠️ 注意：{', '.join(screen_analysis.warnings)}")
            
            parts.append(f"═══════════════════════════════════════")
        
        if knowledge_context:
            parts.append(f"\n参考知识：\n{knowledge_context}")
        
        parts.append("\n请根据【当前屏幕状态】生成操作步骤计划。")
        parts.append("如果任务已经完成或当前屏幕已经是目标状态，请返回 skill_type='完成' 的步骤。")
        
        return "\n".join(parts)
    
    async def _get_relevant_knowledge(self, intent: Intent) -> str:
        """获取相关知识 - 优先使用RAG服务"""
        query = intent.normalized_text or intent.raw_text
        
        # 优先使用 RAG 服务（向量语义检索）
        if self._rag_service:
            try:
                # 使用带查询扩展的检索（支持老年人语言映射）
                rag_result = await self._rag_service.retrieve_with_expansion(
                    query=query,
                    top_k=3,
                    min_score=0.3,  # 降低阈值以获取更多结果
                )
                
                if rag_result.context:
                    logger.debug(f"RAG检索成功，置信度: {rag_result.confidence:.2f}")
                    return rag_result.context
                    
            except Exception as e:
                logger.warning(f"RAG检索失败，回退到知识图谱: {e}")
        
        # 回退到知识图谱的简单关键词匹配
        if not self._knowledge_graph:
            return ""
        
        # 搜索相关操作指南
        guides = self._knowledge_graph.search_guides(query, top_k=3)
        
        if not guides:
            return ""
        
        # 合并指南
        if len(guides) > 1:
            merged = self._knowledge_graph.merge_guides(guides)
            guides = [merged]
        
        # 格式化为上下文
        context_parts = []
        for guide in guides:
            context_parts.append(f"【{guide.title}】")
            for i, step in enumerate(guide.friendly_steps or guide.steps, 1):
                context_parts.append(f"{i}. {step}")
        
        return "\n".join(context_parts)
    
    def _parse_plan(self, content: str, intent: Intent) -> TaskPlan:
        """解析计划（支持标准化 Skill Set 格式，带严格验证）"""
        import json
        
        plan = TaskPlan(intent=intent)
        
        # 合法的 skill_type 列表
        VALID_SKILL_TYPES = {
            "单击", "双击", "右键单击", "拖动",
            "向上滚动", "向下滚动",
            "输入", "按下", "组合键",
            "等待", "等待出现",
            "完成"
        }
        
        try:
            # 提取JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                
                invalid_steps = []  # 记录无效步骤
                
                for step_data in data.get("steps", []):
                    # 解析技能类型
                    skill_type_str = step_data.get("skill_type", "单击")
                    original_skill_type = skill_type_str
                    
                    # 验证 skill_type 是否合法
                    if skill_type_str not in VALID_SKILL_TYPES:
                        logger.warning(f"非法的 skill_type: {skill_type_str}，尝试修正")
                        skill_type_str = self._fix_invalid_skill_type(skill_type_str)
                        if skill_type_str not in VALID_SKILL_TYPES:
                            logger.error(f"无法修正的 skill_type: {original_skill_type}，跳过此步骤")
                            invalid_steps.append(f"步骤{step_data.get('step_number', '?')}: {original_skill_type}")
                            continue
                    
                    action_type = self._skill_type_to_action_type(skill_type_str)
                    
                    # 构建动作
                    action = Action(
                        action_type=action_type,
                        element_description=step_data.get("target", ""),
                        text=step_data.get("text"),
                        key=step_data.get("key"),
                        hotkey=step_data.get("hotkey"),
                        visual_hint=step_data.get("visual_hint", ""),
                    )
                    
                    # 设置滚动方向
                    if skill_type_str == "向上滚动":
                        action.scroll_direction = "up"
                    elif skill_type_str == "向下滚动":
                        action.scroll_direction = "down"
                    
                    # 设置等待时间
                    if skill_type_str == "等待":
                        wait_seconds = step_data.get("wait_seconds", 1)
                        action.wait_ms = int(wait_seconds * 1000)
                    
                    # 构建步骤
                    step = TaskStep(
                        step_number=step_data.get("step_number", len(plan.steps) + 1),
                        description=f"{skill_type_str}{{{step_data.get('target', '')}}}",
                        friendly_instruction=step_data.get("friendly_description", ""),
                        action=action,
                        expected_result=step_data.get("expected_result", ""),
                        error_recovery_hint=step_data.get("error_recovery", ""),
                        visual_hint=step_data.get("visual_hint", ""),
                    )
                    plan.steps.append(step)
                
                # 如果有无效步骤，记录警告
                if invalid_steps:
                    logger.warning(f"计划中有 {len(invalid_steps)} 个无效步骤被跳过: {invalid_steps}")
                    
        except json.JSONDecodeError:
            logger.warning("无法解析计划JSON，尝试文本解析")
            plan = self._parse_plan_from_text(content, intent)
        
        return plan
    
    def _fix_invalid_skill_type(self, skill_type: str) -> str:
        """尝试修正非法的 skill_type"""
        # 常见的错误映射
        fix_mapping = {
            "点击": "单击",
            "左键点击": "单击",
            "左键单击": "单击",
            "鼠标点击": "单击",
            "click": "单击",
            "双击打开": "双击",
            "double_click": "双击",
            "右键点击": "右键单击",
            "右击": "右键单击",
            "right_click": "右键单击",
            "拖拽": "拖动",
            "drag": "拖动",
            "滚动": "向下滚动",
            "scroll": "向下滚动",
            "向上滑动": "向上滚动",
            "向下滑动": "向下滚动",
            "键入": "输入",
            "打字": "输入",
            "type": "输入",
            "按键": "按下",
            "press": "按下",
            "快捷键": "组合键",
            "hotkey": "组合键",
            "等一下": "等待",
            "wait": "等待",
            "done": "完成",
            "结束": "完成",
            "任务完成": "完成",
        }
        
        # 尝试直接映射
        if skill_type.lower() in fix_mapping:
            return fix_mapping[skill_type.lower()]
        
        # 尝试部分匹配
        for wrong, correct in fix_mapping.items():
            if wrong in skill_type.lower():
                return correct
        
        return skill_type
    
    def _skill_type_to_action_type(self, skill_type: str) -> ActionType:
        """将技能类型转换为动作类型"""
        mapping = {
            "单击": ActionType.CLICK,
            "双击": ActionType.DOUBLE_CLICK,
            "右键单击": ActionType.RIGHT_CLICK,
            "拖动": ActionType.DRAG,
            "向上滚动": ActionType.SCROLL,
            "向下滚动": ActionType.SCROLL,
            "输入": ActionType.TYPE,
            "按下": ActionType.KEY_PRESS,
            "组合键": ActionType.HOTKEY,
            "等待": ActionType.WAIT,
            "等待出现": ActionType.WAIT_ELEMENT,
            "完成": ActionType.DONE,
        }
        return mapping.get(skill_type, ActionType.CLICK)
    
    def _parse_plan_from_text(self, content: str, intent: Intent) -> TaskPlan:
        """从文本解析计划"""
        plan = TaskPlan(intent=intent)
        
        lines = content.strip().split("\n")
        step_number = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是步骤行
            if line[0].isdigit() or line.startswith("-") or line.startswith("•"):
                step_number += 1
                # 清理行首的数字和符号
                text = line.lstrip("0123456789.-•) ").strip()
                
                step = TaskStep(
                    step_number=step_number,
                    description=text,
                    friendly_instruction=text,
                    action=Action(
                        action_type=ActionType.CLICK,
                        element_description=text,
                    ),
                )
                plan.steps.append(step)
        
        return plan
    
    async def replan_on_error(
        self,
        task: Task,
        error_description: str,
        current_screen: Optional[ScreenAnalysis] = None,
    ) -> TaskPlan:
        """错误后重新规划"""
        if not self._client:
            logger.warning("Planner客户端未初始化，无法重新规划")
            return TaskPlan(intent=task.intent or Intent())
        
        if not task.plan:
            logger.warning("任务没有计划，尝试创建新计划")
            return await self.create_plan(
                intent=task.intent or Intent(),
                screen_analysis=current_screen,
            )
        
        # 获取已完成的步骤
        completed_steps = [
            s for s in task.plan.steps
            if s.status == ActionStatus.SUCCESS
        ]
        
        prompt = f"""任务执行过程中遇到了问题，需要重新规划。

原始意图：{task.intent.normalized_text if task.intent else "未知"}

已完成的步骤：
{self._format_completed_steps(completed_steps)}

遇到的问题：{error_description}

当前屏幕状态：{current_screen.description if current_screen else "未知"}

请生成新的操作步骤，从当前状态继续完成任务。
如果需要回退，请先说明如何回退到安全状态。"""
        
        try:
            content = await self._call_llm(
                system_prompt=self._get_system_prompt(),
                user_prompt=prompt,
                max_tokens=2000,
            )
            return self._parse_plan(content, task.intent or Intent())
            
        except Exception as e:
            logger.error(f"重新规划失败: {e}")
            return TaskPlan()
    
    def _format_completed_steps(self, steps: list[TaskStep]) -> str:
        """格式化已完成步骤"""
        if not steps:
            return "无"
        
        return "\n".join(
            f"{s.step_number}. {s.description}"
            for s in steps
        )
    
    async def suggest_next_action(
        self,
        context: PlannerContext,
    ) -> ReActStep:
        """ReAct模式：建议下一步动作"""
        # 构建ReAct提示
        prompt = self._build_react_prompt(context)
        
        try:
            # 使用快速模型进行单步决策
            content = await self._call_llm(
                system_prompt=self._get_react_system_prompt(),
                user_prompt=prompt,
                max_tokens=500,
                use_fast_model=True,  # 使用Qwen3-14B快速响应
            )
            return self._parse_react_step(content)
            
        except Exception as e:
            logger.error(f"ReAct步骤生成失败: {e}")
            return ReActStep(thought="无法生成下一步", state=PlannerState.FAILED)
    
    async def plan_next_step(
        self,
        intent: Intent,
        screen_analysis: ScreenAnalysis,
        history: list[str] = None,
    ) -> TaskStep:
        """快速规划下一步操作（混合模式 - 单步规划）
        
        使用Qwen3-14B快速模型，目标响应时间<3s
        使用ReAct风格的提示和解析
        """
        history = history or []
        
        # 构建目标状态信息
        target_state_info = ""
        if intent.target_state:
            target_state_info = f"\n目标完成状态：{intent.target_state}"
        if intent.success_criteria:
            target_state_info += f"\n成功判断条件：{', '.join(intent.success_criteria)}"
        
        # 使用ReAct风格的提示
        prompt = f"""目标：{intent.normalized_text or intent.raw_text}
{target_state_info}

当前屏幕：
- 应用：{screen_analysis.app_name}
- 页面：{screen_analysis.screen_type}
- 描述：{screen_analysis.description}

{"已完成步骤：" + chr(10).join(history) if history else "这是第一步"}

【重要】判断任务是否完成的规则：
1. 只有当前屏幕状态完全符合"目标完成状态"时，才能输出"完成"
2. 如果当前在桌面，而目标是打开某个应用或网站，任务显然未完成
3. 如果目标应用还没打开，必须先打开应用
4. 不要因为"可以完成"就输出完成，要等到"已经完成"

请给出下一步的Thought和Action。"""

        try:
            content = await self._call_llm(
                system_prompt=self._get_react_system_prompt(),
                user_prompt=prompt,
                max_tokens=500,
                use_fast_model=True,  # 使用Qwen3-14B
            )
            
            logger.debug(f"快速规划响应: {content[:300] if content else 'None'}")
            
            # 使用ReAct解析方式
            react_step = self._parse_react_step(content)
            
            logger.info(f"[规划] 解析结果: action_type={react_step.action.action_type.value if react_step.action else 'None'}, "
                       f"target={react_step.action.element_description if react_step.action else 'None'}")
            
            # 额外检查：如果解析出"完成"，验证是否真的完成
            if react_step.action and react_step.action.action_type == ActionType.DONE:
                logger.info(f"[规划] 检测到完成动作，验证任务状态...")
                logger.info(f"[规划] 当前屏幕: {screen_analysis.app_name} / {screen_analysis.screen_type}")
                logger.info(f"[规划] 目标应用: {intent.target_app}, 目标状态: {intent.target_state}")
                
                if not self._verify_task_completion(intent, screen_analysis):
                    logger.warning("[规划] 验证失败：目标状态未达到，生成后备步骤")
                    # 强制返回一个实际操作而不是完成
                    return self._generate_fallback_step(intent, screen_analysis, history)
                else:
                    logger.info("[规划] 验证通过：任务确实完成")
            
            if react_step.action:
                # 生成友好的操作描述（只描述动作，不包含推理过程）
                friendly = self._generate_friendly_instruction(react_step.action)
                
                return TaskStep(
                    step_number=len(history) + 1,
                    description=f"{react_step.action.action_type.value}: {react_step.action.element_description or react_step.action.text or ''}",
                    friendly_instruction=friendly,
                    action=react_step.action,
                )
            else:
                # 如果没有解析出动作，尝试从thought中提取简短描述
                logger.warning(f"未能解析出动作，thought: {react_step.thought[:100] if react_step.thought else 'None'}")
                # 返回一个等待动作，让用户确认
                return TaskStep(
                    step_number=len(history) + 1,
                    description="等待确认",
                    friendly_instruction="请告诉我您想做什么",
                    action=Action(action_type=ActionType.WAIT, wait_ms=3000),
                )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"快速规划HTTP错误: {e.response.status_code} - {e.response.text[:200]}")
        except Exception as e:
            logger.error(f"快速规划失败: {type(e).__name__}: {e}")
        
        # 返回默认等待步骤（而不是完成）
        return TaskStep(
            step_number=1,
            description="等待",
            friendly_instruction="请稍等，正在分析",
            action=Action(action_type=ActionType.WAIT, wait_ms=2000),
        )
    
    def _verify_task_completion(self, intent: Intent, screen_analysis: ScreenAnalysis) -> bool:
        """验证任务是否真的完成
        
        返回 True 表示任务确实完成，可以输出"完成"
        返回 False 表示任务未完成，需要继续执行
        """
        current_app = screen_analysis.app_name.lower()
        current_type = screen_analysis.screen_type.lower()
        current_desc = screen_analysis.description.lower()
        
        # 规则1：如果当前在桌面，且目标不是"回到桌面"，则未完成
        is_on_desktop = "桌面" in current_app or "桌面" in current_type or "desktop" in current_app
        
        if is_on_desktop:
            # 检查目标是否就是回到桌面
            target_text = (intent.normalized_text or intent.raw_text).lower()
            if "桌面" not in target_text and "desktop" not in target_text:
                logger.debug(f"验证失败：当前在桌面，但目标不是桌面 (目标: {target_text})")
                return False
        
        # 规则2：如果有目标应用，检查当前应用是否匹配
        if intent.target_app:
            target_app_lower = intent.target_app.lower()
            
            # 特殊处理：浏览器类应用
            browser_keywords = ["浏览器", "edge", "chrome", "firefox", "360", "browser"]
            target_is_browser = any(kw in target_app_lower for kw in browser_keywords)
            current_is_browser = any(kw in current_app for kw in browser_keywords)
            
            if target_is_browser:
                if not current_is_browser:
                    logger.debug(f"验证失败：目标是浏览器，但当前不是浏览器 (当前: {current_app})")
                    return False
            else:
                # 非浏览器应用，检查名称匹配
                if target_app_lower not in current_app and current_app not in target_app_lower:
                    logger.debug(f"验证失败：目标应用 {target_app_lower} 与当前应用 {current_app} 不匹配")
                    return False
        
        # 规则3：如果有明确的目标状态，检查当前屏幕是否匹配
        if intent.target_state:
            target_state_lower = intent.target_state.lower()
            current_state = f"{current_app} {current_type} {current_desc}"
            
            # 提取目标状态的关键词
            target_keywords = [kw for kw in target_state_lower.split() if len(kw) > 1]
            
            if target_keywords:
                # 检查目标关键词是否出现在当前状态中
                match_count = sum(1 for kw in target_keywords if kw in current_state)
                match_ratio = match_count / len(target_keywords)
                
                if match_ratio < 0.3:  # 至少匹配30%的关键词
                    logger.debug(f"验证失败：目标状态关键词匹配率 {match_ratio:.1%} < 30%")
                    return False
        
        # 规则4：如果有成功条件，检查是否满足
        if intent.success_criteria:
            for criterion in intent.success_criteria:
                criterion_lower = criterion.lower()
                if criterion_lower not in current_desc and criterion_lower not in current_type:
                    logger.debug(f"验证失败：成功条件 '{criterion}' 未满足")
                    return False
        
        logger.debug("验证通过：任务可以标记为完成")
        return True
    
    def _generate_fallback_step(self, intent: Intent, screen_analysis: ScreenAnalysis, history: list[str]) -> TaskStep:
        """生成后备步骤（当错误判断为完成时）"""
        # 如果在桌面，需要打开应用
        if "桌面" in screen_analysis.app_name or "桌面" in screen_analysis.screen_type:
            return TaskStep(
                step_number=len(history) + 1,
                description="单击: 开始按钮",
                friendly_instruction="请点击开始按钮",
                action=Action(action_type=ActionType.CLICK, element_description="开始按钮"),
            )
        
        # 如果有目标应用但还没打开
        if intent.target_app:
            return TaskStep(
                step_number=len(history) + 1,
                description=f"输入: {intent.target_app}",
                friendly_instruction=f"请输入{intent.target_app}",
                action=Action(action_type=ActionType.TYPE, text=intent.target_app),
            )
        
        # 默认等待
        return TaskStep(
            step_number=len(history) + 1,
            description="等待",
            friendly_instruction="请稍等",
            action=Action(action_type=ActionType.WAIT, wait_ms=2000),
        )
    
    def _generate_friendly_instruction(self, action: Action) -> str:
        """根据动作生成友好的操作描述（只描述动作本身）"""
        action_type = action.action_type
        target = action.element_description or ""
        text = action.text or ""
        key = action.key or ""
        hotkey = action.hotkey or ""
        
        if action_type == ActionType.CLICK:
            return f"请点击{target}" if target else "请点击目标"
        elif action_type == ActionType.DOUBLE_CLICK:
            return f"请双击{target}" if target else "请双击目标"
        elif action_type == ActionType.RIGHT_CLICK:
            return f"请右键点击{target}" if target else "请右键点击目标"
        elif action_type == ActionType.TYPE:
            return f'请输入"{text}"' if text else "请输入内容"
        elif action_type == ActionType.KEY_PRESS:
            return f"请按{key}键" if key else "请按键"
        elif action_type == ActionType.HOTKEY:
            return f"请按组合键{hotkey}" if hotkey else "请按组合键"
        elif action_type == ActionType.SCROLL:
            direction = action.scroll_direction or "down"
            dir_text = "向上" if direction == "up" else "向下"
            return f"请{dir_text}滚动{target}" if target else f"请{dir_text}滚动"
        elif action_type == ActionType.DRAG:
            return f"请拖动{target}" if target else "请拖动"
        elif action_type == ActionType.WAIT:
            return "请稍等"
        elif action_type == ActionType.WAIT_ELEMENT:
            return f"请等待{target}出现" if target else "请等待"
        elif action_type == ActionType.DONE:
            return "任务已完成"
        else:
            return f"请执行{action_type.value}操作"
    
    def _get_react_system_prompt(self) -> str:
        """ReAct系统提示（使用标准化 Skill Set）"""
        return """你是一个使用ReAct模式的AI助手，帮助老年人操作电脑。

═══════════════════════════════════════════════════════════════
                    【重要限制】输出动作必须严格限制在以下技能集内
═══════════════════════════════════════════════════════════════

【技能集 Skill Set - 你只能使用以下 12 种原子操作】

1. 单击{目标}        - 用鼠标左键点击一次
2. 双击{目标}        - 用鼠标左键快速点击两次
3. 右键单击{目标}    - 用鼠标右键点击一次
4. 拖动{对象}至{目标位置} - 按住鼠标左键拖动到目标位置
5. 向上滚动{区域}    - 在指定区域向上滚动鼠标滚轮
6. 向下滚动{区域}    - 在指定区域向下滚动鼠标滚轮
7. 输入{文本内容}    - 用键盘输入指定文字
8. 按下{按键}        - 按下单个按键（如回车键、F2键）
9. 组合键{按键组合}  - 同时按下多个按键（如Ctrl+C）
10. 等待{秒数}       - 等待指定时间
11. 等待{元素}出现   - 等待某个元素出现在屏幕上
12. 完成            - 任务已完成，无需更多操作

【严格限制】
- ⛔ 禁止使用技能集以外的任何操作
- ⛔ 禁止发明新的动作类型（如"点击"应该写成"单击"）
- ✅ 如果任务已经完成，使用"完成"操作

每一步你需要：
1. Thought（思考）：分析当前情况，决定下一步做什么
2. Action（动作）：具体要执行的操作（必须是上述12种之一）
3. 等待Observation（观察）：执行后的结果

输出格式：
Thought: [你的思考]
Action: [skill_type]: [参数]

示例：
Thought: 用户想打开微信，我需要先点击开始按钮
Action: 单击: 开始按钮

Thought: 需要在搜索框输入微信
Action: 输入: 微信

Thought: 微信已经打开了，任务完成
Action: 完成"""
    
    def _build_react_prompt(self, context: PlannerContext) -> str:
        """构建ReAct提示"""
        parts = [f"目标：{context.intent.normalized_text or context.intent.raw_text}"]
        
        if context.current_screen:
            parts.append(f"\n当前屏幕：{context.current_screen.description}")
        
        if context.knowledge_context:
            parts.append(f"\n参考知识：{context.knowledge_context}")
        
        if context.history:
            parts.append("\n历史记录：")
            for step in context.history[-5:]:
                parts.append(f"Thought: {step.thought}")
                if step.action:
                    parts.append(f"Action: {step.action.action_type.value}")
                parts.append(f"Observation: {step.observation}")
        
        parts.append("\n请给出下一步的Thought和Action。")
        
        return "\n".join(parts)
    
    def _parse_react_step(self, content: str) -> ReActStep:
        """解析ReAct步骤 - 支持从reasoning_content中提取
        
        重要：对"完成"的判断必须非常严格，避免误判
        """
        step = ReActStep()
        
        if not content:
            return step
        
        lines = content.strip().split("\n")
        
        # 先尝试标准格式解析（Action: xxx）
        action_line_found = False
        for line in lines:
            line = line.strip()
            if line.lower().startswith("thought:"):
                step.thought = line[8:].strip()
            elif line.lower().startswith("action:"):
                action_str = line[7:].strip()
                step.action = self._parse_react_action_string(action_str)
                step.state = PlannerState.ACTING
                action_line_found = True
        
        # 如果找到了标准格式的Action行，直接返回（不再做后续模糊匹配）
        if action_line_found and step.action:
            # 设置简短的thought
            if not step.thought or len(step.thought) > 50:
                step.thought = f"{step.action.action_type.value}"
            return step
        
        # 如果没有解析出action，尝试从内容中提取关键操作
        if not step.action:
            import re
            
            # 优先查找标准格式的动作（技能集中的12种操作）
            # 注意：不包含"完成"，完成需要单独严格判断
            skill_patterns = [
                (r'单击[：:\s]*[「"\'【]?([^」"\'】\n，。]{2,20})[」"\'】]?', ActionType.CLICK),
                (r'双击[：:\s]*[「"\'【]?([^」"\'】\n，。]{2,20})[」"\'】]?', ActionType.DOUBLE_CLICK),
                (r'右键单击[：:\s]*[「"\'【]?([^」"\'】\n，。]{2,20})[」"\'】]?', ActionType.RIGHT_CLICK),
                (r'输入[：:\s]*[「"\'【]?([^」"\'】\n，。]{1,50})[」"\'】]?', ActionType.TYPE),
                (r'按下[：:\s]*[「"\'【]?([^」"\'】\n，。]{1,20})[」"\'】]?', ActionType.KEY_PRESS),
                (r'组合键[：:\s]*[「"\'【]?([^」"\'】\n，。]{2,20})[」"\'】]?', ActionType.HOTKEY),
                (r'向上滚动[：:\s]*[「"\'【]?([^」"\'】\n，。]{0,20})[」"\'】]?', ActionType.SCROLL),
                (r'向下滚动[：:\s]*[「"\'【]?([^」"\'】\n，。]{0,20})[」"\'】]?', ActionType.SCROLL),
                (r'等待[：:\s]*(\d+)[秒s]?', ActionType.WAIT),
            ]
            
            for pattern, action_type in skill_patterns:
                match = re.search(pattern, content)
                if match:
                    target = match.group(1).strip() if match.lastindex and match.group(1) else ""
                    
                    step.action = Action(action_type=action_type)
                    
                    if action_type == ActionType.TYPE:
                        step.action.text = target
                    elif action_type == ActionType.KEY_PRESS:
                        step.action.key = target
                    elif action_type == ActionType.HOTKEY:
                        step.action.hotkey = target
                    elif action_type == ActionType.SCROLL:
                        step.action.scroll_direction = "up" if "向上" in pattern else "down"
                        step.action.element_description = target
                    elif action_type == ActionType.WAIT:
                        try:
                            step.action.wait_ms = int(target) * 1000
                        except:
                            step.action.wait_ms = 3000
                    elif action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK):
                        step.action.element_description = target
                    
                    step.thought = f"{action_type.value}: {target}"
                    step.state = PlannerState.ACTING
                    break
            
            # 严格判断"完成"：只有在非常明确的情况下才返回完成
            # 必须是独立的"完成"指令，而不是出现在推理过程中
            if not step.action:
                # 严格的完成模式：必须是明确的完成指令格式
                strict_done_patterns = [
                    r'^完成$',                          # 单独一行只有"完成"
                    r'^Action[：:\s]*完成',              # Action: 完成
                    r'^动作[：:\s]*完成',                # 动作: 完成
                    r'skill_type["\']?\s*[：:]\s*["\']?完成',  # JSON格式
                ]
                
                is_done = False
                for pattern in strict_done_patterns:
                    if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                        is_done = True
                        break
                
                if is_done:
                    step.action = Action(action_type=ActionType.DONE)
                    step.thought = "任务已完成"
                    step.state = PlannerState.COMPLETED
        
        # 设置简短的thought（不超过50字符）
        if not step.thought or len(step.thought) > 50:
            if step.action:
                step.thought = f"{step.action.action_type.value}"
            else:
                step.thought = "分析中"
        
        return step
    
    def _parse_react_action_string(self, action_str: str) -> Action:
        """解析ReAct动作字符串（支持中文 skill_type）"""
        parts = action_str.split(":", 1)
        skill_type_str = parts[0].strip()
        params = parts[1].strip() if len(parts) > 1 else ""
        
        # 尝试修正非法的 skill_type
        skill_type_str = self._fix_invalid_skill_type(skill_type_str)
        
        # 转换为 ActionType
        action_type = self._skill_type_to_action_type(skill_type_str)
        
        action = Action(action_type=action_type)
        
        if action_type == ActionType.TYPE:
            action.text = params
        elif action_type == ActionType.KEY_PRESS:
            action.key = params
        elif action_type == ActionType.HOTKEY:
            action.hotkey = params
        elif action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK, ActionType.RIGHT_CLICK):
            action.element_description = params
        elif action_type == ActionType.SCROLL:
            action.element_description = params
            action.scroll_direction = "up" if "向上" in skill_type_str else "down"
        elif action_type == ActionType.WAIT:
            # 尝试解析等待时间
            import re
            match = re.search(r'(\d+)', params)
            if match:
                action.wait_ms = int(match.group(1)) * 1000
        elif action_type == ActionType.WAIT_ELEMENT:
            action.element_description = params
        
        return action
