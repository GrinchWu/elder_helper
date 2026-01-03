"""任务规划服务 - 使用标准化 Skill Set"""

from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

import httpx
from loguru import logger

from ..config import config
from ..models.intent import Intent
from ..models.action import Action, ActionType
from ..models.task import TaskStep, TaskPlan
from ..models.knowledge import KnowledgeGraph
from .vision_service import ScreenAnalysis

if TYPE_CHECKING:
    from ..knowledge.rag_service import RAGService


class PlannerService:
    """任务规划服务"""
    
    def __init__(self) -> None:
        self._base_url = config.api.sophnet_base_url
        self._api_key = config.api.api_key
        self._model = config.api.llm_model
        self._client: Optional[httpx.AsyncClient] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._rag_service: Optional["RAGService"] = None
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=120.0)
        self._knowledge_graph = KnowledgeGraph()
        logger.info("Planner服务初始化完成")
        logger.info(f"  - API URL: {self._base_url}/chat/completions")
        logger.info(f"  - 模型: {self._model}")
    
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
    
    async def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """调用LLM API
        
        Args:
            system_prompt: 系统提示
            user_prompt: 用户提示
            max_tokens: 最大token数
        """
        if not self._client:
            raise RuntimeError("Planner服务未初始化")
        
        try:
            logger.debug(f"调用LLM API: {self._base_url}/chat/completions, 模型: {self._model}")
            
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json={
                    "model": self._model,
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
        history: list[str] = None,  # <--- 新增参数
    ) -> str:
        """构建规划提示"""
        parts = [f"用户想要：{intent.normalized_text or intent.raw_text}"]
        
        if intent.target_app:
            parts.append(f"目标应用：{intent.target_app}")
        
        if intent.target_contact:
            parts.append(f"目标联系人：{intent.target_contact}")
        
        # --- 新增：插入历史记录，这对于重规划至关重要 ---
        if history:
            parts.append(f"\n═══════════════════════════════════════")
            parts.append(f"【已执行的历史与结果】(请根据此调整计划)")
            parts.append(f"═══════════════════════════════════════")
            for item in history:
                parts.append(f"- {item}")
            parts.append(f"⚠️ 注意：之前的步骤可能失败了，请分析原因并尝试不同的路径。")
        # -----------------------------------------------

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
        
        parts.append("\n请根据【当前屏幕状态】一次性生成完整的操作步骤计划。")
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
