"""任务规划服务 - ReAct风格的规划器"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx
from loguru import logger

from ..config import config
from ..models.intent import Intent
from ..models.action import Action, ActionType, ActionStatus
from ..models.task import Task, TaskStep, TaskPlan, TaskStatus
from ..models.knowledge import KnowledgeGraph, OperationGuide
from .vision_service import ScreenAnalysis


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
        self._api_url = config.api.qwen_llm_url
        self._client: Optional[httpx.AsyncClient] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=60.0)
        self._knowledge_graph = KnowledgeGraph()
        logger.info("Planner服务初始化完成")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_knowledge_graph(self, kg: KnowledgeGraph) -> None:
        """设置知识图谱"""
        self._knowledge_graph = kg
    
    async def create_plan(
        self,
        intent: Intent,
        screen_analysis: Optional[ScreenAnalysis] = None,
    ) -> TaskPlan:
        """创建任务计划"""
        if not self._client:
            raise RuntimeError("Planner服务未初始化")
        
        # 获取相关知识
        knowledge_context = await self._get_relevant_knowledge(intent)
        
        # 构建规划提示
        prompt = self._build_planning_prompt(
            intent=intent,
            screen_analysis=screen_analysis,
            knowledge_context=knowledge_context,
        )
        
        try:
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_plan(content, intent)
            
        except httpx.HTTPError as e:
            logger.error(f"创建计划失败: {e}")
            return TaskPlan(intent=intent)
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        return """你是一个帮助老年人操作电脑的AI规划器。

你的任务是将用户的意图分解为简单、清晰的操作步骤。

规则：
1. 每个步骤必须是单一、明确的操作
2. 使用老年人能理解的语言描述
3. 考虑可能的错误情况和恢复方法
4. 优先使用最简单的操作路径
5. 避免需要精确点击的操作

输出格式（JSON）：
{
    "steps": [
        {
            "step_number": 1,
            "description": "技术描述",
            "friendly_instruction": "老年人友好的指令",
            "action_type": "click/type/scroll/wait",
            "target_element": "目标元素描述",
            "expected_result": "预期结果",
            "error_recovery": "如果出错怎么办"
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
            parts.append(f"\n当前屏幕：{screen_analysis.description}")
            parts.append(f"当前应用：{screen_analysis.app_name}")
            
            if screen_analysis.elements:
                elements_desc = [
                    f"- {e.description or e.text}"
                    for e in screen_analysis.elements[:10]
                ]
                parts.append(f"可见元素：\n" + "\n".join(elements_desc))
        
        if knowledge_context:
            parts.append(f"\n参考知识：\n{knowledge_context}")
        
        parts.append("\n请生成操作步骤计划。")
        
        return "\n".join(parts)
    
    async def _get_relevant_knowledge(self, intent: Intent) -> str:
        """获取相关知识"""
        if not self._knowledge_graph:
            return ""
        
        # 搜索相关操作指南
        query = intent.normalized_text or intent.raw_text
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
        """解析计划"""
        import json
        
        plan = TaskPlan(intent=intent)
        
        try:
            # 提取JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                
                for step_data in data.get("steps", []):
                    action_type_str = step_data.get("action_type", "click")
                    try:
                        action_type = ActionType(action_type_str)
                    except ValueError:
                        action_type = ActionType.CLICK
                    
                    action = Action(
                        action_type=action_type,
                        element_description=step_data.get("target_element", ""),
                        text=step_data.get("input_text"),
                    )
                    
                    step = TaskStep(
                        step_number=step_data.get("step_number", len(plan.steps) + 1),
                        description=step_data.get("description", ""),
                        friendly_instruction=step_data.get("friendly_instruction", ""),
                        action=action,
                        expected_result=step_data.get("expected_result", ""),
                        error_recovery_hint=step_data.get("error_recovery", ""),
                    )
                    plan.steps.append(step)
                    
        except json.JSONDecodeError:
            logger.warning("无法解析计划JSON，尝试文本解析")
            plan = self._parse_plan_from_text(content, intent)
        
        return plan
    
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
        if not self._client or not task.plan:
            raise RuntimeError("无法重新规划")
        
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
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_plan(content, task.intent or Intent())
            
        except httpx.HTTPError as e:
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
        if not self._client:
            raise RuntimeError("Planner服务未初始化")
        
        # 构建ReAct提示
        prompt = self._build_react_prompt(context)
        
        try:
            response = await self._client.post(
                f"{self._api_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [
                        {"role": "system", "content": self._get_react_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self._parse_react_step(content)
            
        except httpx.HTTPError as e:
            logger.error(f"ReAct步骤生成失败: {e}")
            return ReActStep(thought="无法生成下一步", state=PlannerState.FAILED)
    
    def _get_react_system_prompt(self) -> str:
        """ReAct系统提示"""
        return """你是一个使用ReAct模式的AI助手，帮助老年人操作电脑。

每一步你需要：
1. Thought（思考）：分析当前情况，决定下一步做什么
2. Action（动作）：具体要执行的操作
3. 等待Observation（观察）：执行后的结果

输出格式：
Thought: [你的思考]
Action: [动作类型]: [动作参数]

动作类型包括：
- click: 点击某个元素
- type: 输入文字
- scroll: 滚动页面
- wait: 等待
- done: 任务完成
- ask: 需要询问用户"""
    
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
        """解析ReAct步骤"""
        step = ReActStep()
        
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.lower().startswith("thought:"):
                step.thought = line[8:].strip()
            elif line.lower().startswith("action:"):
                action_str = line[7:].strip()
                step.action = self._parse_action_string(action_str)
                step.state = PlannerState.ACTING
        
        if not step.thought:
            step.thought = content
        
        return step
    
    def _parse_action_string(self, action_str: str) -> Action:
        """解析动作字符串"""
        parts = action_str.split(":", 1)
        action_type_str = parts[0].strip().lower()
        params = parts[1].strip() if len(parts) > 1 else ""
        
        action_type_map = {
            "click": ActionType.CLICK,
            "type": ActionType.TYPE,
            "scroll": ActionType.SCROLL,
            "wait": ActionType.WAIT,
            "done": ActionType.CONFIRM,
            "ask": ActionType.SPEAK,
        }
        
        action_type = action_type_map.get(action_type_str, ActionType.CLICK)
        
        action = Action(action_type=action_type)
        
        if action_type == ActionType.TYPE:
            action.text = params
        elif action_type == ActionType.CLICK:
            action.element_description = params
        
        return action
