"""老年人电脑助手Agent - 核心类"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional
from uuid import UUID, uuid4

from loguru import logger

from ..config import config
from ..models.intent import Intent
from ..models.action import Action, ActionResult, ActionStatus
from ..models.task import Task, TaskStatus, TaskPlan
from ..models.session import Session, SessionState, UserProfile
from ..models.knowledge import KnowledgeGraph
from ..services.asr_service import ASRService, ASRResult, ASRConfig
from ..services.tts_service import TTSService
from ..services.vision_service import VisionService, ScreenAnalysis
from ..services.llm_service import LLMService
from ..services.planner_service import PlannerService
from ..services.safety_service import SafetyService, SafetyCheckResult
from ..services.embedding_service import EmbeddingService
from .executor import ActionExecutor


class AgentState(str, Enum):
    """Agent状态"""
    IDLE = "idle"                    # 空闲
    LISTENING = "listening"          # 监听中
    UNDERSTANDING = "understanding"  # 理解意图中
    PLANNING = "planning"            # 规划中
    EXECUTING = "executing"          # 执行中
    WAITING_USER = "waiting_user"    # 等待用户
    ERROR_RECOVERY = "error_recovery"  # 错误恢复中


@dataclass
class AgentConfig:
    """Agent配置"""
    voice_speed: float = 0.8
    auto_execute: bool = False       # 是否自动执行（否则需要用户确认每一步）
    safety_check_enabled: bool = True
    max_retries: int = 3
    step_timeout_seconds: int = 60


class ElderlyAssistantAgent:
    """老年人电脑助手Agent"""
    
    def __init__(self, agent_config: Optional[AgentConfig] = None) -> None:
        self._config = agent_config or AgentConfig()
        self._state = AgentState.IDLE
        
        # 服务
        self._asr: Optional[ASRService] = None
        self._tts: Optional[TTSService] = None
        self._vision: Optional[VisionService] = None
        self._llm: Optional[LLMService] = None
        self._planner: Optional[PlannerService] = None
        self._safety: Optional[SafetyService] = None
        self._embedding: Optional[EmbeddingService] = None
        self._executor: Optional[ActionExecutor] = None
        
        # 会话
        self._session: Optional[Session] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        
        # 回调
        self._on_state_change: Optional[Callable[[AgentState], None]] = None
        self._on_speak: Optional[Callable[[str], None]] = None
        self._on_step_complete: Optional[Callable[[int, int, bool], None]] = None
    
    async def initialize(self) -> None:
        """初始化Agent"""
        logger.info("正在初始化老年人助手Agent...")
        
        # 初始化所有服务
        # ASR使用配置初始化
        asr_config = ASRConfig(
            project_id=config.asr.project_id,
            easyllm_id=config.asr.easyllm_id,
            api_key=config.asr.api_key,
            format=config.asr.format,
            sample_rate=config.asr.sample_rate,
            heartbeat=config.asr.heartbeat,
        )
        self._asr = ASRService(asr_config)
        self._tts = TTSService()
        self._vision = VisionService()
        self._llm = LLMService()
        self._planner = PlannerService()
        self._safety = SafetyService()
        self._embedding = EmbeddingService()
        self._executor = ActionExecutor()
        
        await asyncio.gather(
            self._asr.initialize(),
            self._tts.initialize(),
            self._vision.initialize(),
            self._llm.initialize(),
            self._planner.initialize(),
            self._embedding.initialize(),
            self._executor.initialize(),
        )
        
        # 初始化知识图谱
        self._knowledge_graph = KnowledgeGraph()
        self._planner.set_knowledge_graph(self._knowledge_graph)
        
        # 创建会话
        self._session = Session()
        
        # 设置TTS语速
        self._tts.set_speed(self._config.voice_speed)
        
        logger.info("Agent初始化完成")
        await self._speak("您好！我是您的电脑助手，有什么可以帮您的？")
    
    async def close(self) -> None:
        """关闭Agent"""
        logger.info("正在关闭Agent...")
        
        if self._asr:
            await self._asr.close()
        if self._tts:
            await self._tts.close()
        if self._vision:
            await self._vision.close()
        if self._llm:
            await self._llm.close()
        if self._planner:
            await self._planner.close()
        if self._embedding:
            await self._embedding.close()
        
        logger.info("Agent已关闭")
    
    def set_user_profile(self, profile: UserProfile) -> None:
        """设置用户画像"""
        if self._session:
            self._session.user_profile = profile
    
    def set_callbacks(
        self,
        on_state_change: Optional[Callable[[AgentState], None]] = None,
        on_speak: Optional[Callable[[str], None]] = None,
        on_step_complete: Optional[Callable[[int, int, bool], None]] = None,
    ) -> None:
        """设置回调函数"""
        self._on_state_change = on_state_change
        self._on_speak = on_speak
        self._on_step_complete = on_step_complete
    
    async def process_voice_input(self, audio_data: bytes) -> None:
        """处理语音输入"""
        if not self._asr:
            return
        
        self._set_state(AgentState.LISTENING)
        
        # 语音识别
        result = await self._asr.recognize_audio(audio_data)
        
        if result.text:
            logger.info(f"识别到: {result.text}")
            await self.process_text_input(result.text)
    
    async def process_text_input(self, text: str) -> None:
        """处理文本输入"""
        if not text.strip():
            return
        
        logger.info(f"处理输入: {text}")
        
        # 添加到对话历史
        if self._session:
            self._session.add_conversation("user", text)
        
        # 安全检查
        if self._config.safety_check_enabled and self._safety:
            safety_result = self._safety.check_text_safety(text)
            if not safety_result.is_safe:
                await self._handle_safety_warning(safety_result)
                return
        
        # 理解意图
        self._set_state(AgentState.UNDERSTANDING)
        intent = await self._understand_intent(text)
        
        if intent.confidence.is_low:
            # 置信度低，需要澄清
            await self._ask_for_clarification(text)
            return
        
        # 创建任务
        await self._create_and_execute_task(intent)
    
    async def _understand_intent(self, text: str) -> Intent:
        """理解用户意图"""
        if not self._llm:
            return Intent(raw_text=text)
        
        profile = self._session.user_profile if self._session else None
        history = self._session.conversation_history if self._session else None
        
        return await self._llm.understand_intent(
            user_input=text,
            user_profile=profile,
            conversation_history=history,
        )
    
    async def _create_and_execute_task(self, intent: Intent) -> None:
        """创建并执行任务"""
        if not self._planner or not self._vision:
            return
        
        # 截取当前屏幕
        screenshot = await self._vision.capture_screen()
        screen_analysis = await self._vision.analyze_screen(
            screenshot,
            intent.normalized_text or intent.raw_text,
        )
        
        # 检查屏幕安全
        if self._config.safety_check_enabled and self._safety:
            screen_safety = self._safety.check_screen_content(
                screen_analysis.description,
                [e.text for e in screen_analysis.elements],
            )
            if screen_safety.warnings:
                await self._speak(self._safety.generate_safety_warning(screen_safety))
        
        # 创建计划
        self._set_state(AgentState.PLANNING)
        plan = await self._planner.create_plan(intent, screen_analysis)
        
        if not plan.steps:
            await self._speak("抱歉，我不太确定该怎么帮您完成这个操作。您能再说详细一点吗？")
            return
        
        # 创建任务
        task = Task(
            intent=intent,
            plan=plan,
            status=TaskStatus.READY,
        )
        
        if self._session:
            self._session.current_task = task
        
        # 告知用户计划
        await self._announce_plan(plan)
        
        # 执行任务
        await self._execute_task(task)
    
    async def _announce_plan(self, plan: TaskPlan) -> None:
        """宣布计划"""
        total = plan.total_steps
        await self._speak(f"好的，我来帮您。一共需要{total}步，我会一步一步告诉您怎么做。")
    
    async def _execute_task(self, task: Task) -> None:
        """执行任务"""
        if not task.plan or not self._executor or not self._vision:
            return
        
        self._set_state(AgentState.EXECUTING)
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        while task.plan.current_step:
            step = task.plan.current_step
            step_num = task.plan.current_step_index + 1
            total = task.plan.total_steps
            
            # 语音指导
            await self._tts.speak_step_instruction(
                step_num,
                total,
                step.friendly_instruction or step.description,
            )
            
            # 等待用户准备
            if not self._config.auto_execute:
                self._set_state(AgentState.WAITING_USER)
                # 在实际实现中，这里应该等待用户确认
                await asyncio.sleep(2)
            
            # 截取执行前的屏幕
            before_screenshot = await self._vision.capture_screen()
            
            # 执行动作
            if step.action:
                # 如果动作没有坐标，尝试从屏幕分析中获取
                if step.action.x is None and step.action.element_description:
                    element = await self._vision.find_element(
                        before_screenshot,
                        step.action.element_description,
                    )
                    if element:
                        # 计算点击中心
                        step.action.x = element.bbox[0] + element.bbox[2] // 2
                        step.action.y = element.bbox[1] + element.bbox[3] // 2
                
                result = await self._executor.execute_with_tolerance(step.action)
                step.status = step.action.status
                
                # 记录动作
                task.record_action(step.action)
            else:
                result = ActionResult.ok()
                step.status = ActionStatus.SUCCESS
            
            # 等待页面响应
            await asyncio.sleep(0.5)
            
            # 截取执行后的屏幕
            after_screenshot = await self._vision.capture_screen()
            
            # 验证结果
            if step.expected_result:
                success, description = await self._vision.verify_action_result(
                    before_screenshot,
                    after_screenshot,
                    step.expected_result,
                )
                
                if not success:
                    # 操作可能失败
                    await self._handle_step_failure(task, step, description)
                    continue
            
            # 成功反馈
            if step.status == ActionStatus.SUCCESS:
                await self._tts.speak_success("好的，这一步完成了。")
                
                if self._on_step_complete:
                    self._on_step_complete(step_num, total, True)
            
            # 前进到下一步
            next_step = task.plan.advance_to_next_step()
            if not next_step:
                break
        
        # 任务完成
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        
        await self._speak("太棒了！任务完成了！您做得很好！")
        
        if self._session:
            self._session.complete_current_task(success=True)
        
        self._set_state(AgentState.IDLE)
    
    async def _handle_step_failure(
        self,
        task: Task,
        step,
        error_description: str,
    ) -> None:
        """处理步骤失败"""
        self._set_state(AgentState.ERROR_RECOVERY)
        
        # 使用无过错假设的语言
        recovery_hint = step.error_recovery_hint or "我们重新试一次"
        
        await self._tts.speak_error_recovery(
            error_description,
            recovery_hint,
        )
        
        # 尝试重新规划
        if task.can_retry() and self._planner and self._vision:
            task.retry_count += 1
            
            screenshot = await self._vision.capture_screen()
            screen_analysis = await self._vision.analyze_screen(screenshot)
            
            new_plan = await self._planner.replan_on_error(
                task,
                error_description,
                screen_analysis,
            )
            
            if new_plan.steps:
                task.plan = new_plan
                await self._speak("我想到了另一个方法，我们试试这个。")
                self._set_state(AgentState.EXECUTING)
            else:
                await self._speak("抱歉，我暂时想不到其他方法。您可以找家人帮忙看看。")
                task.status = TaskStatus.FAILED
                self._set_state(AgentState.IDLE)
        else:
            await self._speak("我们已经试了几次了。建议您找家人帮忙看看。")
            task.status = TaskStatus.FAILED
            self._set_state(AgentState.IDLE)
    
    async def _handle_safety_warning(self, safety_result: SafetyCheckResult) -> None:
        """处理安全警告"""
        warning_text = self._safety.generate_safety_warning(safety_result) if self._safety else ""
        
        if warning_text:
            await self._speak(warning_text)
        
        if safety_result.blocked_reason:
            await self._speak("为了您的安全，我不能帮您做这个操作。")
    
    async def _ask_for_clarification(self, original_text: str) -> None:
        """请求澄清"""
        await self._speak(
            f'您说的是"{original_text}"对吗？'
            "我不太确定您想做什么，能再说详细一点吗？"
        )
        self._set_state(AgentState.LISTENING)
    
    async def _speak(self, text: str) -> None:
        """语音输出"""
        logger.info(f"[语音] {text}")
        
        if self._on_speak:
            self._on_speak(text)
        
        if self._tts:
            await self._tts.speak(text)
        
        if self._session:
            self._session.add_conversation("assistant", text)
    
    def _set_state(self, state: AgentState) -> None:
        """设置状态"""
        old_state = self._state
        self._state = state
        
        if old_state != state:
            logger.debug(f"状态变化: {old_state} -> {state}")
            if self._on_state_change:
                self._on_state_change(state)
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @property
    def session(self) -> Optional[Session]:
        return self._session
