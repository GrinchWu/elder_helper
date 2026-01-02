"""任务执行服务 - 管理任务执行流程"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any
from queue import Queue

from loguru import logger

from ..config import config
from ..models.intent import Intent
from ..models.action import Action, ActionType, ActionStatus
from ..models.task import Task, TaskStep, TaskPlan, TaskStatus
from .vision_service import VisionService, ScreenAnalysis, ScreenStateAnalysis, VLConfig, PageStatus
from ..agent.executor import ActionExecutor
from .planner_service import PlannerService


class StepStatus(str, Enum):
    """步骤执行状态"""
    PENDING = "pending"              # 等待执行
    WAITING_USER = "waiting_user"    # 等待用户操作
    VERIFYING = "verifying"          # 验证中
    LOADING = "loading"              # 页面加载中
    SUCCESS = "success"              # 成功
    FAILED = "failed"                # 失败
    REPLANNING = "replanning"        # 重新规划中


class StepCompletionResult(str, Enum):
    """步骤完成判断结果"""
    COMPLETED = "completed"          # 当前步骤完成，继续下一步
    TASK_COMPLETED = "task_completed"  # 整体任务已完成，无需继续
    NEED_RETRY = "need_retry"        # 需要重试当前步骤
    NEED_REPLAN = "need_replan"      # 需要重新规划
    WAITING = "waiting"              # 继续等待用户操作
    TIMEOUT = "timeout"              # 超时，询问用户


class ScreenState(str, Enum):
    """屏幕状态"""
    NORMAL = "normal"                # 正常
    LOADING = "loading"              # 加载中
    ERROR = "error"                  # 错误
    CHANGED = "changed"              # 已变化
    UNCHANGED = "unchanged"          # 未变化


@dataclass
class UserInputEvent:
    """用户输入事件（鼠标/键盘）"""
    event_type: str  # "mouse_click", "mouse_move", "key_press"
    x: int = 0
    y: int = 0
    button: str = ""
    key: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionContext:
    """执行上下文"""
    task: Task
    plan: TaskPlan
    current_step_index: int = 0
    step_status: StepStatus = StepStatus.PENDING
    last_screenshot: bytes = b""
    last_screen_state: Optional[ScreenStateAnalysis] = None  # 使用新的状态分析
    last_screen_analysis: Optional[ScreenAnalysis] = None    # 兼容旧接口
    user_feedback: str = ""
    retry_count: int = 0
    max_retries: int = 3
    loading_wait_time: float = 5.0
    
    # 新增：用于更鲁棒的完成判断
    last_user_input_time: Optional[datetime] = None  # 最后一次用户输入时间
    idle_timeout: float = 30.0  # 无操作超时时间（秒）
    task_goal: str = ""  # 整体任务目标（用于判断是否提前完成）
    
    @property
    def current_step(self) -> Optional[TaskStep]:
        if 0 <= self.current_step_index < len(self.plan.steps):
            return self.plan.steps[self.current_step_index]
        return None
    
    @property
    def is_completed(self) -> bool:
        return self.current_step_index >= len(self.plan.steps)
    
    @property
    def seconds_since_last_input(self) -> float:
        """距离上次用户输入的秒数"""
        if not self.last_user_input_time:
            return 0.0
        return (datetime.now() - self.last_user_input_time).total_seconds()


@dataclass
class MouseClickEvent:
    """鼠标点击事件"""
    x: int
    y: int
    button: str
    timestamp: datetime = field(default_factory=datetime.now)


class InputListener:
    """用户输入监听器（鼠标+键盘）"""
    
    def __init__(self):
        self._mouse_listener = None
        self._keyboard_listener = None
        self._event_queue: Queue[UserInputEvent] = Queue()
        self._is_listening = False
        self._lock = threading.Lock()
    
    def start(self):
        """开始监听"""
        if self._is_listening:
            return
        
        try:
            from pynput import mouse, keyboard
            
            def on_click(x, y, button, pressed):
                if pressed:  # 只记录按下事件
                    event = UserInputEvent(
                        event_type="mouse_click",
                        x=int(x),
                        y=int(y),
                        button=str(button),
                    )
                    self._event_queue.put(event)
                    logger.debug(f"鼠标点击: ({x}, {y}) {button}")
            
            def on_key_press(key):
                try:
                    key_str = key.char if hasattr(key, 'char') else str(key)
                except AttributeError:
                    key_str = str(key)
                
                event = UserInputEvent(
                    event_type="key_press",
                    key=key_str,
                )
                self._event_queue.put(event)
                logger.debug(f"键盘按键: {key_str}")
            
            self._mouse_listener = mouse.Listener(on_click=on_click)
            self._mouse_listener.start()
            
            self._keyboard_listener = keyboard.Listener(on_press=on_key_press)
            self._keyboard_listener.start()
            
            self._is_listening = True
            logger.info("输入监听器已启动（鼠标+键盘）")
            
        except ImportError:
            logger.warning("pynput未安装，输入监听不可用")
        except Exception as e:
            logger.error(f"启动输入监听失败: {e}")
    
    def stop(self):
        """停止监听"""
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        self._is_listening = False
        logger.info("输入监听器已停止")
    
    def get_event(self, timeout: float = None) -> Optional[UserInputEvent]:
        """获取输入事件（阻塞）"""
        try:
            return self._event_queue.get(timeout=timeout)
        except:
            return None
    
    def has_event(self) -> bool:
        """是否有待处理的事件"""
        return not self._event_queue.empty()
    
    def clear(self):
        """清空事件队列"""
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except:
                break


class ExecutorService:
    """任务执行服务"""
    
    def __init__(self):
        self._vision: Optional[VisionService] = None
        self._planner: Optional[PlannerService] = None
        self._input_listener: Optional[InputListener] = None
        self._context: Optional[ExecutionContext] = None
        self._action_executor: Optional[ActionExecutor] = None
        
        # 标记是否使用外部服务
        self._external_vision = False
        self._external_planner = False
        
        # 回调函数
        self._on_step_start: Optional[Callable[[TaskStep], None]] = None
        self._on_step_complete: Optional[Callable[[TaskStep, bool], None]] = None
        self._on_need_replan: Optional[Callable[[str], None]] = None
        self._on_task_complete: Optional[Callable[[Task, bool], None]] = None
        self._on_status_update: Optional[Callable[[str], None]] = None
        self._on_ask_user: Optional[Callable[[str], None]] = None  # 新增：询问用户回调
    
    def set_vision_service(self, vision: VisionService) -> None:
        """设置外部 Vision 服务"""
        self._vision = vision
        self._external_vision = True
        logger.info("Executor已关联外部Vision服务")
    
    def set_planner_service(self, planner: PlannerService) -> None:
        """设置外部 Planner 服务"""
        self._planner = planner
        self._external_planner = True
        logger.info("Executor已关联外部Planner服务")
    
    async def initialize(self):
        """初始化服务"""
        # 如果没有外部 Vision 服务，则初始化内部的
        if not self._vision:
            vl_config = VLConfig(
                api_key=config.api.api_key,
                model=config.api.vl_model,
            )
            self._vision = VisionService(vl_config)
            await self._vision.initialize()
        
        # 如果没有外部 Planner 服务，则初始化内部的
        if not self._planner:
            self._planner = PlannerService()
            await self._planner.initialize()
        
        # 初始化输入监听器
        self._input_listener = InputListener()
        
        # 初始化动作执行器
        self._action_executor = ActionExecutor()
        await self._action_executor.initialize()
        
        logger.info("ExecutorService初始化完成")
    
    async def close(self):
        """关闭服务"""
        if self._input_listener:
            self._input_listener.stop()
        # 只关闭内部创建的服务
        if self._vision and not self._external_vision:
            await self._vision.close()
        if self._planner and not self._external_planner:
            await self._planner.close()
    
    def set_callbacks(
        self,
        on_step_start: Callable[[TaskStep], None] = None,
        on_step_complete: Callable[[TaskStep, bool], None] = None,
        on_need_replan: Callable[[str], None] = None,
        on_task_complete: Callable[[Task, bool], None] = None,
        on_status_update: Callable[[str], None] = None,
        on_ask_user: Callable[[str], None] = None,
    ):
        """设置回调函数"""
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_need_replan = on_need_replan
        self._on_task_complete = on_task_complete
        self._on_status_update = on_status_update
        self._on_ask_user = on_ask_user
    
    def _notify_status(self, message: str):
        """通知状态更新"""
        logger.info(message)
        if self._on_status_update:
            self._on_status_update(message)
    
    def _ask_user(self, question: str):
        """询问用户"""
        logger.info(f"[询问用户] {question}")
        if self._on_ask_user:
            self._on_ask_user(question)
    
    async def execute_step(self, step: TaskStep) -> bool:
        """
        执行单个步骤
        
        Args:
            step: 要执行的任务步骤
            
        Returns:
            bool: 执行是否成功
        """
        if not step or not step.action:
            logger.warning("步骤或动作为空")
            return False
        
        if not self._action_executor:
            logger.error("ActionExecutor 未初始化")
            return False
        
        try:
            logger.info(f"执行步骤: {step.friendly_instruction or step.description}")
            
            # 执行动作
            result = await self._action_executor.execute(step.action)
            
            if result.success:
                logger.info(f"步骤执行成功: {result.message}")
                step.status = ActionStatus.SUCCESS
                return True
            else:
                logger.warning(f"步骤执行失败: {result.message}")
                step.status = ActionStatus.FAILED
                return False
                
        except Exception as e:
            logger.error(f"执行步骤时出错: {e}")
            step.status = ActionStatus.FAILED
            return False
    
    async def execute_task(self, intent: Intent, plan: Optional[TaskPlan] = None) -> Task:
        """
        执行任务的主入口
        
        Args:
            intent: 用户意图
            plan: 可选的任务计划，如果不传则内部生成
        """
        # 验证 intent 参数类型
        if not isinstance(intent, Intent):
            logger.error(f"execute_task 接收到非法的 intent 类型: {type(intent).__name__}")
            # 返回失败的任务
            task = Task(intent=None)
            task.status = TaskStatus.FAILED
            return task
            
        task = Task(intent=intent)
        
        try:
            # 1. 截取当前屏幕
            self._notify_status("正在分析当前屏幕...")
            screenshot, original_size = await self._vision.capture_screen()
            
            # 使用第一层分析：页面状态分析（轻量级）
            user_intent_text = intent.normalized_text or intent.raw_text if intent else ""
            screen_state = await self._vision.analyze_screen_state(
                screenshot, 
                user_intent=user_intent_text,
            )
            
            self._notify_status(f"当前应用: {screen_state.app_name}")
            self._notify_status(f"页面状态: {screen_state.screen_state}")
            
            # 转换为兼容格式供 Planner 使用
            screen_analysis = ScreenAnalysis(
                app_name=screen_state.app_name,
                screen_type=screen_state.screen_state,
                description=screen_state.description,
                suggested_actions=[screen_state.suggested_action] if screen_state.suggested_action else [],
                warnings=screen_state.warnings,
            )
            
            # 2. 使用外部传入的计划，或生成新计划
            if plan and plan.steps:
                self._notify_status(f"使用已有计划，共 {len(plan.steps)} 步")
            else:
                self._notify_status("正在生成任务计划...")
                plan = await self._planner.create_plan(
                    intent=intent,
                    screen_analysis=screen_analysis,
                )
                
                if not plan.steps:
                    self._notify_status("无法生成任务计划")
                    task.status = TaskStatus.FAILED
                    return task
                
                self._notify_status(f"已生成 {len(plan.steps)} 步计划")
            
            # 3. 创建执行上下文
            self._context = ExecutionContext(
                task=task,
                plan=plan,
                last_screenshot=screenshot,
                last_screen_state=screen_state,
                last_screen_analysis=screen_analysis,
                task_goal=intent.normalized_text or intent.raw_text,  # 保存整体任务目标
            )
            
            # 4. 开始执行循环
            await self._execution_loop()
            
            # 5. 返回结果
            if self._context.is_completed:
                task.status = TaskStatus.COMPLETED
                self._notify_status("任务完成！")
            else:
                task.status = TaskStatus.FAILED
                self._notify_status("任务未完成")
            
            return task
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}")
            import traceback
            traceback.print_exc()
            task.status = TaskStatus.FAILED
            return task
    
    async def _execution_loop(self):
        """执行循环 - 基于用户输入事件触发"""
        if not self._context:
            return
        
        # 启动输入监听
        self._input_listener.start()
        self._input_listener.clear()
        
        try:
            while not self._context.is_completed:
                step = self._context.current_step
                if not step:
                    break
                
                # 通知步骤开始
                self._notify_status(f"\n--- 步骤 {step.step_number} ---")
                self._notify_status(f"📋 {step.friendly_instruction or step.description}")
                
                if self._on_step_start:
                    self._on_step_start(step)
                
                # 检查是否是 "完成" 动作类型 - 直接标记任务完成，无需等待用户输入
                if step.action and step.action.action_type == ActionType.DONE:
                    self._notify_status("🎉 任务已完成，无需更多操作！")
                    self._context.step_status = StepStatus.SUCCESS
                    self._context.current_step_index = len(self._context.plan.steps)  # 跳到最后
                    if self._on_task_complete:
                        self._on_task_complete(self._context.task, True)
                    break
                
                # 设置状态为等待用户
                self._context.step_status = StepStatus.WAITING_USER
                self._notify_status("⏳ 等待您完成操作...")
                
                # 等待用户输入事件
                input_event = await self._wait_for_user_input()
                
                if input_event:
                    # 记录用户输入时间
                    self._context.last_user_input_time = input_event.timestamp
                    
                    if input_event.event_type == "mouse_click":
                        self._notify_status(f"🖱️ 检测到点击: ({input_event.x}, {input_event.y})")
                    elif input_event.event_type == "key_press":
                        self._notify_status(f"⌨️ 检测到按键")
                    
                    # 评估步骤和任务完成情况
                    self._context.step_status = StepStatus.VERIFYING
                    result = await self._evaluate_step_and_task(step)
                    
                    if result == StepCompletionResult.TASK_COMPLETED:
                        # 整体任务已完成，提前结束
                        self._notify_status("🎉 任务目标已达成！")
                        self._context.current_step_index = len(self._context.plan.steps)
                        if self._on_task_complete:
                            self._on_task_complete(self._context.task, True)
                        break
                    
                    elif result == StepCompletionResult.COMPLETED:
                        # 当前步骤完成，继续下一步
                        self._context.step_status = StepStatus.SUCCESS
                        self._context.current_step_index += 1
                        self._context.retry_count = 0
                        
                        if self._on_step_complete:
                            self._on_step_complete(step, True)
                        
                        self._notify_status("✅ 步骤完成")
                    
                    elif result == StepCompletionResult.NEED_RETRY:
                        # 需要重试
                        self._context.retry_count += 1
                        if self._context.retry_count >= self._context.max_retries:
                            await self._handle_step_failure(step)
                        else:
                            self._notify_status(f"⚠️ 请重试操作 (第 {self._context.retry_count} 次)")
                    
                    elif result == StepCompletionResult.NEED_REPLAN:
                        # 需要重新规划
                        await self._handle_step_failure(step)
                    
                    elif result == StepCompletionResult.WAITING:
                        # 继续等待（页面动态效果，非用户操作导致的变化）
                        self._notify_status("⏳ 继续等待您的操作...")
                        continue
                
                elif self._context.seconds_since_last_input >= self._context.idle_timeout:
                    # 超时处理
                    result = await self._handle_timeout(step)
                    if result == StepCompletionResult.TASK_COMPLETED:
                        break
                
                # 检查用户反馈
                if self._context.user_feedback:
                    await self._handle_user_feedback()
                
        finally:
            self._input_listener.stop()
    
    async def _wait_for_user_input(self, timeout: float = 5.0) -> Optional[UserInputEvent]:
        """等待用户输入事件（短超时，用于轮询检查）"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 检查是否有输入事件
            event = self._input_listener.get_event(timeout=1.0)
            if event:
                return event
            
            # 检查用户反馈
            if self._context and self._context.user_feedback:
                return None
            
            # 检查是否超时
            if self._context and self._context.seconds_since_last_input >= self._context.idle_timeout:
                return None
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def _evaluate_step_and_task(self, step: TaskStep) -> StepCompletionResult:
        """
        评估步骤和整体任务的完成情况
        
        返回:
        - TASK_COMPLETED: 整体任务已完成，无需继续剩余步骤
        - COMPLETED: 当前步骤完成，继续下一步
        - NEED_RETRY: 需要重试当前步骤
        - NEED_REPLAN: 需要重新规划
        - WAITING: 继续等待（页面动态效果）
        """
        self._notify_status("🔍 正在验证操作结果...")
        
        # 等待一小段时间让页面响应
        await asyncio.sleep(0.5)
        
        # 截取新屏幕
        new_screenshot, original_size = await self._vision.capture_screen()
        
        # 使用第一层分析：页面状态分析（轻量级）
        new_state = await self._vision.analyze_screen_state(new_screenshot)
        
        # 检测屏幕状态
        screen_state = self._detect_screen_state_from_analysis(new_state)
        
        self._notify_status(f"   屏幕状态: {screen_state.value}")
        
        # 处理加载状态
        if screen_state == ScreenState.LOADING:
            self._notify_status("⏳ 页面加载中，请稍候...")
            success = await self._wait_for_loading_complete()
            if success:
                new_screenshot, _ = await self._vision.capture_screen()
                new_state = await self._vision.analyze_screen_state(new_screenshot)
                screen_state = ScreenState.CHANGED
            else:
                return StepCompletionResult.NEED_RETRY
        
        # 处理错误状态
        if screen_state == ScreenState.ERROR:
            self._notify_status("❌ 检测到页面错误")
            return StepCompletionResult.NEED_REPLAN
        
        # 检查是否是页面动态效果（非用户操作导致的变化）
        if screen_state == ScreenState.UNCHANGED:
            # 页面没有变化，可能用户操作没有生效
            is_dynamic_effect = await self._is_dynamic_page_effect(
                self._context.last_screenshot, 
                new_screenshot
            )
            if is_dynamic_effect:
                # 是页面动态效果，继续等待
                return StepCompletionResult.WAITING
            else:
                # 不是动态效果，用户操作可能没有生效
                self._context.retry_count += 1
                if self._context.retry_count < self._context.max_retries:
                    self._notify_status("⚠️ 页面似乎没有变化，请重试操作")
                    return StepCompletionResult.NEED_RETRY
        
        # 首先检查整体任务是否已完成
        task_completed, task_reason = await self._check_task_goal_achieved(
            new_screenshot, 
            new_state
        )
        if task_completed:
            self._notify_status(f"✨ {task_reason}")
            # 更新上下文
            self._context.last_screenshot = new_screenshot
            self._context.last_screen_state = new_state
            return StepCompletionResult.TASK_COMPLETED
        
        # 使用 VL 模型验证当前步骤是否完成
        if step.expected_result or step.description:
            success, changes, reason = await self._vision.verify_step_completion(
                before_screenshot=self._context.last_screenshot,
                after_screenshot=new_screenshot,
                step_description=step.friendly_instruction or step.description,
                expected_result=step.expected_result or "操作成功完成",
            )
            
            if changes:
                self._notify_status(f"   变化: {changes}")
            
            if not success:
                self._notify_status(f"⚠️ {reason}")
                # 更新上下文后返回
                self._context.last_screenshot = new_screenshot
                self._context.last_screen_state = new_state
                return StepCompletionResult.NEED_RETRY
        
        # 更新上下文
        self._context.last_screenshot = new_screenshot
        self._context.last_screen_state = new_state
        
        return StepCompletionResult.COMPLETED
    
    async def _check_task_goal_achieved(
        self, 
        screenshot: bytes, 
        screen_state: ScreenStateAnalysis
    ) -> tuple[bool, str]:
        """
        检查整体任务目标是否已达成
        
        返回: (是否完成, 原因说明)
        """
        if not self._context or not self._context.task_goal:
            return False, ""
        
        try:
            import base64
            image_b64 = base64.b64encode(screenshot).decode("utf-8")
            
            prompt = f"""判断用户的任务目标是否已经达成。

任务目标：{self._context.task_goal}

当前页面状态：
- 应用：{screen_state.app_name}
- 页面：{screen_state.screen_state}
- 描述：{screen_state.description}

请判断：
1. 当前页面是否显示任务目标已经完成？
2. 用户是否已经达到了他想要的结果？

只返回JSON：
{{
  "goal_achieved": true或false,
  "reason": "判断理由，用简单语言描述"
}}

注意：
- 如果任务是"发送消息给某人"，看到消息已发送就算完成
- 如果任务是"打开某个应用"，看到应用已打开就算完成
- 如果任务是"查看某个信息"，看到信息已显示就算完成
- 不要因为还有其他可以做的操作就判断为未完成"""
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            content = await self._vision._call_vl_api(
                messages,
                model=self._vision._config.model_light,
                max_tokens=300,
            )
            
            # 解析结果
            import json
            json_str = self._vision._extract_json(content)
            if json_str:
                data = json.loads(json_str)
                return data.get("goal_achieved", False), data.get("reason", "")
            
        except Exception as e:
            logger.warning(f"检查任务目标失败: {e}")
        
        return False, ""
    
    async def _is_dynamic_page_effect(
        self, 
        before_screenshot: bytes, 
        after_screenshot: bytes
    ) -> bool:
        """
        判断页面变化是否是动态效果（广告、动画、通知等），而非用户操作导致
        
        返回: True 表示是动态效果，False 表示是用户操作导致的变化
        """
        try:
            import base64
            before_b64 = base64.b64encode(before_screenshot).decode("utf-8")
            after_b64 = base64.b64encode(after_screenshot).decode("utf-8")
            
            prompt = """比较这两张截图，判断页面变化的原因。

请分析：
1. 页面是否有变化？
2. 如果有变化，是什么类型的变化？
   - 用户操作导致的变化（点击按钮、输入文字、滚动页面等）
   - 页面动态效果（广告轮播、动画、通知弹窗、自动刷新等）

只返回JSON：
{
  "has_change": true或false,
  "change_type": "user_action" 或 "dynamic_effect" 或 "none",
  "description": "变化描述"
}"""
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            content = await self._vision._call_vl_api(
                messages,
                model=self._vision._config.model_light,
                max_tokens=300,
            )
            
            # 解析结果
            import json
            json_str = self._vision._extract_json(content)
            if json_str:
                data = json.loads(json_str)
                change_type = data.get("change_type", "none")
                return change_type == "dynamic_effect"
            
        except Exception as e:
            logger.warning(f"判断页面变化类型失败: {e}")
        
        return False
    
    async def _handle_timeout(self, step: TaskStep) -> StepCompletionResult:
        """处理超时情况"""
        self._notify_status(f"⏰ 已等待 {int(self._context.idle_timeout)} 秒没有操作")
        
        # 询问用户
        question = f"您是否需要帮助完成这一步：{step.friendly_instruction or step.description}？"
        self._ask_user(question)
        
        # 重置超时计时器
        self._context.last_user_input_time = datetime.now()
        
        return StepCompletionResult.WAITING
    
    def _detect_screen_state_from_analysis(self, state: ScreenStateAnalysis) -> ScreenState:
        """从页面状态分析结果检测屏幕状态"""
        # 使用 PageStatus 枚举判断
        if state.page_status == PageStatus.LOADING:
            return ScreenState.LOADING
        if state.page_status == PageStatus.ERROR:
            return ScreenState.ERROR
        
        # 检查描述中的关键词
        description = (state.description or "").lower()
        
        loading_keywords = ["加载", "loading", "请稍候", "正在", "处理中"]
        for keyword in loading_keywords:
            if keyword in description:
                return ScreenState.LOADING
        
        error_keywords = ["错误", "失败", "error", "failed", "无法连接"]
        for keyword in error_keywords:
            if keyword in description:
                return ScreenState.ERROR
        
        # 比较与上次状态的差异
        if self._context.last_screen_state:
            old_state = self._context.last_screen_state
            if (old_state.app_name == state.app_name and 
                old_state.screen_state == state.screen_state and
                set(old_state.available_elements) == set(state.available_elements)):
                return ScreenState.UNCHANGED
        
        return ScreenState.CHANGED
    
    async def _wait_for_loading_complete(self, max_wait: float = 10.0) -> bool:
        """等待页面加载完成"""
        start_time = time.time()
        check_interval = 2.0
        
        while time.time() - start_time < max_wait:
            await asyncio.sleep(check_interval)
            
            # 重新截图检查（使用轻量级分析）
            screenshot, _ = await self._vision.capture_screen()
            state = await self._vision.analyze_screen_state(screenshot)
            
            screen_state = self._detect_screen_state_from_analysis(state)
            
            if screen_state != ScreenState.LOADING:
                self._notify_status("✅ 页面加载完成")
                self._context.last_screenshot = screenshot
                self._context.last_screen_state = state
                return screen_state != ScreenState.ERROR
        
        self._notify_status("⚠️ 页面加载超时")
        return False
    
    async def _handle_step_failure(self, step: TaskStep):
        """处理步骤失败"""
        self._context.step_status = StepStatus.FAILED
        self._context.retry_count += 1
        
        if self._context.retry_count >= self._context.max_retries:
            self._notify_status("🔄 需要重新规划...")
            await self._replan(f"步骤 {step.step_number} 执行失败")
        else:
            self._notify_status(f"⚠️ 请重试操作 (第 {self._context.retry_count} 次)")
            if step.error_recovery_hint:
                self._notify_status(f"💡 提示: {step.error_recovery_hint}")
    
    async def _replan(self, reason: str):
        """重新规划"""
        self._context.step_status = StepStatus.REPLANNING
        
        if self._on_need_replan:
            self._on_need_replan(reason)
        
        # 获取当前屏幕状态（使用轻量级分析）
        screenshot, original_size = await self._vision.capture_screen()
        screen_state = await self._vision.analyze_screen_state(screenshot)
        
        # 转换为兼容格式
        screen_analysis = ScreenAnalysis(
            app_name=screen_state.app_name,
            screen_type=screen_state.screen_state,
            description=screen_state.description,
        )
        
        # 调用重规划
        new_plan = await self._planner.replan_on_error(
            task=self._context.task,
            error_description=reason,
            current_screen=screen_analysis,
        )
        
        if new_plan.steps:
            self._notify_status(f"✅ 已生成新计划，共 {len(new_plan.steps)} 步")
            self._context.plan = new_plan
            self._context.task.plan = new_plan
            self._context.current_step_index = 0
            self._context.retry_count = 0
        else:
            self._notify_status("❌ 无法生成新计划")
            self._context.step_status = StepStatus.FAILED
    
    async def _handle_user_feedback(self):
        """处理用户反馈"""
        feedback = self._context.user_feedback
        self._context.user_feedback = ""
        
        if not feedback:
            return
        
        self._notify_status(f"📝 收到用户反馈: {feedback}")
        
        # 根据反馈重新规划
        await self._replan(f"用户反馈: {feedback}")
    
    def submit_user_feedback(self, feedback: str):
        """提交用户反馈（供外部调用）"""
        if self._context:
            self._context.user_feedback = feedback
            self._notify_status(f"已收到反馈: {feedback}")
    
    def get_current_step(self) -> Optional[TaskStep]:
        """获取当前步骤"""
        if self._context:
            return self._context.current_step
        return None
    
    def get_progress(self) -> tuple[int, int]:
        """获取进度 (当前步骤, 总步骤)"""
        if self._context:
            return (self._context.current_step_index + 1, len(self._context.plan.steps))
        return (0, 0)
