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


class ScreenState(str, Enum):
    """屏幕状态"""
    NORMAL = "normal"                # 正常
    LOADING = "loading"              # 加载中
    ERROR = "error"                  # 错误
    CHANGED = "changed"              # 已变化
    UNCHANGED = "unchanged"          # 未变化


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
    
    @property
    def current_step(self) -> Optional[TaskStep]:
        if 0 <= self.current_step_index < len(self.plan.steps):
            return self.plan.steps[self.current_step_index]
        return None
    
    @property
    def is_completed(self) -> bool:
        return self.current_step_index >= len(self.plan.steps)


@dataclass
class MouseClickEvent:
    """鼠标点击事件"""
    x: int
    y: int
    button: str
    timestamp: datetime = field(default_factory=datetime.now)


class MouseListener:
    """鼠标点击监听器"""
    
    def __init__(self):
        self._listener = None
        self._click_queue: Queue[MouseClickEvent] = Queue()
        self._is_listening = False
        self._lock = threading.Lock()
    
    def start(self):
        """开始监听"""
        if self._is_listening:
            return
        
        try:
            from pynput import mouse
            
            def on_click(x, y, button, pressed):
                if pressed:  # 只记录按下事件
                    event = MouseClickEvent(
                        x=int(x),
                        y=int(y),
                        button=str(button),
                    )
                    self._click_queue.put(event)
                    logger.debug(f"鼠标点击: ({x}, {y}) {button}")
            
            self._listener = mouse.Listener(on_click=on_click)
            self._listener.start()
            self._is_listening = True
            logger.info("鼠标监听器已启动")
            
        except ImportError:
            logger.warning("pynput未安装，鼠标监听不可用")
        except Exception as e:
            logger.error(f"启动鼠标监听失败: {e}")
    
    def stop(self):
        """停止监听"""
        if self._listener:
            self._listener.stop()
            self._listener = None
        self._is_listening = False
        logger.info("鼠标监听器已停止")
    
    def get_click(self, timeout: float = None) -> Optional[MouseClickEvent]:
        """获取点击事件（阻塞）"""
        try:
            return self._click_queue.get(timeout=timeout)
        except:
            return None
    
    def has_click(self) -> bool:
        """是否有待处理的点击"""
        return not self._click_queue.empty()
    
    def clear(self):
        """清空点击队列"""
        while not self._click_queue.empty():
            try:
                self._click_queue.get_nowait()
            except:
                break


class ExecutorService:
    """任务执行服务"""
    
    def __init__(self):
        self._vision: Optional[VisionService] = None
        self._planner: Optional[PlannerService] = None
        self._mouse_listener: Optional[MouseListener] = None
        self._context: Optional[ExecutionContext] = None
        
        # 回调函数
        self._on_step_start: Optional[Callable[[TaskStep], None]] = None
        self._on_step_complete: Optional[Callable[[TaskStep, bool], None]] = None
        self._on_need_replan: Optional[Callable[[str], None]] = None
        self._on_task_complete: Optional[Callable[[Task, bool], None]] = None
        self._on_status_update: Optional[Callable[[str], None]] = None
    
    async def initialize(self):
        """初始化服务"""
        # 初始化Vision服务
        vl_config = VLConfig(
            api_key=config.api.api_key,
            model=config.api.vl_model,
        )
        self._vision = VisionService(vl_config)
        await self._vision.initialize()
        
        # 初始化Planner服务
        self._planner = PlannerService()
        await self._planner.initialize()
        
        # 初始化鼠标监听器
        self._mouse_listener = MouseListener()
        
        logger.info("ExecutorService初始化完成")
    
    async def close(self):
        """关闭服务"""
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._vision:
            await self._vision.close()
        if self._planner:
            await self._planner.close()
    
    def set_callbacks(
        self,
        on_step_start: Callable[[TaskStep], None] = None,
        on_step_complete: Callable[[TaskStep, bool], None] = None,
        on_need_replan: Callable[[str], None] = None,
        on_task_complete: Callable[[Task, bool], None] = None,
        on_status_update: Callable[[str], None] = None,
    ):
        """设置回调函数"""
        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_need_replan = on_need_replan
        self._on_task_complete = on_task_complete
        self._on_status_update = on_status_update
    
    def _notify_status(self, message: str):
        """通知状态更新"""
        logger.info(message)
        if self._on_status_update:
            self._on_status_update(message)
    
    async def execute_task(self, intent: Intent) -> Task:
        """执行任务的主入口"""
        task = Task(intent=intent)
        
        try:
            # 1. 截取当前屏幕
            self._notify_status("正在分析当前屏幕...")
            screenshot, original_size = await self._vision.capture_screen()
            
            # 使用第一层分析：页面状态分析（轻量级）
            screen_state = await self._vision.analyze_screen_state(
                screenshot, 
                user_intent=intent.raw_text,
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
            
            # 2. 生成全局计划
            self._notify_status("正在生成任务计划...")
            plan = await self._planner.create_plan(
                intent=intent,
                screen_analysis=screen_analysis,
            )
            
            if not plan.steps:
                self._notify_status("无法生成任务计划")
                task.status = TaskStatus.FAILED
                return task
            
            task.plan = plan
            self._notify_status(f"已生成 {len(plan.steps)} 步计划")
            
            # 3. 创建执行上下文
            self._context = ExecutionContext(
                task=task,
                plan=plan,
                last_screenshot=screenshot,
                last_screen_state=screen_state,
                last_screen_analysis=screen_analysis,
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
        """执行循环"""
        if not self._context:
            return
        
        # 启动鼠标监听
        self._mouse_listener.start()
        self._mouse_listener.clear()
        
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
                
                # 设置状态为等待用户
                self._context.step_status = StepStatus.WAITING_USER
                self._notify_status("⏳ 等待您完成操作（点击鼠标后继续）...")
                
                # 等待用户操作（鼠标点击）
                click_event = await self._wait_for_user_action()
                
                if click_event:
                    self._notify_status(f"🖱️ 检测到点击: ({click_event.x}, {click_event.y})")
                    
                    # 验证执行结果
                    self._context.step_status = StepStatus.VERIFYING
                    success = await self._verify_step_result(step)
                    
                    if success:
                        # 步骤成功，继续下一步
                        self._context.step_status = StepStatus.SUCCESS
                        self._context.current_step_index += 1
                        self._context.retry_count = 0
                        
                        if self._on_step_complete:
                            self._on_step_complete(step, True)
                        
                        self._notify_status("✅ 步骤完成")
                    else:
                        # 步骤失败，尝试重规划
                        await self._handle_step_failure(step)
                
                # 检查用户反馈
                if self._context.user_feedback:
                    await self._handle_user_feedback()
                
        finally:
            self._mouse_listener.stop()
    
    async def _wait_for_user_action(self, timeout: float = 300) -> Optional[MouseClickEvent]:
        """等待用户操作"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 检查是否有鼠标点击
            click = self._mouse_listener.get_click(timeout=1.0)
            if click:
                return click
            
            # 检查用户反馈
            if self._context and self._context.user_feedback:
                return None
            
            await asyncio.sleep(0.1)
        
        return None
    
    async def _verify_step_result(self, step: TaskStep) -> bool:
        """验证步骤执行结果"""
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
        
        if screen_state == ScreenState.LOADING:
            self._notify_status("⏳ 页面加载中，请稍候...")
            success = await self._wait_for_loading_complete()
            if success:
                new_screenshot, _ = await self._vision.capture_screen()
                new_state = await self._vision.analyze_screen_state(new_screenshot)
                screen_state = ScreenState.CHANGED
        
        if screen_state == ScreenState.ERROR:
            self._notify_status("❌ 检测到页面错误")
            return False
        
        if screen_state == ScreenState.UNCHANGED:
            self._context.retry_count += 1
            if self._context.retry_count < self._context.max_retries:
                self._notify_status("⚠️ 页面似乎没有变化，请重试操作")
                return False
        
        # 使用 VL 模型验证是否符合预期
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
                # 更新上下文后返回失败
                self._context.last_screenshot = new_screenshot
                self._context.last_screen_state = new_state
                return False
        
        # 更新上下文
        self._context.last_screenshot = new_screenshot
        self._context.last_screen_state = new_state
        
        return True
    
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
