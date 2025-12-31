"""任务模型"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from .action import Action, ActionStatus
from .intent import Intent


class TaskStatus(str, Enum):
    """任务状态"""
    PLANNING = "planning"            # 规划中
    READY = "ready"                  # 准备就绪
    IN_PROGRESS = "in_progress"      # 进行中
    WAITING_USER = "waiting_user"    # 等待用户操作
    PAUSED = "paused"                # 暂停
    COMPLETED = "completed"          # 完成
    FAILED = "failed"                # 失败
    CANCELLED = "cancelled"          # 取消


@dataclass
class TaskStep:
    """任务步骤"""
    id: UUID = field(default_factory=uuid4)
    step_number: int = 0
    description: str = ""                    # 步骤描述
    friendly_instruction: str = ""           # 老年人友好的指令
    action: Optional[Action] = None
    status: ActionStatus = ActionStatus.PENDING
    
    # 视觉提示
    highlight_area: Optional[tuple[int, int, int, int]] = None  # x, y, width, height
    
    # 预期结果描述
    expected_result: str = ""
    
    # 错误恢复指导
    error_recovery_hint: str = ""
    
    def to_voice_instruction(self) -> str:
        """生成语音指令"""
        return self.friendly_instruction or self.description


@dataclass
class TaskPlan:
    """任务执行计划"""
    id: UUID = field(default_factory=uuid4)
    intent: Optional[Intent] = None
    steps: list[TaskStep] = field(default_factory=list)
    current_step_index: int = 0
    
    # 来源知识
    source_video_ids: list[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def current_step(self) -> Optional[TaskStep]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    @property
    def total_steps(self) -> int:
        return len(self.steps)
    
    @property
    def progress_percentage(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == ActionStatus.SUCCESS)
        return (completed / len(self.steps)) * 100
    
    def advance_to_next_step(self) -> Optional[TaskStep]:
        """前进到下一步"""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return self.current_step
        return None
    
    def rollback_to_previous_step(self) -> Optional[TaskStep]:
        """回退到上一步"""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            return self.current_step
        return None


@dataclass
class Task:
    """完整任务"""
    id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    intent: Optional[Intent] = None
    plan: Optional[TaskPlan] = None
    status: TaskStatus = TaskStatus.PLANNING
    
    # 执行历史
    action_history: list[Action] = field(default_factory=list)
    
    # 错误信息
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries
    
    def record_action(self, action: Action) -> None:
        """记录执行的动作"""
        self.action_history.append(action)
    
    def get_rollback_actions(self) -> list[Action]:
        """获取可回滚的动作列表"""
        return [a for a in reversed(self.action_history) if a.rollback_action]
