"""会话模型"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from .task import Task


class SessionState(str, Enum):
    """会话状态"""
    IDLE = "idle"                    # 空闲
    LISTENING = "listening"          # 监听中
    PROCESSING = "processing"        # 处理中
    GUIDING = "guiding"              # 引导中
    WAITING_CONFIRM = "waiting_confirm"  # 等待确认
    ERROR = "error"                  # 错误状态


@dataclass
class UserProfile:
    """用户画像"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    
    # 家庭成员映射 (老二 -> 张三)
    family_mapping: dict[str, str] = field(default_factory=dict)
    
    # 常用联系人
    frequent_contacts: list[str] = field(default_factory=list)
    
    # 常用应用
    frequent_apps: list[str] = field(default_factory=list)
    
    # 操作偏好
    preferred_voice_speed: float = 0.8  # 语速 (0.5-1.5)
    preferred_font_size: str = "large"  # small, medium, large, xlarge
    
    # 历史记录
    completed_tasks_count: int = 0
    failed_tasks_count: int = 0
    
    # 焦虑指数 (0-1, 越低越好)
    anxiety_index: float = 0.5
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def resolve_family_reference(self, reference: str) -> Optional[str]:
        """解析家庭成员引用"""
        # 常见的家庭称呼映射
        common_refs = {
            "老大": "大儿子/大女儿",
            "老二": "二儿子/二女儿",
            "老三": "三儿子/三女儿",
            "闺女": "女儿",
            "儿子": "儿子",
            "老伴": "配偶",
            "孙子": "孙子",
            "孙女": "孙女",
            "外孙": "外孙",
            "外孙女": "外孙女",
        }
        
        # 先查用户自定义映射
        if reference in self.family_mapping:
            return self.family_mapping[reference]
        
        # 再查通用映射
        return common_refs.get(reference)
    
    def update_anxiety_index(self, task_success: bool) -> None:
        """更新焦虑指数"""
        if task_success:
            # 成功降低焦虑
            self.anxiety_index = max(0.0, self.anxiety_index - 0.05)
            self.completed_tasks_count += 1
        else:
            # 失败增加焦虑
            self.anxiety_index = min(1.0, self.anxiety_index + 0.1)
            self.failed_tasks_count += 1
        self.updated_at = datetime.now()


@dataclass
class Session:
    """用户会话"""
    id: UUID = field(default_factory=uuid4)
    user_profile: UserProfile = field(default_factory=UserProfile)
    state: SessionState = SessionState.IDLE
    
    # 当前任务
    current_task: Optional[Task] = None
    
    # 任务历史
    task_history: list[Task] = field(default_factory=list)
    
    # 对话历史 (用于上下文理解)
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    max_history_length: int = 10
    
    # 最后活动时间
    last_activity: datetime = field(default_factory=datetime.now)
    
    # 会话超时 (秒)
    timeout_seconds: int = 1800  # 30分钟
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_conversation(self, role: str, content: str) -> None:
        """添加对话记录"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        
        # 保持历史长度
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        self.last_activity = datetime.now()
    
    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        if not self.conversation_history:
            return "这是新的对话开始。"
        
        recent = self.conversation_history[-3:]
        summary_parts = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            summary_parts.append(f"{role}: {msg['content'][:50]}...")
        
        return "\n".join(summary_parts)
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed > self.timeout_seconds
    
    def complete_current_task(self, success: bool) -> None:
        """完成当前任务"""
        if self.current_task:
            self.task_history.append(self.current_task)
            self.user_profile.update_anxiety_index(success)
            self.current_task = None
