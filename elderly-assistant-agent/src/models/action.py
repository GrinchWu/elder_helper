"""操作动作模型"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4


class ActionType(str, Enum):
    """动作类型（与 Skill Set 对应）"""
    # 鼠标操作
    CLICK = "click"                  # 单击
    DOUBLE_CLICK = "double_click"    # 双击
    RIGHT_CLICK = "right_click"      # 右键单击
    DRAG = "drag"                    # 拖动
    
    # 滚动操作
    SCROLL = "scroll"                # 滚动（向上/向下）
    
    # 键盘操作
    TYPE = "type"                    # 输入文字
    KEY_PRESS = "key_press"          # 按下单个按键
    HOTKEY = "hotkey"                # 组合键
    
    # 等待操作
    WAIT = "wait"                    # 等待指定时间
    WAIT_ELEMENT = "wait_element"    # 等待元素出现
    
    # 完成操作
    DONE = "done"                    # 任务完成
    
    # 系统操作
    SCREENSHOT = "screenshot"        # 截图
    SPEAK = "speak"                  # 语音反馈
    CONFIRM = "confirm"              # 确认操作
    CANCEL = "cancel"                # 取消操作
    BACK = "back"                    # 返回上一步


class ActionStatus(str, Enum):
    """动作状态"""
    PENDING = "pending"              # 待执行
    EXECUTING = "executing"          # 执行中
    SUCCESS = "success"              # 成功
    FAILED = "failed"                # 失败
    CANCELLED = "cancelled"          # 已取消
    ROLLBACK = "rollback"            # 已回滚


@dataclass
class ActionResult:
    """动作执行结果"""
    success: bool
    message: str = ""
    screenshot_after: Optional[bytes] = None
    error_code: Optional[str] = None
    rollback_available: bool = True
    
    @classmethod
    def ok(cls, message: str = "操作成功") -> ActionResult:
        return cls(success=True, message=message)
    
    @classmethod
    def fail(cls, message: str, error_code: Optional[str] = None) -> ActionResult:
        return cls(success=False, message=message, error_code=error_code)


@dataclass
class Action:
    """单个操作动作"""
    id: UUID = field(default_factory=uuid4)
    action_type: ActionType = ActionType.CLICK
    
    # 位置信息 (用于点击、拖拽等)
    x: Optional[int] = None
    y: Optional[int] = None
    target_x: Optional[int] = None  # 拖拽目标位置
    target_y: Optional[int] = None
    
    # 输入信息
    text: Optional[str] = None
    
    # 按键信息（新增）
    key: Optional[str] = None        # 单个按键
    hotkey: Optional[str] = None     # 组合键（如 "Ctrl+C"）
    
    # 滚动信息
    scroll_direction: Optional[str] = None  # up, down, left, right
    scroll_amount: int = 100
    
    # 等待时间 (毫秒)
    wait_ms: int = 500
    
    # 元素描述 (用于定位和语音反馈)
    element_description: str = ""
    
    # 视觉提示（新增）
    visual_hint: str = ""            # 帮助用户找到目标的提示
    
    # 状态
    status: ActionStatus = ActionStatus.PENDING
    result: Optional[ActionResult] = None
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    
    # 回滚信息
    rollback_action: Optional[Action] = None
    
    def to_friendly_description(self) -> str:
        """生成老年人友好的操作描述"""
        descriptions = {
            ActionType.CLICK: f"单击{{{self.element_description}}}",
            ActionType.DOUBLE_CLICK: f"双击{{{self.element_description}}}",
            ActionType.RIGHT_CLICK: f"右键单击{{{self.element_description}}}",
            ActionType.DRAG: f"拖动{{{self.element_description}}}",
            ActionType.TYPE: f"输入{{{self.text}}}",
            ActionType.KEY_PRESS: f"按下{{{self.key}}}",
            ActionType.HOTKEY: f"组合键{{{self.hotkey}}}",
            ActionType.SCROLL: f"向{self._scroll_direction_cn()}滚动",
            ActionType.WAIT: f"等待{{{self.wait_ms // 1000}秒}}",
            ActionType.WAIT_ELEMENT: f"等待{{{self.element_description}}}出现",
            ActionType.DONE: "完成",
            ActionType.BACK: "返回上一步",
        }
        return descriptions.get(self.action_type, "执行操作")
    
    def to_skill_instruction(self) -> str:
        """转换为标准化技能指令格式"""
        if self.action_type == ActionType.CLICK:
            return f"单击{{{self.element_description}}}"
        elif self.action_type == ActionType.DOUBLE_CLICK:
            return f"双击{{{self.element_description}}}"
        elif self.action_type == ActionType.RIGHT_CLICK:
            return f"右键单击{{{self.element_description}}}"
        elif self.action_type == ActionType.DRAG:
            return f"拖动{{{self.element_description}}}至{{{self.target_x}, {self.target_y}}}"
        elif self.action_type == ActionType.SCROLL:
            direction = "向上" if self.scroll_direction == "up" else "向下"
            return f"{direction}滚动{{{self.element_description or '当前区域'}}}"
        elif self.action_type == ActionType.TYPE:
            return f"输入{{{self.text}}}"
        elif self.action_type == ActionType.KEY_PRESS:
            return f"按下{{{self.key}}}"
        elif self.action_type == ActionType.HOTKEY:
            return f"组合键{{{self.hotkey}}}"
        elif self.action_type == ActionType.WAIT:
            return f"等待{{{self.wait_ms // 1000}秒}}"
        elif self.action_type == ActionType.WAIT_ELEMENT:
            return f"等待{{{self.element_description}}}出现"
        return "未知操作"
    
    def _scroll_direction_cn(self) -> str:
        """滚动方向中文"""
        mapping = {"up": "上", "down": "下", "left": "左", "right": "右"}
        return mapping.get(self.scroll_direction or "down", "下")
