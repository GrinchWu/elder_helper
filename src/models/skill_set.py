"""
标准化操作技能集 (Skill Set)

将所有计算机操作原子化、标准化，确保模型输出的一致性和可执行性。

设计原则：
1. 所有操作都是原子操作，不可再分
2. 格式统一：动作{对象} 或 动作{对象}至{目标}
3. 对象必须是可识别的 GUI 元素或系统元素
4. 每个技能都有明确的验证方式
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


# ==================== 动作类型定义 ====================

class SkillType(str, Enum):
    """原子操作类型"""
    # 鼠标操作
    CLICK = "单击"           # 单击{目标}
    DOUBLE_CLICK = "双击"    # 双击{目标}
    RIGHT_CLICK = "右键单击"  # 右键单击{目标}
    DRAG = "拖动"            # 拖动{对象}至{目标位置}
    
    # 滚动操作
    SCROLL_UP = "向上滚动"    # 向上滚动{区域}
    SCROLL_DOWN = "向下滚动"  # 向下滚动{区域}
    
    # 键盘操作
    TYPE = "输入"            # 输入{文本内容}
    PRESS = "按下"           # 按下{按键}
    HOTKEY = "组合键"        # 组合键{按键1}+{按键2}
    
    # 等待操作
    WAIT_TIME = "等待"       # 等待{秒数}
    WAIT_ELEMENT = "等待出现" # 等待{元素}出现


# ==================== 系统元素定义 ====================

class SystemElement(str, Enum):
    """系统级 GUI 元素（跨应用通用）"""
    
    # 窗口控制按钮（右上角）
    CLOSE_BUTTON = "关闭按钮"           # × 按钮
    MINIMIZE_BUTTON = "最小化按钮"       # - 按钮
    MAXIMIZE_BUTTON = "最大化按钮"       # □ 按钮
    RESTORE_BUTTON = "还原按钮"          # 还原窗口大小
    
    # 任务栏元素
    START_BUTTON = "开始按钮"            # Windows 图标
    TASKBAR = "任务栏"                   # 屏幕底部任务栏
    SYSTEM_TRAY = "系统托盘"             # 右下角托盘区
    SEARCH_BOX = "搜索框"                # 任务栏搜索框
    
    # 桌面元素
    DESKTOP = "桌面"                     # 桌面区域
    DESKTOP_ICON = "桌面图标"            # 桌面上的图标
    
    # 通用控件
    SCROLL_BAR = "滚动条"
    MENU_BAR = "菜单栏"
    TITLE_BAR = "标题栏"
    STATUS_BAR = "状态栏"
    
    # 对话框元素
    OK_BUTTON = "确定按钮"
    CANCEL_BUTTON = "取消按钮"
    YES_BUTTON = "是按钮"
    NO_BUTTON = "否按钮"
    APPLY_BUTTON = "应用按钮"
    
    # 输入控件
    TEXT_INPUT = "文本输入框"
    PASSWORD_INPUT = "密码输入框"
    DROPDOWN = "下拉菜单"
    CHECKBOX = "复选框"
    RADIO_BUTTON = "单选按钮"
    
    # 导航元素
    BACK_BUTTON = "返回按钮"
    FORWARD_BUTTON = "前进按钮"
    REFRESH_BUTTON = "刷新按钮"
    HOME_BUTTON = "主页按钮"


class KeyboardKey(str, Enum):
    """键盘按键"""
    # 功能键
    ENTER = "回车键"
    ESCAPE = "Esc键"
    TAB = "Tab键"
    BACKSPACE = "退格键"
    DELETE = "删除键"
    SPACE = "空格键"
    
    # 方向键
    UP = "上箭头"
    DOWN = "下箭头"
    LEFT = "左箭头"
    RIGHT = "右箭头"
    
    # 修饰键
    CTRL = "Ctrl键"
    ALT = "Alt键"
    SHIFT = "Shift键"
    WIN = "Windows键"
    
    # 功能键 F1-F12
    F1 = "F1键"
    F2 = "F2键"
    F3 = "F3键"
    F4 = "F4键"
    F5 = "F5键"
    F11 = "F11键"
    F12 = "F12键"
    
    # 其他
    HOME = "Home键"
    END = "End键"
    PAGE_UP = "PageUp键"
    PAGE_DOWN = "PageDown键"
    PRINT_SCREEN = "截图键"


# ==================== 常用组合键 ====================

class CommonHotkey(str, Enum):
    """常用组合键"""
    COPY = "Ctrl+C"           # 复制
    PASTE = "Ctrl+V"          # 粘贴
    CUT = "Ctrl+X"            # 剪切
    UNDO = "Ctrl+Z"           # 撤销
    REDO = "Ctrl+Y"           # 重做
    SAVE = "Ctrl+S"           # 保存
    SELECT_ALL = "Ctrl+A"     # 全选
    FIND = "Ctrl+F"           # 查找
    NEW = "Ctrl+N"            # 新建
    OPEN = "Ctrl+O"           # 打开
    CLOSE_TAB = "Ctrl+W"      # 关闭标签页
    CLOSE_WINDOW = "Alt+F4"   # 关闭窗口
    SWITCH_APP = "Alt+Tab"    # 切换应用
    TASK_MANAGER = "Ctrl+Shift+Esc"  # 任务管理器
    SCREENSHOT = "Win+Shift+S"       # 截图
    LOCK_SCREEN = "Win+L"            # 锁屏
    SHOW_DESKTOP = "Win+D"           # 显示桌面
    FILE_EXPLORER = "Win+E"          # 打开文件资源管理器
    SETTINGS = "Win+I"               # 打开设置
    RENAME = "F2"                    # 重命名


# ==================== 技能定义 ====================

@dataclass
class Skill:
    """单个技能（原子操作）"""
    skill_type: SkillType
    target: str = ""                    # 目标元素描述
    target_position: str = ""           # 目标位置（用于拖动）
    text: str = ""                      # 输入文本
    key: str = ""                       # 按键
    hotkey: str = ""                    # 组合键
    wait_seconds: float = 0.0           # 等待秒数
    
    # 元数据
    friendly_description: str = ""      # 老年人友好描述
    expected_result: str = ""           # 预期结果
    verification_method: str = ""       # 验证方式
    
    def to_instruction(self) -> str:
        """转换为标准指令格式"""
        if self.skill_type == SkillType.CLICK:
            return f"单击{{{self.target}}}"
        elif self.skill_type == SkillType.DOUBLE_CLICK:
            return f"双击{{{self.target}}}"
        elif self.skill_type == SkillType.RIGHT_CLICK:
            return f"右键单击{{{self.target}}}"
        elif self.skill_type == SkillType.DRAG:
            return f"拖动{{{self.target}}}至{{{self.target_position}}}"
        elif self.skill_type == SkillType.SCROLL_UP:
            return f"向上滚动{{{self.target}}}"
        elif self.skill_type == SkillType.SCROLL_DOWN:
            return f"向下滚动{{{self.target}}}"
        elif self.skill_type == SkillType.TYPE:
            return f"输入{{{self.text}}}"
        elif self.skill_type == SkillType.PRESS:
            return f"按下{{{self.key}}}"
        elif self.skill_type == SkillType.HOTKEY:
            return f"组合键{{{self.hotkey}}}"
        elif self.skill_type == SkillType.WAIT_TIME:
            return f"等待{{{self.wait_seconds}秒}}"
        elif self.skill_type == SkillType.WAIT_ELEMENT:
            return f"等待{{{self.target}}}出现"
        return ""
    
    def to_friendly_instruction(self) -> str:
        """转换为老年人友好的指令"""
        if self.friendly_description:
            return self.friendly_description
        return self.to_instruction()


@dataclass
class SkillStep:
    """技能步骤（包含技能和上下文）"""
    step_number: int
    skill: Skill
    precondition: str = ""              # 前置条件
    postcondition: str = ""             # 后置条件
    error_recovery: str = ""            # 错误恢复提示
    visual_hint: str = ""               # 视觉提示（帮助用户找到目标）


@dataclass
class SkillPlan:
    """技能计划（一系列技能步骤）"""
    goal: str = ""                      # 目标描述
    steps: list[SkillStep] = field(default_factory=list)
    total_estimated_time: float = 0.0   # 预计总时间（秒）


# ==================== 技能集定义（Prompt 用）====================

SKILL_SET_PROMPT = """
## 可用操作技能集

你只能使用以下标准化操作，不能使用其他任何操作：

### 鼠标操作
1. 单击{目标} - 用鼠标左键点击一次
2. 双击{目标} - 用鼠标左键快速点击两次
3. 右键单击{目标} - 用鼠标右键点击一次
4. 拖动{对象}至{目标位置} - 按住鼠标左键拖动

### 滚动操作
5. 向上滚动{区域} - 在指定区域向上滚动
6. 向下滚动{区域} - 在指定区域向下滚动

### 键盘操作
7. 输入{文本内容} - 用键盘输入文字
8. 按下{按键} - 按下单个按键
9. 组合键{按键1}+{按键2} - 同时按下多个按键

### 等待操作
10. 等待{秒数} - 等待指定时间
11. 等待{元素}出现 - 等待某个元素出现在屏幕上

## 系统元素（可作为{目标}使用）

### 窗口控制（右上角）
- 关闭按钮（×）
- 最小化按钮（-）
- 最大化按钮（□）

### 任务栏（屏幕底部）
- 开始按钮（Windows图标，左下角）
- 搜索框（开始按钮旁边）
- 任务栏图标（各应用图标）
- 系统托盘（右下角，时间旁边）

### 通用控件
- 确定按钮、取消按钮、是按钮、否按钮
- 文本输入框、密码输入框
- 下拉菜单、复选框、单选按钮
- 滚动条、菜单栏、标题栏

## 常用按键
- 回车键、Esc键、Tab键、退格键、删除键、空格键
- 上/下/左/右箭头
- Ctrl键、Alt键、Shift键、Windows键
- F1-F12功能键

## 常用组合键
- Ctrl+C（复制）、Ctrl+V（粘贴）、Ctrl+X（剪切）
- Ctrl+Z（撤销）、Ctrl+S（保存）、Ctrl+A（全选）
- Alt+F4（关闭窗口）、Alt+Tab（切换应用）
- Win+D（显示桌面）、Win+E（打开文件资源管理器）
- F2（重命名）

## 输出格式要求

每个步骤必须严格按照以下 JSON 格式输出：

```json
{
  "steps": [
    {
      "step_number": 1,
      "skill_type": "单击",
      "target": "开始按钮",
      "visual_hint": "屏幕左下角的Windows图标",
      "expected_result": "开始菜单弹出",
      "friendly_description": "请点击屏幕左下角的Windows图标（开始按钮）"
    },
    {
      "step_number": 2,
      "skill_type": "输入",
      "text": "记事本",
      "expected_result": "搜索结果显示记事本应用",
      "friendly_description": "在搜索框中输入"记事本""
    }
  ]
}
```

## 重要规则

1. **只能使用上述技能**：不要发明新的操作
2. **目标必须具体**：不要说"找到某个按钮"，要说"单击{关闭按钮}"
3. **提供视觉提示**：告诉用户目标在屏幕的什么位置
4. **一步一操作**：每个步骤只能包含一个原子操作
5. **预期结果明确**：每步都要说明操作后应该看到什么
"""


# ==================== 常见任务的标准技能序列 ====================

COMMON_SKILL_SEQUENCES = {
    "打开开始菜单": [
        Skill(SkillType.CLICK, target="开始按钮", 
              friendly_description="点击屏幕左下角的Windows图标",
              expected_result="开始菜单弹出")
    ],
    
    "关闭当前窗口": [
        Skill(SkillType.CLICK, target="关闭按钮",
              friendly_description="点击窗口右上角的×按钮",
              expected_result="窗口关闭")
    ],
    
    "最小化当前窗口": [
        Skill(SkillType.CLICK, target="最小化按钮",
              friendly_description="点击窗口右上角的-按钮",
              expected_result="窗口最小化到任务栏")
    ],
    
    "复制选中内容": [
        Skill(SkillType.HOTKEY, hotkey="Ctrl+C",
              friendly_description="同时按下Ctrl键和C键",
              expected_result="内容已复制到剪贴板")
    ],
    
    "粘贴内容": [
        Skill(SkillType.HOTKEY, hotkey="Ctrl+V",
              friendly_description="同时按下Ctrl键和V键",
              expected_result="剪贴板内容被粘贴")
    ],
    
    "重命名文件": [
        Skill(SkillType.CLICK, target="目标文件",
              friendly_description="先单击选中要重命名的文件",
              expected_result="文件被选中（高亮显示）"),
        Skill(SkillType.PRESS, key="F2",
              friendly_description="按下F2键",
              expected_result="文件名变为可编辑状态"),
        Skill(SkillType.TYPE, text="新文件名",
              friendly_description="输入新的文件名",
              expected_result="显示输入的新文件名"),
        Skill(SkillType.PRESS, key="回车键",
              friendly_description="按下回车键确认",
              expected_result="文件名修改完成")
    ],
    
    "搜索应用": [
        Skill(SkillType.CLICK, target="搜索框",
              friendly_description="点击任务栏上的搜索框",
              expected_result="搜索框获得焦点"),
        Skill(SkillType.TYPE, text="应用名称",
              friendly_description="输入要搜索的应用名称",
              expected_result="显示搜索结果"),
        Skill(SkillType.CLICK, target="搜索结果中的应用",
              friendly_description="点击搜索结果中显示的应用",
              expected_result="应用启动")
    ],
}


def get_skill_set_prompt() -> str:
    """获取技能集提示词"""
    return SKILL_SET_PROMPT


def get_common_sequence(task_name: str) -> list[Skill]:
    """获取常见任务的标准技能序列"""
    return COMMON_SKILL_SEQUENCES.get(task_name, [])
