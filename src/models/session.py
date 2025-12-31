"""会话模型 - 包含用户画像和BDI心智模型"""

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


class TechLevel(str, Enum):
    """技术熟练度"""
    NOVICE = "novice"           # 完全不会，需要手把手教
    BEGINNER = "beginner"       # 会基本操作，但容易忘记
    INTERMEDIATE = "intermediate"  # 能独立完成简单任务
    ADVANCED = "advanced"       # 较熟练，偶尔需要帮助


class CognitiveStyle(str, Enum):
    """认知风格"""
    VISUAL = "visual"           # 视觉型：喜欢看图示、颜色提示
    AUDITORY = "auditory"       # 听觉型：喜欢语音指导
    KINESTHETIC = "kinesthetic" # 动觉型：喜欢动手尝试


@dataclass
class FamilyMember:
    """家庭成员信息"""
    nickname: str = ""          # 称呼（老大、闺女）
    real_name: str = ""         # 真实姓名
    relationship: str = ""      # 关系（儿子、女儿、配偶）
    contact_app: str = ""       # 常用联系方式（微信、电话）
    contact_id: str = ""        # 联系方式ID（微信号、电话号码）


@dataclass
class AppUsageRecord:
    """应用使用记录"""
    app_name: str = ""
    last_used: Optional[datetime] = None
    use_count: int = 0
    common_tasks: list[str] = field(default_factory=list)  # 常做的操作
    difficulties: list[str] = field(default_factory=list)  # 遇到的困难


@dataclass
class UserProfile:
    """
    用户画像 - 用于BDI心智理论建模
    
    BDI模型组成：
    - Beliefs (信念): 用户对世界的认知和理解
    - Desires (愿望): 用户想要达成的目标
    - Intentions (意图): 用户计划采取的行动
    
    用户画像帮助我们推断这三个要素
    """
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    age: int = 65
    
    # ========== 基础信息 ==========
    gender: str = ""                    # 性别
    education: str = ""                 # 教育背景
    occupation_before_retire: str = ""  # 退休前职业
    
    # ========== 技术能力评估 (影响Beliefs) ==========
    tech_level: TechLevel = TechLevel.BEGINNER
    cognitive_style: CognitiveStyle = CognitiveStyle.VISUAL
    
    # 已掌握的技能
    known_skills: list[str] = field(default_factory=list)
    # 例如: ["打开微信", "发送文字消息", "接听视频通话"]
    
    # 常见误解/错误认知 (Beliefs中的错误部分)
    misconceptions: list[str] = field(default_factory=list)
    # 例如: ["认为关闭窗口会删除文件", "认为点错按钮会弄坏电脑"]
    
    # ========== 家庭与社交 (影响Desires) ==========
    family_members: list[FamilyMember] = field(default_factory=list)
    
    # 家庭称呼映射 (老二 -> 张三)
    family_mapping: dict[str, str] = field(default_factory=dict)
    
    # 常用联系人
    frequent_contacts: list[str] = field(default_factory=list)
    
    # ========== 应用使用情况 (影响Intentions) ==========
    # 已安装的应用
    installed_apps: list[str] = field(default_factory=list)
    
    # 常用应用及使用记录
    app_usage: dict[str, AppUsageRecord] = field(default_factory=dict)
    
    # 常用应用（简化版）
    frequent_apps: list[str] = field(default_factory=list)
    
    # ========== 日常需求模式 (影响Desires) ==========
    # 日常任务模式
    daily_patterns: dict[str, list[str]] = field(default_factory=dict)
    # 例如: {"早上": ["看新闻", "查天气"], "晚上": ["和家人视频"]}
    
    # 兴趣爱好
    interests: list[str] = field(default_factory=list)
    # 例如: ["看戏曲", "养生知识", "新闻时事"]
    
    # ========== 交互偏好 ==========
    preferred_voice_speed: float = 0.8  # 语速 (0.5-1.5)
    preferred_font_size: str = "large"  # small, medium, large, xlarge
    preferred_volume: float = 0.9       # 音量 (0-1)
    
    # ========== 心理状态 (影响所有BDI) ==========
    # 焦虑指数 (0-1, 越低越好)
    anxiety_index: float = 0.5
    
    # 自我效能感 (0-1, 越高越好) - 对自己能完成任务的信心
    self_efficacy: float = 0.5
    
    # 技术恐惧程度 (0-1, 越低越好)
    tech_anxiety: float = 0.5
    
    # ========== 历史记录 ==========
    completed_tasks_count: int = 0
    failed_tasks_count: int = 0
    
    # 成功完成的任务类型（用于推断能力）
    successful_task_types: list[str] = field(default_factory=list)
    
    # 失败/放弃的任务类型（用于识别困难点）
    failed_task_types: list[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def resolve_family_reference(self, reference: str) -> Optional[str]:
        """解析家庭成员引用"""
        # 先查用户自定义映射
        if reference in self.family_mapping:
            return self.family_mapping[reference]
        
        # 查找家庭成员列表
        for member in self.family_members:
            if member.nickname == reference:
                return member.real_name
        
        # 通用映射
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
        return common_refs.get(reference)
    
    def get_family_contact(self, reference: str) -> Optional[FamilyMember]:
        """获取家庭成员联系方式"""
        for member in self.family_members:
            if member.nickname == reference or member.real_name == reference:
                return member
        return None
    
    def update_anxiety_index(self, task_success: bool) -> None:
        """更新焦虑指数和自我效能感"""
        if task_success:
            # 成功降低焦虑，提高自我效能
            self.anxiety_index = max(0.0, self.anxiety_index - 0.05)
            self.self_efficacy = min(1.0, self.self_efficacy + 0.05)
            self.tech_anxiety = max(0.0, self.tech_anxiety - 0.03)
            self.completed_tasks_count += 1
        else:
            # 失败增加焦虑，降低自我效能
            self.anxiety_index = min(1.0, self.anxiety_index + 0.1)
            self.self_efficacy = max(0.0, self.self_efficacy - 0.08)
            self.tech_anxiety = min(1.0, self.tech_anxiety + 0.05)
            self.failed_tasks_count += 1
        self.updated_at = datetime.now()
    
    def record_task_result(self, task_type: str, success: bool) -> None:
        """记录任务结果"""
        if success:
            if task_type not in self.successful_task_types:
                self.successful_task_types.append(task_type)
        else:
            if task_type not in self.failed_task_types:
                self.failed_task_types.append(task_type)
        self.update_anxiety_index(success)
    
    def get_bdi_summary(self) -> dict:
        """
        获取BDI心智模型摘要
        用于SimToM推理
        """
        return {
            "beliefs": {
                "tech_level": self.tech_level.value,
                "known_skills": self.known_skills,
                "misconceptions": self.misconceptions,
                "self_efficacy": self.self_efficacy,
                "tech_anxiety": self.tech_anxiety,
            },
            "desires": {
                "interests": self.interests,
                "daily_patterns": self.daily_patterns,
                "frequent_contacts": self.frequent_contacts,
                "family_members": [m.nickname for m in self.family_members],
            },
            "intentions": {
                "frequent_apps": self.frequent_apps,
                "successful_tasks": self.successful_task_types,
                "failed_tasks": self.failed_task_types,
            }
        }
    
    def to_context_string(self) -> str:
        """转换为上下文字符串，用于LLM"""
        parts = []
        
        if self.name:
            parts.append(f"用户：{self.name}")
        if self.age:
            parts.append(f"年龄：{self.age}岁")
        
        parts.append(f"技术水平：{self._tech_level_desc()}")
        parts.append(f"认知风格：{self._cognitive_style_desc()}")
        
        if self.family_mapping:
            family_str = "、".join([f"{k}={v}" for k, v in self.family_mapping.items()])
            parts.append(f"家庭成员：{family_str}")
        
        if self.frequent_apps:
            parts.append(f"常用应用：{', '.join(self.frequent_apps)}")
        
        if self.interests:
            parts.append(f"兴趣爱好：{', '.join(self.interests)}")
        
        if self.known_skills:
            parts.append(f"已掌握技能：{', '.join(self.known_skills[:5])}")
        
        parts.append(f"焦虑指数：{self.anxiety_index:.1f}")
        parts.append(f"自我效能感：{self.self_efficacy:.1f}")
        
        return "\n".join(parts)
    
    def _tech_level_desc(self) -> str:
        """技术水平描述"""
        descs = {
            TechLevel.NOVICE: "完全不会，需要手把手教",
            TechLevel.BEGINNER: "会基本操作，但容易忘记",
            TechLevel.INTERMEDIATE: "能独立完成简单任务",
            TechLevel.ADVANCED: "较熟练，偶尔需要帮助",
        }
        return descs.get(self.tech_level, "未知")
    
    def _cognitive_style_desc(self) -> str:
        """认知风格描述"""
        descs = {
            CognitiveStyle.VISUAL: "视觉型（喜欢看图示、颜色提示）",
            CognitiveStyle.AUDITORY: "听觉型（喜欢语音指导）",
            CognitiveStyle.KINESTHETIC: "动觉型（喜欢动手尝试）",
        }
        return descs.get(self.cognitive_style, "未知")


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
