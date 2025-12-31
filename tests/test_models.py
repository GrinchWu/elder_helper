"""模型测试"""

import pytest
from uuid import uuid4

from src.models.intent import Intent, IntentType, Confidence
from src.models.action import Action, ActionType, ActionResult, ActionStatus
from src.models.task import Task, TaskStep, TaskPlan, TaskStatus
from src.models.session import Session, UserProfile


class TestConfidence:
    """置信度测试"""
    
    def test_valid_confidence(self):
        """测试有效置信度"""
        c = Confidence(0.5)
        assert c.value == 0.5
        assert c.is_medium
    
    def test_high_confidence(self):
        """测试高置信度"""
        c = Confidence(0.9)
        assert c.is_high
        assert not c.is_medium
        assert not c.is_low
    
    def test_low_confidence(self):
        """测试低置信度"""
        c = Confidence(0.3)
        assert c.is_low
        assert not c.is_high
    
    def test_invalid_confidence(self):
        """测试无效置信度"""
        with pytest.raises(ValueError):
            Confidence(1.5)
        
        with pytest.raises(ValueError):
            Confidence(-0.1)


class TestIntent:
    """意图测试"""
    
    def test_normalize_elderly_language(self):
        """测试老年人语言标准化"""
        intent = Intent()
        
        result = intent.normalize_elderly_language("手机吃钱了")
        assert "流量超标" in result or "扣费" in result
        
        result = intent.normalize_elderly_language("屏幕上有脏东西关不掉")
        assert "悬浮窗广告" in result


class TestAction:
    """动作测试"""
    
    def test_action_result_ok(self):
        """测试成功结果"""
        result = ActionResult.ok("操作成功")
        assert result.success
        assert result.message == "操作成功"
    
    def test_action_result_fail(self):
        """测试失败结果"""
        result = ActionResult.fail("操作失败", "ERR001")
        assert not result.success
        assert result.error_code == "ERR001"
    
    def test_friendly_description(self):
        """测试友好描述"""
        action = Action(
            action_type=ActionType.CLICK,
            element_description="确定按钮",
        )
        desc = action.to_friendly_description()
        assert "确定按钮" in desc


class TestTaskPlan:
    """任务计划测试"""
    
    def test_progress_percentage(self):
        """测试进度百分比"""
        plan = TaskPlan()
        
        # 空计划
        assert plan.progress_percentage == 0.0
        
        # 添加步骤
        plan.steps = [
            TaskStep(step_number=1, status=ActionStatus.SUCCESS),
            TaskStep(step_number=2, status=ActionStatus.SUCCESS),
            TaskStep(step_number=3, status=ActionStatus.PENDING),
            TaskStep(step_number=4, status=ActionStatus.PENDING),
        ]
        
        assert plan.progress_percentage == 50.0
    
    def test_advance_step(self):
        """测试步骤前进"""
        plan = TaskPlan()
        plan.steps = [
            TaskStep(step_number=1),
            TaskStep(step_number=2),
            TaskStep(step_number=3),
        ]
        
        assert plan.current_step_index == 0
        
        next_step = plan.advance_to_next_step()
        assert next_step is not None
        assert plan.current_step_index == 1
        
        plan.advance_to_next_step()
        next_step = plan.advance_to_next_step()
        assert next_step is None  # 已经是最后一步
    
    def test_rollback_step(self):
        """测试步骤回退"""
        plan = TaskPlan()
        plan.steps = [
            TaskStep(step_number=1),
            TaskStep(step_number=2),
        ]
        plan.current_step_index = 1
        
        prev_step = plan.rollback_to_previous_step()
        assert prev_step is not None
        assert plan.current_step_index == 0
        
        prev_step = plan.rollback_to_previous_step()
        assert prev_step is None  # 已经是第一步


class TestUserProfile:
    """用户画像测试"""
    
    def test_resolve_family_reference(self):
        """测试家庭成员引用解析"""
        profile = UserProfile()
        profile.family_mapping = {"老二": "张三"}
        
        # 自定义映射
        assert profile.resolve_family_reference("老二") == "张三"
        
        # 通用映射
        result = profile.resolve_family_reference("闺女")
        assert result == "女儿"
    
    def test_update_anxiety_index(self):
        """测试焦虑指数更新"""
        profile = UserProfile()
        profile.anxiety_index = 0.5
        
        # 成功降低焦虑
        profile.update_anxiety_index(task_success=True)
        assert profile.anxiety_index < 0.5
        assert profile.completed_tasks_count == 1
        
        # 失败增加焦虑
        profile.anxiety_index = 0.5
        profile.update_anxiety_index(task_success=False)
        assert profile.anxiety_index > 0.5
        assert profile.failed_tasks_count == 1


class TestSession:
    """会话测试"""
    
    def test_add_conversation(self):
        """测试添加对话"""
        session = Session()
        
        session.add_conversation("user", "你好")
        session.add_conversation("assistant", "您好！有什么可以帮您的？")
        
        assert len(session.conversation_history) == 2
        assert session.conversation_history[0]["role"] == "user"
    
    def test_conversation_history_limit(self):
        """测试对话历史限制"""
        session = Session()
        session.max_history_length = 3
        
        for i in range(5):
            session.add_conversation("user", f"消息{i}")
        
        assert len(session.conversation_history) == 3
        assert "消息2" in session.conversation_history[0]["content"]
