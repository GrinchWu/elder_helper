"""知识库测试"""

import pytest
from uuid import uuid4

from src.models.knowledge import (
    KnowledgeNode, KnowledgeGraph, OperationGuide,
    NodeType, EdgeType
)


class TestKnowledgeNode:
    """知识节点测试"""
    
    def test_matches_query(self):
        """测试查询匹配"""
        node = KnowledgeNode(
            name="微信",
            description="即时通讯软件",
            aliases=["WeChat", "那个绿色的"],
        )
        
        assert node.matches_query("微信")
        assert node.matches_query("wechat")
        assert node.matches_query("绿色")
        assert not node.matches_query("支付宝")


class TestKnowledgeGraph:
    """知识图谱测试"""
    
    @pytest.fixture
    def knowledge_graph(self):
        kg = KnowledgeGraph()
        
        # 添加节点
        wechat = KnowledgeNode(
            id=uuid4(),
            node_type=NodeType.APP,
            name="微信",
            description="即时通讯软件",
        )
        video_call = KnowledgeNode(
            id=uuid4(),
            node_type=NodeType.FEATURE,
            name="视频通话",
            description="与好友视频聊天",
        )
        
        kg.add_node(wechat)
        kg.add_node(video_call)
        kg.add_edge(wechat.id, video_call.id, EdgeType.HAS_FEATURE)
        
        return kg
    
    def test_add_node(self, knowledge_graph):
        """测试添加节点"""
        assert len(knowledge_graph._nodes) == 2
    
    def test_add_guide(self, knowledge_graph):
        """测试添加指南"""
        guide = OperationGuide(
            title="如何用微信视频通话",
            app_name="微信",
            feature_name="视频通话",
            steps=["打开微信", "点击通讯录", "选择联系人", "点击视频通话"],
        )
        
        knowledge_graph.add_guide(guide)
        assert len(knowledge_graph._guides) == 1
    
    def test_search_guides(self, knowledge_graph):
        """测试搜索指南"""
        guide1 = OperationGuide(
            title="如何用微信视频通话",
            app_name="微信",
            feature_name="视频通话",
        )
        guide2 = OperationGuide(
            title="如何用支付宝付款",
            app_name="支付宝",
            feature_name="付款",
        )
        
        knowledge_graph.add_guide(guide1)
        knowledge_graph.add_guide(guide2)
        
        results = knowledge_graph.search_guides("微信")
        assert len(results) == 1
        assert results[0].app_name == "微信"
    
    def test_merge_guides(self, knowledge_graph):
        """测试合并指南"""
        guide1 = OperationGuide(
            title="微信视频通话方法1",
            app_name="微信",
            steps=["打开微信", "点击通讯录"],
            friendly_steps=["打开那个绿色的软件", "点击通讯录"],
        )
        guide2 = OperationGuide(
            title="微信视频通话方法2",
            app_name="微信",
            steps=["打开微信", "选择联系人", "点击视频"],
            friendly_steps=["打开那个绿色的软件", "找到要联系的人", "点击视频图标"],
        )
        
        merged = knowledge_graph.merge_guides([guide1, guide2])
        
        # 应该去重
        assert "打开微信" in merged.steps
        assert merged.steps.count("打开微信") == 1


class TestOperationGuide:
    """操作指南测试"""
    
    def test_guide_creation(self):
        """测试指南创建"""
        guide = OperationGuide(
            title="如何发微信消息",
            app_name="微信",
            feature_name="发消息",
            steps=["打开微信", "点击聊天", "输入消息", "点击发送"],
            friendly_steps=[
                "打开那个绿色的软件",
                "点击要聊天的人",
                "在下面的框里打字",
                "点击发送按钮",
            ],
        )
        
        assert guide.title == "如何发微信消息"
        assert len(guide.steps) == 4
        assert len(guide.friendly_steps) == 4
