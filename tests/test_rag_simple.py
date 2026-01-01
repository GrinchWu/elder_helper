#!/usr/bin/env python3
"""
测试 RAG 服务逻辑
使用模拟服务避免外部依赖
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.knowledge.rag_service import RAGService, RAGResult
from src.models.knowledge import KnowledgeNode, OperationGuide, KnowledgeGraph, NodeType
from src.services.embedding_service import EmbeddingService


def create_mock_embedding_service():
    """创建模拟的嵌入服务"""
    service = MagicMock(spec=EmbeddingService)
    service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    service.cosine_similarity = MagicMock(return_value=0.85)
    return service


def create_mock_knowledge_graph():
    """创建模拟的知识图谱"""
    kg = MagicMock(spec=KnowledgeGraph)
    kg._guides = {}
    kg._nodes = {}
    kg._graph = MagicMock()  # 模拟 NetworkX 图
    kg._graph.neighbors = MagicMock(return_value=[])
    kg._graph.edges = MagicMock(return_value={})
    
    def add_guide(g): kg._guides[g.id] = g
    def add_node(n): kg._nodes[n.id] = n
    
    kg.add_guide = MagicMock(side_effect=add_guide)
    kg.add_node = MagicMock(side_effect=add_node)
    return kg


@pytest.mark.asyncio
async def test_rag_service_initialization():
    """测试 RAG 服务初始化"""
    rag = RAGService()
    embedding_service = create_mock_embedding_service()
    knowledge_graph = create_mock_knowledge_graph()
    
    await rag.initialize(embedding_service, knowledge_graph)
    
    assert rag._embedding_service is not None
    assert rag._knowledge_graph is not None


@pytest.mark.asyncio
async def test_rag_service_indexing():
    """测试索引指南和节点"""
    rag = RAGService()
    embedding_service = create_mock_embedding_service()
    knowledge_graph = create_mock_knowledge_graph()
    
    await rag.initialize(embedding_service, knowledge_graph)
    
    guide = OperationGuide(
        title="微信发消息",
        app_name="微信",
        feature_name="发消息",
        steps=["打开微信", "点击联系人", "输入消息", "点击发送"],
        friendly_steps=["先打开微信 app", "找到要聊天的朋友", "写下你要说的话", "按发送按钮"],
        quality_score=0.9,
    )
    
    node = KnowledgeNode(
        node_type=NodeType.FEATURE,
        name="发消息",
        description="在微信中向联系人发送文字消息的功能",
        aliases=["发信息", "聊天", "发文字"]
    )
    
    await rag.index_guide(guide)
    await rag.index_node(node)
    
    assert guide.id in rag._guide_embeddings
    assert node.id in rag._node_embeddings
    assert guide.embedding is not None
    assert node.embedding is not None


@pytest.mark.asyncio
async def test_rag_service_retrieve():
    """测试基本检索"""
    rag = RAGService()
    embedding_service = create_mock_embedding_service()
    knowledge_graph = create_mock_knowledge_graph()
    
    await rag.initialize(embedding_service, knowledge_graph)
    
    guide = OperationGuide(
        title="微信发消息",
        app_name="微信",
        feature_name="发消息",
        steps=["打开微信", "点击联系人", "输入消息", "点击发送"],
        quality_score=0.9,
    )
    
    node = KnowledgeNode(
        node_type=NodeType.FEATURE,
        name="发消息",
        description="在微信中向联系人发送文字消息的功能",
    )
    
    await rag.index_guide(guide)
    await rag.index_node(node)
    
    result = await rag.retrieve("怎么发微信消息", top_k=5, min_score=0.5)
    
    assert isinstance(result, RAGResult)
    assert len(result.guides) > 0 or len(result.nodes) > 0
    assert result.confidence >= 0.0
    assert isinstance(result.context, str)


@pytest.mark.asyncio
async def test_rag_service_expand_query():
    """测试查询扩展"""
    rag = RAGService()
    
    expanded = await rag.expand_query("手机吃钱")
    
    assert "手机吃钱" in expanded
    assert any("话费" in q for q in expanded)
    assert any("流量" in q for q in expanded)


@pytest.mark.asyncio
async def test_rag_service_retrieve_with_expansion():
    """测试带扩展的检索"""
    rag = RAGService()
    embedding_service = create_mock_embedding_service()
    knowledge_graph = create_mock_knowledge_graph()
    
    await rag.initialize(embedding_service, knowledge_graph)
    
    guide = OperationGuide(
        title="查询话费",
        app_name="手机",
        feature_name="话费查询",
        steps=["打开设置", "点击流量和话费"],
        quality_score=0.8,
    )
    
    await rag.index_guide(guide)
    
    result = await rag.retrieve_with_expansion("手机吃钱", top_k=5, min_score=0.5)
    
    assert isinstance(result, RAGResult)
    # 扩展查询应能找到相关指南


@pytest.mark.asyncio
async def test_rag_service_retrieve_hybrid():
    """测试混合检索"""
    rag = RAGService()
    embedding_service = create_mock_embedding_service()
    knowledge_graph = create_mock_knowledge_graph()
    
    # 设置图的邻居
    knowledge_graph._graph.neighbors.return_value = []
    
    await rag.initialize(embedding_service, knowledge_graph)
    
    guide = OperationGuide(
        title="微信发消息",
        app_name="微信",
        feature_name="发消息",
        steps=["打开微信", "点击联系人", "输入消息", "点击发送"],
        quality_score=0.9,
    )
    
    node = KnowledgeNode(
        node_type=NodeType.FEATURE,
        name="发消息",
        description="在微信中向联系人发送文字消息的功能",
    )
    
    await rag.index_guide(guide)
    await rag.index_node(node)
    
    result = await rag.retrieve_hybrid("怎么发微信消息", top_k=5, min_score=0.5)
    
    assert isinstance(result, RAGResult)
    assert "paths" in result.__dict__ or hasattr(result, 'paths')