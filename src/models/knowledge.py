"""知识图谱模型 - 用于存储操作知识"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

import networkx as nx


class NodeType(str, Enum):
    """知识节点类型"""
    APP = "app"                      # 应用
    FEATURE = "feature"              # 功能
    OPERATION = "operation"          # 操作
    STEP = "step"                    # 步骤
    UI_ELEMENT = "ui_element"        # UI元素
    CONCEPT = "concept"              # 概念


class EdgeType(str, Enum):
    """边类型"""
    HAS_FEATURE = "has_feature"      # 应用->功能
    REQUIRES = "requires"            # 功能->操作
    NEXT_STEP = "next_step"          # 步骤->步骤
    ALTERNATIVE = "alternative"      # 替代方案
    CONTAINS = "contains"            # 包含
    SIMILAR_TO = "similar_to"        # 相似


@dataclass
class KnowledgeNode:
    """知识节点"""
    id: UUID = field(default_factory=uuid4)
    node_type: NodeType = NodeType.CONCEPT
    name: str = ""
    description: str = ""
    
    # 老年人友好的别名
    aliases: list[str] = field(default_factory=list)
    
    # 向量嵌入 (用于语义搜索)
    embedding: Optional[list[float]] = None
    
    # 来源信息
    source_video_id: Optional[str] = None
    source_url: Optional[str] = None
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    def matches_query(self, query: str) -> bool:
        """检查是否匹配查询"""
        query_lower = query.lower()
        if query_lower in self.name.lower():
            return True
        return any(query_lower in alias.lower() for alias in self.aliases)


@dataclass
class OperationGuide:
    """操作指南 - 从视频提取的操作步骤"""
    id: UUID = field(default_factory=uuid4)
    title: str = ""
    app_name: str = ""
    feature_name: str = ""
    
    # 步骤列表
    steps: list[str] = field(default_factory=list)
    
    # 老年人友好的步骤描述
    friendly_steps: list[str] = field(default_factory=list)
    
    # 常见问题和解决方案
    faq: dict[str, str] = field(default_factory=dict)
    
    # 来源
    source_video_id: str = ""
    source_url: str = ""
    
    # 质量评分 (0-1)
    quality_score: float = 0.0
    
    # 向量嵌入
    embedding: Optional[list[float]] = None
    
    created_at: datetime = field(default_factory=datetime.now)


class KnowledgeGraph:
    """知识图谱 - 使用思维导图方式压缩上下文"""
    
    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[UUID, KnowledgeNode] = {}
        self._guides: dict[UUID, OperationGuide] = {}
    
    def add_node(self, node: KnowledgeNode) -> None:
        """添加节点"""
        self._nodes[node.id] = node
        self._graph.add_node(
            str(node.id),
            node_type=node.node_type.value,
            name=node.name,
            description=node.description,
        )
    
    def add_edge(
        self,
        from_node_id: UUID,
        to_node_id: UUID,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> None:
        """添加边"""
        self._graph.add_edge(
            str(from_node_id),
            str(to_node_id),
            edge_type=edge_type.value,
            weight=weight,
        )
    
    def add_guide(self, guide: OperationGuide) -> None:
        """添加操作指南"""
        self._guides[guide.id] = guide
    
    def find_operation_path(
        self,
        app_name: str,
        feature_name: str,
    ) -> list[KnowledgeNode]:
        """查找操作路径"""
        # 找到应用节点
        app_node = self._find_node_by_name(app_name, NodeType.APP)
        if not app_node:
            return []
        
        # 找到功能节点
        feature_node = self._find_node_by_name(feature_name, NodeType.FEATURE)
        if not feature_node:
            return []
        
        # 查找路径
        try:
            path = nx.shortest_path(
                self._graph,
                str(app_node.id),
                str(feature_node.id),
            )
            return [self._nodes[UUID(node_id)] for node_id in path]
        except nx.NetworkXNoPath:
            return []
    
    def _find_node_by_name(
        self,
        name: str,
        node_type: Optional[NodeType] = None,
    ) -> Optional[KnowledgeNode]:
        """按名称查找节点"""
        for node in self._nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if node.matches_query(name):
                return node
        return None
    
    def search_guides(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[OperationGuide]:
        """搜索操作指南"""
        # 简单的关键词匹配，实际应使用向量搜索
        results: list[tuple[float, OperationGuide]] = []
        query_lower = query.lower()
        
        for guide in self._guides.values():
            score = 0.0
            if query_lower in guide.title.lower():
                score += 0.5
            if query_lower in guide.app_name.lower():
                score += 0.3
            if query_lower in guide.feature_name.lower():
                score += 0.2
            
            if score > 0:
                results.append((score, guide))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [guide for _, guide in results[:top_k]]
    
    def merge_guides(
        self,
        guides: list[OperationGuide],
    ) -> OperationGuide:
        """合并多个指南，压缩上下文"""
        if not guides:
            raise ValueError("至少需要一个指南")
        
        if len(guides) == 1:
            return guides[0]
        
        # 合并步骤，去重
        merged_steps: list[str] = []
        merged_friendly: list[str] = []
        seen_steps: set[str] = set()
        
        for guide in guides:
            for i, step in enumerate(guide.steps):
                step_key = step.lower().strip()
                if step_key not in seen_steps:
                    seen_steps.add(step_key)
                    merged_steps.append(step)
                    if i < len(guide.friendly_steps):
                        merged_friendly.append(guide.friendly_steps[i])
        
        # 合并FAQ
        merged_faq: dict[str, str] = {}
        for guide in guides:
            merged_faq.update(guide.faq)
        
        return OperationGuide(
            title=guides[0].title,
            app_name=guides[0].app_name,
            feature_name=guides[0].feature_name,
            steps=merged_steps,
            friendly_steps=merged_friendly,
            faq=merged_faq,
            quality_score=sum(g.quality_score for g in guides) / len(guides),
        )
    
    def to_mindmap_context(self, root_node_id: UUID, max_depth: int = 3) -> str:
        """将知识图谱转换为思维导图格式的上下文"""
        if str(root_node_id) not in self._graph:
            return ""
        
        lines: list[str] = []
        visited: set[str] = set()
        
        def traverse(node_id: str, depth: int, prefix: str) -> None:
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            node = self._nodes.get(UUID(node_id))
            if not node:
                return
            
            lines.append(f"{prefix}- {node.name}: {node.description}")
            
            for neighbor in self._graph.neighbors(node_id):
                edge_data = self._graph.edges[node_id, neighbor]
                edge_type = edge_data.get("edge_type", "")
                traverse(neighbor, depth + 1, prefix + "  ")
        
        traverse(str(root_node_id), 0, "")
        return "\n".join(lines)
