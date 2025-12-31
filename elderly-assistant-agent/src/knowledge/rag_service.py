"""RAG服务 - 检索增强生成"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from loguru import logger

from ..config import config
from ..models.knowledge import KnowledgeGraph, KnowledgeNode, OperationGuide, NodeType, EdgeType
from ..services.embedding_service import EmbeddingService


@dataclass
class RAGResult:
    """RAG检索结果"""
    guides: list[OperationGuide] = field(default_factory=list)
    nodes: list[KnowledgeNode] = field(default_factory=list)
    context: str = ""
    confidence: float = 0.0


class RAGService:
    """RAG服务 - 从知识库检索相关信息"""
    
    def __init__(self) -> None:
        self._embedding_service: Optional[EmbeddingService] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        
        # 缓存
        self._guide_embeddings: dict[UUID, list[float]] = {}
        self._node_embeddings: dict[UUID, list[float]] = {}
    
    async def initialize(
        self,
        embedding_service: EmbeddingService,
        knowledge_graph: KnowledgeGraph,
    ) -> None:
        """初始化"""
        self._embedding_service = embedding_service
        self._knowledge_graph = knowledge_graph
        logger.info("RAG服务初始化完成")
    
    async def index_guide(self, guide: OperationGuide) -> None:
        """索引操作指南"""
        if not self._embedding_service or not self._knowledge_graph:
            return
        
        # 生成嵌入
        text_to_embed = f"{guide.title} {guide.app_name} {guide.feature_name} {' '.join(guide.steps)}"
        embedding = await self._embedding_service.embed_text(text_to_embed)
        
        guide.embedding = embedding
        self._guide_embeddings[guide.id] = embedding
        
        # 添加到知识图谱
        self._knowledge_graph.add_guide(guide)
        
        logger.debug(f"已索引指南: {guide.title}")
    
    async def index_node(self, node: KnowledgeNode) -> None:
        """索引知识节点"""
        if not self._embedding_service or not self._knowledge_graph:
            return
        
        # 生成嵌入
        text_to_embed = f"{node.name} {node.description} {' '.join(node.aliases)}"
        embedding = await self._embedding_service.embed_text(text_to_embed)
        
        node.embedding = embedding
        self._node_embeddings[node.id] = embedding
        
        # 添加到知识图谱
        self._knowledge_graph.add_node(node)
        
        logger.debug(f"已索引节点: {node.name}")
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> RAGResult:
        """检索相关知识"""
        if not self._embedding_service or not self._knowledge_graph:
            return RAGResult()
        
        # 获取查询嵌入
        query_embedding = await self._embedding_service.embed_text(query)
        
        # 检索指南
        guides = await self._retrieve_guides(query_embedding, top_k, min_score)
        
        # 检索节点
        nodes = await self._retrieve_nodes(query_embedding, top_k, min_score)
        
        # 生成上下文
        context = self._build_context(guides, nodes)
        
        # 计算整体置信度
        confidence = self._calculate_confidence(guides, nodes)
        
        return RAGResult(
            guides=guides,
            nodes=nodes,
            context=context,
            confidence=confidence,
        )
    
    async def _retrieve_guides(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[OperationGuide]:
        """检索操作指南"""
        if not self._embedding_service or not self._knowledge_graph:
            return []
        
        results: list[tuple[float, OperationGuide]] = []
        
        for guide_id, guide_embedding in self._guide_embeddings.items():
            score = self._embedding_service.cosine_similarity(
                query_embedding,
                guide_embedding,
            )
            
            if score >= min_score:
                guide = self._knowledge_graph._guides.get(guide_id)
                if guide:
                    results.append((score, guide))
        
        # 按分数排序
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [guide for _, guide in results[:top_k]]
    
    async def _retrieve_nodes(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[KnowledgeNode]:
        """检索知识节点"""
        if not self._embedding_service or not self._knowledge_graph:
            return []
        
        results: list[tuple[float, KnowledgeNode]] = []
        
        for node_id, node_embedding in self._node_embeddings.items():
            score = self._embedding_service.cosine_similarity(
                query_embedding,
                node_embedding,
            )
            
            if score >= min_score:
                node = self._knowledge_graph._nodes.get(node_id)
                if node:
                    results.append((score, node))
        
        # 按分数排序
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [node for _, node in results[:top_k]]
    
    def _build_context(
        self,
        guides: list[OperationGuide],
        nodes: list[KnowledgeNode],
    ) -> str:
        """构建上下文"""
        parts = []
        
        # 添加指南信息
        if guides:
            parts.append("【相关操作指南】")
            for guide in guides:
                parts.append(f"\n{guide.title}:")
                for i, step in enumerate(guide.friendly_steps or guide.steps, 1):
                    parts.append(f"  {i}. {step}")
        
        # 添加节点信息
        if nodes:
            parts.append("\n【相关知识】")
            for node in nodes:
                parts.append(f"- {node.name}: {node.description}")
        
        return "\n".join(parts)
    
    def _calculate_confidence(
        self,
        guides: list[OperationGuide],
        nodes: list[KnowledgeNode],
    ) -> float:
        """计算置信度"""
        if not guides and not nodes:
            return 0.0
        
        # 基于检索结果数量和质量计算
        guide_score = min(len(guides) / 3, 1.0) * 0.6
        node_score = min(len(nodes) / 5, 1.0) * 0.4
        
        # 考虑指南质量
        if guides:
            avg_quality = sum(g.quality_score for g in guides) / len(guides)
            guide_score *= avg_quality
        
        return guide_score + node_score
    
    async def expand_query(self, query: str) -> list[str]:
        """扩展查询（同义词、相关词）"""
        # 老年人语言映射
        elderly_mappings = {
            "手机吃钱": ["流量超标", "扣费", "话费"],
            "屏幕上有脏东西": ["广告", "弹窗", "悬浮窗"],
            "那个绿色的": ["微信", "WeChat"],
            "那个蓝色的": ["支付宝", "QQ"],
            "打字的地方": ["输入框", "搜索框"],
            "小红点": ["通知", "消息提醒"],
            "联系": ["打电话", "发消息", "视频通话"],
            "看看": ["查看", "打开", "浏览"],
        }
        
        expanded = [query]
        
        for key, values in elderly_mappings.items():
            if key in query:
                for value in values:
                    expanded.append(query.replace(key, value))
        
        return expanded
    
    async def retrieve_with_expansion(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> RAGResult:
        """带查询扩展的检索"""
        # 扩展查询
        expanded_queries = await self.expand_query(query)
        
        # 对每个扩展查询进行检索
        all_guides: dict[UUID, tuple[float, OperationGuide]] = {}
        all_nodes: dict[UUID, tuple[float, KnowledgeNode]] = {}
        
        for exp_query in expanded_queries:
            result = await self.retrieve(exp_query, top_k, min_score)
            
            for guide in result.guides:
                if guide.id not in all_guides:
                    all_guides[guide.id] = (result.confidence, guide)
                else:
                    # 取最高分
                    if result.confidence > all_guides[guide.id][0]:
                        all_guides[guide.id] = (result.confidence, guide)
            
            for node in result.nodes:
                if node.id not in all_nodes:
                    all_nodes[node.id] = (result.confidence, node)
                else:
                    if result.confidence > all_nodes[node.id][0]:
                        all_nodes[node.id] = (result.confidence, node)
        
        # 排序并取top_k
        guides = sorted(all_guides.values(), key=lambda x: x[0], reverse=True)[:top_k]
        nodes = sorted(all_nodes.values(), key=lambda x: x[0], reverse=True)[:top_k]
        
        final_guides = [g for _, g in guides]
        final_nodes = [n for _, n in nodes]
        
        return RAGResult(
            guides=final_guides,
            nodes=final_nodes,
            context=self._build_context(final_guides, final_nodes),
            confidence=self._calculate_confidence(final_guides, final_nodes),
        )
