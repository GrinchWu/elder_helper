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
    paths: list[list[KnowledgeNode]] = field(default_factory=list)
    context: str = ""
    confidence: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


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
        await self._preload_index()

    async def _preload_index(self) -> None:
        """预热索引，补齐现有节点/指南的嵌入缓存"""
        if not self._embedding_service or not self._knowledge_graph:
            return
        for guide in list(self._knowledge_graph._guides.values()):
            if guide.id not in self._guide_embeddings:
                text_to_embed = f"{guide.title} {guide.app_name} {guide.feature_name} {' '.join(guide.steps)}"
                embedding = await self._embedding_service.embed_text(text_to_embed)
                guide.embedding = embedding
                self._guide_embeddings[guide.id] = embedding
        for node in list(self._knowledge_graph._nodes.values()):
            if node.id not in self._node_embeddings:
                text_to_embed = f"{node.name} {node.description} {' '.join(node.aliases)}"
                embedding = await self._embedding_service.embed_text(text_to_embed)
                node.embedding = embedding
                self._node_embeddings[node.id] = embedding
    
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
        # 计算metrics
        guide_sims = [self._embedding_service.cosine_similarity(query_embedding, self._guide_embeddings.get(g.id)) for g in guides if self._guide_embeddings.get(g.id)]
        node_sims = [self._embedding_service.cosine_similarity(query_embedding, self._node_embeddings.get(n.id)) for n in nodes if self._node_embeddings.get(n.id)]
        similarity_vals = guide_sims + node_sims
        similarity_avg = (sum(similarity_vals) / len(similarity_vals)) if similarity_vals else 0.0
        pre_text_parts = []
        for g in guides:
            steps = g.friendly_steps or g.steps
            pre_text_parts.append(" ".join(steps))
        for n in nodes:
            pre_text_parts.append(n.description)
        pre_len = len(" ".join(pre_text_parts))
        compressed_len = len(context)
        compression_ratio = ((pre_len - compressed_len) / pre_len) if pre_len > 0 else 0.0
        metrics = {"similarity_avg": similarity_avg, "recall_count": len(guides) + len(nodes), "compression_ratio": compression_ratio}
        return RAGResult(guides=guides, nodes=nodes, context=context, confidence=confidence, metrics=metrics)
    
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
        def shorten_text(text: str) -> str:
            fillers = ["请", "然后", "接着", "就", "把", "一下", "这个", "那个", "呢", "啊", "呀", "啦"]
            for f in fillers:
                text = text.replace(f, "")
            text = text.replace("，", " ").replace(",", " ").replace("。", " ").replace(".", " ").strip()
            while "  " in text:
                text = text.replace("  " , " ")
            return text

        parts: list[str] = []

        # 操作路径（压缩格式）
        if guides:
            parts.append("【操作路径】")
            for guide in guides:
                title = f"{guide.title} ({guide.app_name}/{guide.feature_name})" if (guide.app_name or guide.feature_name) else guide.title
                parts.append(f"\n{title}:")
                steps = guide.friendly_steps or guide.steps
                path_elems = [f"{i+1}. {shorten_text(step)} - [截图]" for i, step in enumerate(steps)]
                parts.append("  " + " -> ".join(path_elems))
        
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

    async def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> RAGResult:
        if not self._embedding_service or not self._knowledge_graph:
            return RAGResult()
        query_embedding = await self._embedding_service.embed_text(query)
        guides = await self._retrieve_guides(query_embedding, max(top_k, 5), min_score)
        nodes = await self._retrieve_nodes(query_embedding, max(top_k * 2, 10), min_score * 0.8)
        paths, path_scores = self._infer_paths(query_embedding, guides, nodes, max_depth=6, max_paths=top_k)
        context = self._build_path_context(paths) if paths else self._build_context(guides, nodes)
        confidence = self._calculate_hybrid_confidence(guides, nodes, path_scores)
        metrics = {
            "recall_rate": min((len(guides) + len(nodes)) / (top_k * 2), 1.0),
            "path_completeness": (path_scores[0] if path_scores else 0.0)
        }
        return RAGResult(
            guides=guides,
            nodes=nodes,
            paths=paths,
            context=context,
            confidence=confidence,
            metrics=metrics,
        )

    def _infer_paths(
        self,
        query_embedding: list[float],
        guides: list[OperationGuide],
        nodes: list[KnowledgeNode],
        max_depth: int = 6,
        max_paths: int = 5,
    ) -> tuple[list[list[KnowledgeNode]], list[float]]:
        if not self._embedding_service or not self._knowledge_graph:
            return [], []
        graph = self._knowledge_graph._graph
        if graph is None or len(graph) == 0:
            return [], []
        edge_types_pref = {
            EdgeType.NEXT_STEP.value: 1.0,
            EdgeType.REQUIRES.value: 0.9,
            EdgeType.HAS_FEATURE.value: 0.8,
            EdgeType.CONTAINS.value: 0.6,
            EdgeType.ALTERNATIVE.value: 0.5,
            EdgeType.SIMILAR_TO.value: 0.5,
        }
        start_ids = [str(n.id) for n in nodes]
        paths: list[list[KnowledgeNode]] = []
        scores: list[float] = []
        seen: set[tuple[str, ...]] = set()
        for sid in start_ids:
            stack: list[tuple[str, list[str]]] = [(sid, [sid])]
            while stack:
                current, path_ids = stack.pop()
                if len(path_ids) > max_depth:
                    continue
                score = self._score_path(query_embedding, path_ids, edge_types_pref)
                key = tuple(path_ids)
                if key in seen:
                    continue
                seen.add(key)
                if len(path_ids) > 1 and score >= 0.3:
                    node_list: list[KnowledgeNode] = []
                    for nid in path_ids:
                        try:
                            uid = UUID(nid)
                        except ValueError:
                            continue
                        node_obj = self._knowledge_graph._nodes.get(uid)
                        if node_obj:
                            node_list.append(node_obj)
                    if node_list:
                        paths.append(node_list)
                        scores.append(score)
                        if len(paths) >= max_paths:
                            break
                for neighbor in graph.neighbors(current):
                    edge_data = graph.edges[current, neighbor]
                    edge_type = edge_data.get("edge_type", "")
                    weight = edge_types_pref.get(edge_type, 0.1)
                    if weight >= 0.5:
                        stack.append((neighbor, path_ids + [neighbor]))
            if len(paths) >= max_paths:
                break
        paired = sorted(zip(scores, paths), key=lambda x: x[0], reverse=True)
        if not paired:
            return [], []
        scores_sorted = [p[0] for p in paired]
        paths_sorted = [p[1] for p in paired]
        return paths_sorted, scores_sorted

    def _score_path(
        self,
        query_embedding: list[float],
        path_node_ids: list[str],
        edge_types_pref: dict[str, float],
    ) -> float:
        graph = self._knowledge_graph._graph
        sims: list[float] = []
        for nid in path_node_ids:
            try:
                uid = UUID(nid)
            except ValueError:
                continue
            node = self._knowledge_graph._nodes.get(uid)
            if node and node.embedding:
                sims.append(self._embedding_service.cosine_similarity(query_embedding, node.embedding))
        sim_avg = sum(sims) / len(sims) if sims else 0.0
        cohesion_scores: list[float] = []
        for i in range(len(path_node_ids) - 1):
            et = graph.edges[path_node_ids[i], path_node_ids[i+1]].get("edge_type", "")
            cohesion_scores.append(edge_types_pref.get(et, 0.1))
        cohesion = sum(cohesion_scores) / len(cohesion_scores) if cohesion_scores else 0.0
        length_score = min(len(path_node_ids) / 4, 1.0)
        return 0.5 * sim_avg + 0.3 * cohesion + 0.2 * length_score

    def _build_path_context(self, paths: list[list[KnowledgeNode]]) -> str:
        parts: list[str] = []
        parts.append("【操作路径】")
        for idx, path in enumerate(paths, 1):
            parts.append(f"\n路径{idx}:")
            for i, node in enumerate(path, 1):
                parts.append(f"  {i}. {node.name} - {node.description} [截图]")
        return "\n".join(parts)

    def _calculate_hybrid_confidence(
        self,
        guides: list[OperationGuide],
        nodes: list[KnowledgeNode],
        path_scores: list[float],
    ) -> float:
        base = self._calculate_confidence(guides, nodes)
        path_bonus = 0.0
        if path_scores:
            path_bonus = min(sum(path_scores) / len(path_scores), 1.0) * 0.5
        total = base + path_bonus
        return 1.0 if total > 1.0 else total
