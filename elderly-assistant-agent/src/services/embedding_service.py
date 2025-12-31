"""向量嵌入服务 - 使用BGE-M3"""

from __future__ import annotations

from typing import Optional

import httpx
import numpy as np
from loguru import logger

from ..config import config


class EmbeddingService:
    """向量嵌入服务"""
    
    def __init__(self) -> None:
        self._api_url = config.api.bge_m3_url
        self._client: Optional[httpx.AsyncClient] = None
        self._embedding_dim = 1024  # BGE-M3 维度
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"Embedding服务初始化完成，API地址: {self._api_url}")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def embed_text(self, text: str) -> list[float]:
        """获取文本嵌入"""
        if not self._client:
            raise RuntimeError("Embedding服务未初始化")
        
        if not text.strip():
            return [0.0] * self._embedding_dim
        
        try:
            response = await self._client.post(
                f"{self._api_url}/embeddings",
                json={
                    "input": text,
                    "model": "bge-m3",
                },
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result["data"][0]["embedding"]
            return embedding
            
        except httpx.HTTPError as e:
            logger.error(f"获取嵌入失败: {e}")
            return [0.0] * self._embedding_dim
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本嵌入"""
        if not self._client:
            raise RuntimeError("Embedding服务未初始化")
        
        if not texts:
            return []
        
        try:
            response = await self._client.post(
                f"{self._api_url}/embeddings",
                json={
                    "input": texts,
                    "model": "bge-m3",
                },
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
        except httpx.HTTPError as e:
            logger.error(f"批量获取嵌入失败: {e}")
            return [[0.0] * self._embedding_dim for _ in texts]
    
    def cosine_similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """计算余弦相似度"""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    async def find_most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """找到最相似的候选项"""
        if not candidates:
            return []
        
        query_embedding = await self.embed_text(query)
        candidate_embeddings = await self.embed_texts(candidates)
        
        similarities: list[tuple[int, float]] = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            sim = self.cosine_similarity(query_embedding, candidate_emb)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def semantic_search(
        self,
        query: str,
        documents: list[dict[str, str]],
        text_field: str = "text",
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[tuple[dict[str, str], float]]:
        """语义搜索"""
        if not documents:
            return []
        
        texts = [doc.get(text_field, "") for doc in documents]
        results = await self.find_most_similar(query, texts, top_k=len(documents))
        
        filtered_results: list[tuple[dict[str, str], float]] = []
        for idx, score in results:
            if score >= threshold:
                filtered_results.append((documents[idx], score))
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
