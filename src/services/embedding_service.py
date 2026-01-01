"""向量嵌入服务 - 适配 SophNet 自定义接口"""

from __future__ import annotations

from typing import Optional
import os
import httpx
import numpy as np
from loguru import logger

from ..config import config

class EmbeddingService:
    """向量嵌入服务"""
    
    def __init__(self) -> None:
        # 1. 设置 Base URL (对应 curl 中的 URL 前缀)
        # 务必在 config.py 或 .env 中配置: 
        # BGE_M3_API_URL="https://www.sophnet.com/api/open-apis/projects/1i4tyIY4E0kPbugkacypKS/easyllms"
        # 注意：不要带 /embeddings 后缀，代码里会拼
        default_url = "https://www.sophnet.com/api/open-apis/projects/1i4tyIY4E0kPbugkacypKS/easyllms"
        self._api_url = os.getenv("BGE_M3_API_URL", default_url).rstrip("/")
        
        self._api_key = config.api.api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._embedding_dim = 1024
        
        # 2. 设置 EasyLLM ID (从你的 curl 示例中提取)
        # 如果这个 ID 会变，建议也放入 config
        self._easyllm_id = "5ViNoLlQ46rNHMftptg8Xn" 

    async def initialize(self) -> None:
        """初始化服务"""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        logger.info(f"Embedding服务初始化完成，API地址: {self._api_url}")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def embed_text(self, text: str) -> list[float]:
        """获取单条文本嵌入"""
        # 复用批量接口，只传一个元素的列表
        embeddings = await self.embed_texts([text])
        if embeddings:
            return embeddings[0]
        return [0.0] * self._embedding_dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本嵌入 (核心逻辑)"""
        if not self._client:
            raise RuntimeError("Embedding服务未初始化")
        
        if not texts:
            return []
        
        # 3. 构造 SophNet 要求的自定义 Payload
        payload = {
            "easyllm_id": self._easyllm_id,  # 必填 ID
            "input_texts": texts,            # 注意：参数名是 input_texts，不是 input
            "dimensions": self._embedding_dim
        }

        try:
            response = await self._client.post(
                f"{self._api_url}/embeddings",
                json=payload,
            )
            
            # 4. 错误处理与调试
            if response.status_code != 200:
                logger.error(f"API请求失败 [{response.status_code}]: {response.text}")
                return [[0.0] * self._embedding_dim for _ in texts]

            result = response.json()
            
            # 5. 响应解析 (关键点)
            # SophNet 的成功响应通常在 'result' 字段里
            # 假设结构是: {"status": 0, "message": "success", "result": [[0.1, ...], [0.2, ...]]}
            if "result" in result and result["result"]:
                return result["result"]
            
            # 备用：如果返回的是 OpenAI 格式 (data -> embedding)
            if "data" in result:
                return [item["embedding"] for item in result["data"]]

            logger.error(f"无法解析响应格式: {result}")
            return [[0.0] * self._embedding_dim for _ in texts]

        except httpx.HTTPError as e:
            logger.error(f"网络请求异常: {e}")
            return [[0.0] * self._embedding_dim for _ in texts]
            
    # ... cosine_similarity 等其他辅助方法保持不变 ...
    def cosine_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))