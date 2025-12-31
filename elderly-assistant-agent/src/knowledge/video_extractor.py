"""视频知识提取器 - 从短视频平台提取操作知识"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import httpx
from loguru import logger

from ..config import config
from ..models.knowledge import OperationGuide, KnowledgeNode, NodeType


@dataclass
class VideoInfo:
    """视频信息"""
    video_id: str = ""
    title: str = ""
    description: str = ""
    url: str = ""
    platform: str = ""  # douyin, kuaishou, bilibili
    duration_seconds: int = 0
    transcript: str = ""  # 视频字幕/转录
    frames: list[bytes] = field(default_factory=list)  # 关键帧


class VideoKnowledgeExtractor:
    """视频知识提取器"""
    
    def __init__(self) -> None:
        self._llm_url = config.api.qwen_llm_url
        self._vl_url = config.api.qwen_vl_url
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """初始化"""
        self._client = httpx.AsyncClient(timeout=120.0)
        logger.info("VideoKnowledgeExtractor初始化完成")
    
    async def close(self) -> None:
        """关闭"""
        if self._client:
            await self._client.aclose()
    
    async def extract_from_video(self, video_info: VideoInfo) -> Optional[OperationGuide]:
        """从视频提取操作指南"""
        if not self._client:
            raise RuntimeError("提取器未初始化")
        
        # 1. 分析视频内容
        content_analysis = await self._analyze_video_content(video_info)
        
        if not content_analysis:
            return None
        
        # 2. 提取操作步骤
        steps = await self._extract_steps(content_analysis, video_info.transcript)
        
        if not steps:
            return None
        
        # 3. 生成老年人友好的描述
        friendly_steps = await self._generate_friendly_steps(steps)
        
        # 4. 提取FAQ
        faq = await self._extract_faq(content_analysis)
        
        # 5. 构建操作指南
        guide = OperationGuide(
            id=uuid4(),
            title=video_info.title,
            app_name=content_analysis.get("app_name", ""),
            feature_name=content_analysis.get("feature_name", ""),
            steps=steps,
            friendly_steps=friendly_steps,
            faq=faq,
            source_video_id=video_info.video_id,
            source_url=video_info.url,
            quality_score=self._calculate_quality_score(steps, friendly_steps),
        )
        
        return guide
    
    async def _analyze_video_content(self, video_info: VideoInfo) -> dict:
        """分析视频内容"""
        if not self._client:
            return {}
        
        prompt = f"""分析这个教程视频的内容：

标题：{video_info.title}
描述：{video_info.description}
字幕/转录：{video_info.transcript[:2000] if video_info.transcript else "无"}

请提取以下信息（JSON格式）：
{{
    "app_name": "涉及的应用名称",
    "feature_name": "教授的功能",
    "target_audience": "目标受众",
    "difficulty_level": "难度等级(easy/medium/hard)",
    "main_topic": "主要主题",
    "prerequisites": ["前置条件列表"]
}}"""
        
        try:
            response = await self._client.post(
                f"{self._llm_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析JSON
            import json
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
                
        except Exception as e:
            logger.error(f"分析视频内容失败: {e}")
        
        return {}
    
    async def _extract_steps(
        self,
        content_analysis: dict,
        transcript: str,
    ) -> list[str]:
        """提取操作步骤"""
        if not self._client:
            return []
        
        prompt = f"""从以下视频转录中提取操作步骤：

应用：{content_analysis.get('app_name', '未知')}
功能：{content_analysis.get('feature_name', '未知')}

转录内容：
{transcript[:3000] if transcript else "无转录"}

请提取清晰的操作步骤列表，每个步骤应该是一个具体的操作。
只返回步骤列表，每行一个步骤，用数字编号。"""
        
        try:
            response = await self._client.post(
                f"{self._llm_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析步骤
            steps = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # 清理行首的数字和符号
                    step = line.lstrip("0123456789.-) ").strip()
                    if step:
                        steps.append(step)
            
            return steps
            
        except Exception as e:
            logger.error(f"提取步骤失败: {e}")
        
        return []
    
    async def _generate_friendly_steps(self, steps: list[str]) -> list[str]:
        """生成老年人友好的步骤描述"""
        if not self._client or not steps:
            return steps
        
        prompt = f"""将以下操作步骤转换为老年人能理解的简单描述：

原始步骤：
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(steps))}

要求：
1. 使用简单的日常用语
2. 避免技术术语
3. 描述要具体（如"点击左上角的绿色按钮"）
4. 每个步骤保持简短

请返回转换后的步骤列表，每行一个。"""
        
        try:
            response = await self._client.post(
                f"{self._llm_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析步骤
            friendly_steps = []
            for line in content.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    step = line.lstrip("0123456789.-) ").strip()
                    if step:
                        friendly_steps.append(step)
            
            # 确保数量匹配
            if len(friendly_steps) != len(steps):
                return steps
            
            return friendly_steps
            
        except Exception as e:
            logger.error(f"生成友好步骤失败: {e}")
        
        return steps
    
    async def _extract_faq(self, content_analysis: dict) -> dict[str, str]:
        """提取常见问题"""
        if not self._client:
            return {}
        
        app_name = content_analysis.get("app_name", "")
        feature_name = content_analysis.get("feature_name", "")
        
        prompt = f"""为"{app_name}"的"{feature_name}"功能生成3-5个老年人可能会问的常见问题及答案。

格式：
问：[问题]
答：[答案]

问题应该是老年人可能遇到的困惑，答案要简单易懂。"""
        
        try:
            response = await self._client.post(
                f"{self._llm_url}/chat/completions",
                json={
                    "model": "qwen",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                },
                headers={"Authorization": f"Bearer {config.api.api_key}"} if config.api.api_key else {},
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析FAQ
            faq = {}
            lines = content.strip().split("\n")
            current_q = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("问：") or line.startswith("Q:"):
                    current_q = line[2:].strip()
                elif (line.startswith("答：") or line.startswith("A:")) and current_q:
                    faq[current_q] = line[2:].strip()
                    current_q = ""
            
            return faq
            
        except Exception as e:
            logger.error(f"提取FAQ失败: {e}")
        
        return {}
    
    def _calculate_quality_score(
        self,
        steps: list[str],
        friendly_steps: list[str],
    ) -> float:
        """计算质量分数"""
        score = 0.0
        
        # 步骤数量合理性
        if 3 <= len(steps) <= 15:
            score += 0.3
        elif len(steps) > 0:
            score += 0.1
        
        # 步骤描述长度
        avg_len = sum(len(s) for s in steps) / len(steps) if steps else 0
        if 10 <= avg_len <= 50:
            score += 0.3
        elif avg_len > 0:
            score += 0.1
        
        # 友好步骤质量
        if friendly_steps and len(friendly_steps) == len(steps):
            score += 0.4
        elif friendly_steps:
            score += 0.2
        
        return min(score, 1.0)
    
    async def search_videos(
        self,
        query: str,
        platform: str = "all",
        max_results: int = 10,
    ) -> list[VideoInfo]:
        """搜索相关视频（需要实现具体的平台API调用）"""
        # 这里是占位实现，实际需要调用各平台的API
        logger.info(f"搜索视频: {query}, 平台: {platform}")
        
        # 返回空列表，实际实现需要调用平台API
        return []
