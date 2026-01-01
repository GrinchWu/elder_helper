"""视频知识提取器 - 从短视频平台提取操作知识"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

import httpx
from loguru import logger
from duckduckgo_search import DDGS  # 新增：用于搜索
import yt_dlp  # 新增：用于提取元数据

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
    thumbnail_url: str = "" # 新增：缩略图
    view_count: int = 0     # 新增：播放量（用于筛选热门视频）


class VideoKnowledgeExtractor:
    """视频知识提取器"""
    def __init__(self) -> None:
        # 1. 修正 URL 变量名 (对应你截图里的 sophnet_base_url)
        self._llm_url = config.api.sophnet_base_url 
        
        # 2. 关键：读取你在配置里写好的 Qwen3 模型名
        # 如果 config 里叫 vl_model，这里就用 vl_model
        self._model_name = config.api.vl_model 
        
        self._client: Optional[httpx.AsyncClient] = None
        self._ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,  # 关键：不下载视频文件
            'extract_flat': False,  # 需要深度解析以获取字幕
            'writesubtitles': True, # 尝试获取字幕
            'writeautomaticsub': True, # 尝试获取自动生成字幕
            'subtitleslangs': ['zh-Hans', 'zh-Hant', 'zh', 'en'], # 优先中文
        }
        # 线程池，用于运行同步的 yt-dlp
        self._executor = ThreadPoolExecutor(max_workers=3)
    
    async def initialize(self) -> None:
        """初始化"""
        self._client = httpx.AsyncClient(timeout=120.0)
        logger.info("VideoKnowledgeExtractor初始化完成")
    
    async def close(self) -> None:
        """关闭"""
        if self._client:
            await self._client.aclose()
        self._executor.shutdown(wait=False)

    async def search_videos(
        self,
        query: str,
        platform: str = "bilibili", # 推荐优先用bilibili，字幕和元数据更规范
        max_results: int = 5,
    ) -> list[VideoInfo]:
        """
        真实的视频搜索实现
        
        :param query: 用户的问题，如 "微信怎么放大字体"
        :param platform: 搜索目标 (bilibili, douyin, all)
        :param max_results: 最大返回数量
        """
        logger.info(f"正在搜索视频: {query} [平台: {platform}]")
        
        # 1. 构建搜索关键词 (利用site语法精准搜索)
        search_query = query
        if platform == "bilibili":
            search_query = f"site:bilibili.com {query} 教程"
        elif platform == "douyin":
            search_query = f"site:douyin.com {query} 教程"
        elif platform == "youtube":
            search_query = f"site:youtube.com {query} tutorial"
        else:
            search_query = f"{query} 视频教程"

        # 2. 使用 DuckDuckGo 搜索视频链接
        # DDGS是同步库，建议放入线程池，但这里为了演示简洁直接调用（它速度很快）
        # 实际生产建议 wrap_async
        video_urls = []
        try:
            with DDGS() as ddgs:
                # 搜索结果生成器
                results = ddgs.text(search_query, region='cn-zh', max_results=max_results * 2)
                for r in results:
                    url = r.get('href', '')
                    # 简单过滤，确保是视频链接
                    if any(x in url for x in ['bilibili.com/video', 'douyin.com/video', 'youtube.com/watch']):
                        video_urls.append(url)
                        if len(video_urls) >= max_results:
                            break
        except Exception as e:
            logger.error(f"搜索引擎请求失败: {e}")
            return []

        if not video_urls:
            logger.warning(f"未找到关于 '{query}' 的视频链接")
            return []

        logger.debug(f"找到原始链接: {video_urls}")

        # 3. 并发提取视频元数据
        tasks = [self._fetch_metadata(url) for url in video_urls]
        results = await asyncio.gather(*tasks)
        
        # 4. 过滤无效结果 (None)
        valid_videos = [v for v in results if v is not None]
        
        # 按播放量排序（如果获取到的话），优先处理热门教程
        valid_videos.sort(key=lambda x: x.view_count, reverse=True)
        
        logger.info(f"成功提取 {len(valid_videos)} 个有效视频信息")
        return valid_videos

    async def _fetch_metadata(self, url: str) -> Optional[VideoInfo]:
        """使用 yt-dlp 提取单个视频的详细元数据"""
        
        def _run_ydl():
            try:
                with yt_dlp.YoutubeDL(self._ydl_opts) as ydl:
                    # extract_info 会联网获取数据
                    info = ydl.extract_info(url, download=False)
                    return info
            except Exception as e:
                logger.warning(f"yt-dlp 解析失败 [{url}]: {str(e)[:100]}")
                return None

        # 在线程池中运行同步的 yt-dlp
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(self._executor, _run_ydl)

        if not info:
            return None

        # 数据清洗与转换
        try:
            # 尝试获取字幕/说明
            transcript = info.get('description', '')
            
            # 这里的逻辑是：如果没有字幕，yt-dlp 可能会把自动字幕放在 'automatic_captions'
            # 或者我们需要依赖 description。
            # 真实场景中，提取字幕比较复杂，这里简化为优先取 description，其次是 tags
            if 'captions' in info and info['captions']:
                # 这里省略了复杂的字幕下载解析逻辑，暂用描述代替
                pass
            
            # B站特定优化：B站的 description 通常包含笔记
            # 抖音特定优化：抖音 description 通常很短
            
            video_info = VideoInfo(
                video_id=info.get('id', ''),
                title=info.get('title', '未命名视频'),
                description=info.get('description', ''),
                url=info.get('webpage_url', url),
                platform=info.get('extractor', 'unknown'),
                duration_seconds=int(info.get('duration', 0)),
                thumbnail_url=info.get('thumbnail', ''),
                view_count=info.get('view_count', 0),
                # 将元数据拼接到 transcript 字段，给 LLM 更多上下文
                transcript=f"视频标题: {info.get('title')}\n视频简介: {info.get('description')}\n标签: {info.get('tags')}"
            )
            return video_info
        except Exception as e:
            logger.error(f"数据转换失败: {e}")
            return None
    
    
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
                    "model": self._model_name,
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
                    "model": self._model_name,
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
                    "model": self._model_name,
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
                    "model": self._model_name,
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
    
