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
        platform: str = "bilibili",
        max_results: int = 5,
    ) -> list[VideoInfo]:
        """
        [Plan C] 直接调用 Bilibili 官方搜索 API (最稳定，不依赖 yt-dlp 搜索指令)
        """
        logger.info(f"正在搜索视频: {query} [平台: {platform}]")
        
        if platform != "bilibili":
            logger.warning("目前仅 Bilibili 支持 API 直连搜索，其他平台返回空。")
            return []

        # Bilibili Web 端搜索 API
        api_url = "https://api.bilibili.com/x/web-interface/search/type"
        params = {
            "search_type": "video",
            "keyword": query,
            "page": 1,
            "page_size": max_results
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": "https://www.bilibili.com/"
        }

        valid_videos = []
        
        try:
            # 使用现有的 httpx client 发起请求
            # 注意：B站 API 有时需要 Cookie，如果报错可能需要更复杂的封装
            # 但基础搜索通常是公开的
            resp = await self._client.get(api_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if data['code'] != 0:
                logger.error(f"B站 API 返回错误: {data.get('message')}")
                return []

            # 解析结果列表
            result_list = data.get('data', {}).get('result', [])
            if not result_list:
                logger.warning("B站 API 返回结果为空")
                return []

            for item in result_list:
                # B站 API 返回的数据中，标题通常带有 <em class="keyword"> 高亮标签，需要去除
                raw_title = item.get('title', '未命名')
                clean_title = raw_title.replace('<em class="keyword">', '').replace('</em>', '')
                
                # 转换时长 "02:30" -> 秒数
                duration_str = item.get('duration', '0:0')
                parts = duration_str.split(':')
                duration_sec = 0
                if len(parts) == 2:
                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    duration_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

                video = VideoInfo(
                    video_id=item.get('bvid', ''),
                    title=clean_title,
                    description=item.get('description', ''),
                    url=f"https://www.bilibili.com/video/{item.get('bvid')}",
                    platform="bilibili",
                    duration_seconds=duration_sec,
                    thumbnail_url=f"https:{item.get('pic')}" if item.get('pic', '').startswith('//') else item.get('pic', ''),
                    view_count=item.get('play', 0),
                    # 构造给 LLM 看的文本
                    transcript=f"视频标题: {clean_title}\n视频标签: {item.get('tag')}\n视频简介: {item.get('description')}"
                )
                valid_videos.append(video)

        except Exception as e:
            logger.error(f"Bilibili API 请求失败: {e}")
            return []

        logger.info(f"API 搜索成功，获取 {len(valid_videos)} 个结果")
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

    # ============ 知识库构建功能 ============
    
    @staticmethod
    def get_common_operation_queries() -> list[dict]:
        """获取10种常见计算机基本操作的搜索关键词（面向初级用户）"""
        return [
            {"query": "电脑新手如何打开浏览器上网", "app": "浏览器", "feature": "打开浏览器"},
            {"query": "老年人微信发消息教程", "app": "微信", "feature": "发送消息"},
            {"query": "电脑如何复制粘贴文字", "app": "系统", "feature": "复制粘贴"},
            {"query": "新手如何用电脑看新闻", "app": "浏览器", "feature": "浏览网页"},
            {"query": "微信视频通话怎么打", "app": "微信", "feature": "视频通话"},
            {"query": "电脑如何调节音量大小", "app": "系统", "feature": "调节音量"},
            {"query": "如何用电脑打字输入法", "app": "输入法", "feature": "打字输入"},
            {"query": "电脑如何连接wifi无线网", "app": "系统", "feature": "连接WiFi"},
            {"query": "微信如何发送图片照片", "app": "微信", "feature": "发送图片"},
            {"query": "电脑如何关机重启", "app": "系统", "feature": "关机重启"},
        ]
    
    async def build_knowledge_base(
        self,
        rag_service,
        max_videos_per_query: int = 2,
        use_llm_extract: bool = False,
    ) -> dict:
        """
        从B站搜索常见操作视频，构建知识库
        
        Args:
            rag_service: RAGService实例，用于索引知识
            max_videos_per_query: 每个查询最多处理的视频数
            use_llm_extract: 是否使用LLM提取详细步骤（较慢但更准确）
        
        Returns:
            构建统计信息
        """
        if not self._client:
            await self.initialize()
        
        queries = self.get_common_operation_queries()
        stats = {
            "total_queries": len(queries),
            "videos_found": 0,
            "guides_created": 0,
            "nodes_created": 0,
            "errors": []
        }
        
        logger.info(f"开始构建知识库，共 {len(queries)} 个查询...")
        
        for i, q in enumerate(queries):
            query = q["query"]
            app_name = q["app"]
            feature_name = q["feature"]
            
            logger.info(f"[{i+1}/{len(queries)}] 搜索: {query}")
            
            try:
                # 搜索视频
                videos = await self.search_videos(query, max_results=max_videos_per_query)
                stats["videos_found"] += len(videos)
                
                if not videos:
                    logger.warning(f"  未找到视频: {query}")
                    continue
                
                for video in videos:
                    try:
                        if use_llm_extract:
                            # 使用LLM提取详细步骤（较慢）
                            guide = await self.extract_from_video(video)
                        else:
                            # 快速模式：直接从视频元数据构建指南
                            guide = self._quick_build_guide(video, app_name, feature_name)
                        
                        if guide:
                            await rag_service.index_guide(guide)
                            stats["guides_created"] += 1
                            logger.info(f"  ✅ 已索引: {guide.title[:30]}...")
                    
                    except Exception as e:
                        logger.warning(f"  处理视频失败: {e}")
                        stats["errors"].append(str(e))
                
                # 为每个应用/功能创建知识节点
                node = KnowledgeNode(
                    id=uuid4(),
                    node_type=NodeType.FEATURE,
                    name=f"{app_name}-{feature_name}",
                    description=f"如何在{app_name}中{feature_name}",
                    aliases=[query, feature_name, f"{app_name}{feature_name}"],
                )
                await rag_service.index_node(node)
                stats["nodes_created"] += 1
                
            except Exception as e:
                logger.error(f"查询失败 [{query}]: {e}")
                stats["errors"].append(f"{query}: {e}")
            
            # 避免请求过快
            await asyncio.sleep(0.5)
        
        logger.info(f"知识库构建完成: {stats['guides_created']} 条指南, {stats['nodes_created']} 个节点")
        return stats
    
    def _quick_build_guide(
        self,
        video: VideoInfo,
        app_name: str,
        feature_name: str,
    ) -> OperationGuide:
        """快速构建操作指南（不调用LLM，直接从视频元数据生成）"""
        # 从标题和描述中提取简单步骤
        steps = self._extract_simple_steps(video.title, video.description)
        
        return OperationGuide(
            id=uuid4(),
            title=video.title,
            app_name=app_name,
            feature_name=feature_name,
            steps=steps,
            friendly_steps=steps,  # 快速模式下相同
            faq={},
            source_video_id=video.video_id,
            source_url=video.url,
            quality_score=0.6,  # 快速模式默认分数
        )
    
    def _extract_simple_steps(self, title: str, description: str) -> list[str]:
        """从标题和描述中提取简单步骤"""
        steps = []
        
        # 尝试从描述中提取步骤
        if description:
            lines = description.split('\n')
            for line in lines:
                line = line.strip()
                # 匹配数字开头或特定格式的步骤
                if line and (
                    line[0].isdigit() or 
                    line.startswith('第') or 
                    line.startswith('步骤') or
                    line.startswith('-') or
                    line.startswith('•')
                ):
                    step = line.lstrip('0123456789.-•步骤第： ').strip()
                    if step and len(step) > 2:
                        steps.append(step)
        
        # 如果没有提取到步骤，从标题生成基本步骤
        if not steps:
            steps = [
                f"打开{title.split('如何')[-1] if '如何' in title else title}",
                "按照视频教程操作",
                "完成操作"
            ]
        
        return steps[:10]  # 最多10个步骤
    
    async def build_knowledge_base_with_fallback(
        self,
        rag_service,
    ) -> dict:
        """
        构建知识库（带预置数据回退）
        如果B站搜索失败，使用预置的示例数据
        """
        # 先尝试从B站搜索
        stats = await self.build_knowledge_base(rag_service, max_videos_per_query=2)
        
        # 如果没有成功创建任何指南，使用预置数据
        if stats["guides_created"] == 0:
            logger.warning("B站搜索失败，使用预置示例数据...")
            await self._load_preset_knowledge(rag_service)
            stats["guides_created"] = 10
            stats["nodes_created"] = 5
            stats["fallback_used"] = True
        
        return stats
    
    async def _load_preset_knowledge(self, rag_service) -> None:
        """加载预置的示例知识数据"""
        preset_guides = [
            OperationGuide(
                id=uuid4(),
                title="如何打开浏览器上网",
                app_name="浏览器",
                feature_name="打开浏览器",
                steps=["找到浏览器图标", "双击打开", "在地址栏输入网址", "按回车访问"],
                friendly_steps=["在桌面找到蓝色的e图标或圆形彩色图标", "用鼠标快速点两下打开", "在最上面的长条框里输入网址", "按键盘上的回车键"],
                faq={"找不到浏览器": "在桌面或开始菜单找蓝色e图标"},
                quality_score=0.9
            ),
            OperationGuide(
                id=uuid4(),
                title="微信如何发送消息",
                app_name="微信",
                feature_name="发送消息",
                steps=["打开微信", "选择联系人", "输入消息", "点击发送"],
                friendly_steps=["点开绿色的微信图标", "找到要聊天的人点进去", "在下面的框里打字", "点右边的发送按钮"],
                faq={"找不到联系人": "在最上面的搜索框输入名字查找"},
                quality_score=0.9
            ),
            OperationGuide(
                id=uuid4(),
                title="电脑如何复制粘贴",
                app_name="系统",
                feature_name="复制粘贴",
                steps=["选中要复制的内容", "按Ctrl+C复制", "到目标位置", "按Ctrl+V粘贴"],
                friendly_steps=["用鼠标拖动选中文字（会变蓝色）", "同时按住Ctrl键和C键", "点击要粘贴的位置", "同时按住Ctrl键和V键"],
                faq={"复制不了": "确保先选中内容再复制"},
                quality_score=0.9
            ),
            OperationGuide(
                id=uuid4(),
                title="如何用浏览器看新闻",
                app_name="浏览器",
                feature_name="浏览网页",
                steps=["打开浏览器", "输入新闻网址", "浏览新闻列表", "点击感兴趣的新闻"],
                friendly_steps=["双击浏览器图标打开", "在地址栏输入www.people.com.cn", "看到新闻列表", "点击想看的新闻标题"],
                faq={"网页打不开": "检查网络连接是否正常"},
                quality_score=0.9
            ),
            OperationGuide(
                id=uuid4(),
                title="微信视频通话教程",
                app_name="微信",
                feature_name="视频通话",
                steps=["打开微信", "进入聊天", "点击加号", "选择视频通话"],
                friendly_steps=["打开绿色微信", "点开要通话的人", "点右下角的加号", "点视频通话图标"],
                faq={"对方看不到我": "检查摄像头是否被遮挡"},
                quality_score=0.85
            ),
            OperationGuide(
                id=uuid4(),
                title="电脑调节音量",
                app_name="系统",
                feature_name="调节音量",
                steps=["找到音量图标", "点击音量图标", "拖动滑块调节"],
                friendly_steps=["在屏幕右下角找到喇叭图标", "点一下喇叭图标", "上下拖动调节声音大小"],
                faq={"没有声音": "检查是否静音了，喇叭上有x表示静音"},
                quality_score=0.9
            ),
            OperationGuide(
                id=uuid4(),
                title="电脑打字输入法",
                app_name="输入法",
                feature_name="打字输入",
                steps=["点击输入框", "切换输入法", "输入拼音", "选择汉字"],
                friendly_steps=["点击要打字的地方", "按Ctrl+空格切换中英文", "用键盘打拼音", "按数字键选择汉字"],
                faq={"打不出中文": "按Ctrl+空格切换到中文输入法"},
                quality_score=0.85
            ),
            OperationGuide(
                id=uuid4(),
                title="电脑连接WiFi",
                app_name="系统",
                feature_name="连接WiFi",
                steps=["点击网络图标", "选择WiFi", "输入密码", "点击连接"],
                friendly_steps=["点右下角的网络图标", "找到家里的WiFi名称点击", "输入WiFi密码", "点连接按钮"],
                faq={"连不上WiFi": "确认密码是否正确，WiFi是否开启"},
                quality_score=0.9
            ),
            OperationGuide(
                id=uuid4(),
                title="微信发送图片",
                app_name="微信",
                feature_name="发送图片",
                steps=["打开聊天", "点击加号", "选择相册", "选择图片发送"],
                friendly_steps=["进入聊天界面", "点右下角加号", "点相册图标", "选择图片后点发送"],
                faq={"找不到图片": "图片在相册里，点相册图标查看"},
                quality_score=0.9
            ),
            OperationGuide(
                id=uuid4(),
                title="电脑关机重启",
                app_name="系统",
                feature_name="关机重启",
                steps=["点击开始菜单", "点击电源按钮", "选择关机或重启"],
                friendly_steps=["点屏幕左下角的开始按钮", "点电源图标", "选择关机或重新启动"],
                faq={"关不了机": "长按电源键5秒强制关机"},
                quality_score=0.9
            ),
        ]
        
        preset_nodes = [
            KnowledgeNode(id=uuid4(), node_type=NodeType.APP, name="浏览器",
                         description="用于上网、看新闻、搜索信息", aliases=["上网", "网页", "IE", "Chrome", "Edge"]),
            KnowledgeNode(id=uuid4(), node_type=NodeType.APP, name="微信",
                         description="聊天、发消息、视频通话", aliases=["WeChat", "绿色的", "聊天"]),
            KnowledgeNode(id=uuid4(), node_type=NodeType.APP, name="系统",
                         description="电脑基本操作", aliases=["Windows", "电脑", "桌面"]),
            KnowledgeNode(id=uuid4(), node_type=NodeType.CONCEPT, name="新闻",
                         description="查看资讯新闻", aliases=["看新闻", "新闻网站", "人民网"]),
            KnowledgeNode(id=uuid4(), node_type=NodeType.CONCEPT, name="输入法",
                         description="打字输入中文", aliases=["打字", "拼音", "中文输入"]),
        ]
        
        for guide in preset_guides:
            await rag_service.index_guide(guide)
            logger.info(f"  ✅ 预置指南: {guide.title}")
        
        for node in preset_nodes:
            await rag_service.index_node(node)
            logger.info(f"  ✅ 预置节点: {node.name}")
    