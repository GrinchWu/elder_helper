"""VideoKnowledgeExtractor 单元测试"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

from src.knowledge.video_extractor import VideoKnowledgeExtractor, VideoInfo
from src.models.knowledge import OperationGuide


# ============ Fixtures ============

@pytest.fixture
def extractor():
    """创建提取器实例"""
    return VideoKnowledgeExtractor()


@pytest.fixture
def sample_video_info():
    """示例视频信息"""
    return VideoInfo(
        video_id="BV1234567890",
        title="微信如何发送图片教程",
        description="本视频教你如何在微信中发送图片给好友",
        url="https://www.bilibili.com/video/BV1234567890",
        platform="bilibili",
        duration_seconds=180,
        transcript="视频标题: 微信如何发送图片教程\n视频简介: 本视频教你如何在微信中发送图片给好友",
        thumbnail_url="https://example.com/thumb.jpg",
        view_count=10000
    )


@pytest.fixture
def mock_bilibili_response():
    """模拟B站API响应"""
    return {
        "code": 0,
        "data": {
            "result": [
                {
                    "bvid": "BV1test123",
                    "title": '<em class="keyword">微信</em>发图片教程',
                    "description": "教你发图片",
                    "duration": "3:20",
                    "pic": "//example.com/pic.jpg",
                    "play": 5000,
                    "tag": "教程,微信"
                },
                {
                    "bvid": "BV2test456",
                    "title": "手机<em class=\"keyword\">微信</em>使用",
                    "description": "微信基础教程",
                    "duration": "10:05",
                    "pic": "//example.com/pic2.jpg",
                    "play": 8000,
                    "tag": "微信,手机"
                }
            ]
        }
    }


# ============ search_videos 测试 ============

class TestSearchVideos:
    """search_videos 方法测试"""

    @pytest.mark.asyncio
    async def test_search_bilibili_success(self, extractor, mock_bilibili_response):
        """测试B站搜索成功"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_bilibili_response
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        results = await extractor.search_videos("微信发图片", platform="bilibili", max_results=5)
        
        assert len(results) == 2
        assert results[0].video_id == "BV1test123"
        assert results[0].title == "微信发图片教程"  # em标签已清除
        assert results[0].platform == "bilibili"
        assert results[0].duration_seconds == 200  # 3:20 = 200秒
        assert results[1].duration_seconds == 605  # 10:05 = 605秒

    @pytest.mark.asyncio
    async def test_search_non_bilibili_returns_empty(self, extractor):
        """测试非B站平台返回空列表"""
        extractor._client = AsyncMock()
        
        results = await extractor.search_videos("测试", platform="douyin")
        
        assert results == []

    @pytest.mark.asyncio
    async def test_search_api_error(self, extractor):
        """测试API返回错误码"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": -1, "message": "请求错误"}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        results = await extractor.search_videos("测试")
        
        assert results == []

    @pytest.mark.asyncio
    async def test_search_empty_results(self, extractor):
        """测试搜索结果为空"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"code": 0, "data": {"result": []}}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        results = await extractor.search_videos("不存在的内容xyz")
        
        assert results == []

    @pytest.mark.asyncio
    async def test_search_network_error(self, extractor):
        """测试网络错误"""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("网络连接失败"))
        
        extractor._client = mock_client
        
        results = await extractor.search_videos("测试")
        
        assert results == []


# ============ extract_from_video 测试 ============

class TestExtractFromVideo:
    """extract_from_video 方法测试"""

    @pytest.mark.asyncio
    async def test_extract_success(self, extractor, sample_video_info):
        """测试成功提取操作指南"""
        mock_client = AsyncMock()
        
        # 模拟LLM响应
        def create_mock_response(content):
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": content}}]
            }
            mock_resp.raise_for_status = MagicMock()
            return mock_resp
        
        # 按调用顺序返回不同响应
        mock_client.post = AsyncMock(side_effect=[
            # _analyze_video_content 响应
            create_mock_response('{"app_name": "微信", "feature_name": "发送图片", "difficulty_level": "easy"}'),
            # _extract_steps 响应
            create_mock_response("1. 打开微信\n2. 选择联系人\n3. 点击加号\n4. 选择图片"),
            # _generate_friendly_steps 响应
            create_mock_response("1. 找到手机上的微信图标，点一下打开\n2. 找到要发图片的人，点他的名字\n3. 点右下角的加号按钮\n4. 点相册，选择要发的图片"),
            # _extract_faq 响应
            create_mock_response("问：找不到加号怎么办？\n答：加号在聊天界面的右下角")
        ])
        
        extractor._client = mock_client
        
        guide = await extractor.extract_from_video(sample_video_info)
        
        assert guide is not None
        assert isinstance(guide, OperationGuide)
        assert guide.app_name == "微信"
        assert guide.feature_name == "发送图片"
        assert len(guide.steps) == 4
        assert len(guide.friendly_steps) == 4
        assert guide.source_video_id == "BV1234567890"

    @pytest.mark.asyncio
    async def test_extract_not_initialized(self, extractor, sample_video_info):
        """测试未初始化时抛出异常"""
        extractor._client = None
        
        with pytest.raises(RuntimeError, match="提取器未初始化"):
            await extractor.extract_from_video(sample_video_info)

    @pytest.mark.asyncio
    async def test_extract_analysis_fails(self, extractor, sample_video_info):
        """测试内容分析失败返回None"""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("LLM调用失败"))
        
        extractor._client = mock_client
        
        guide = await extractor.extract_from_video(sample_video_info)
        
        assert guide is None


# ============ _analyze_video_content 测试 ============

class TestAnalyzeVideoContent:
    """_analyze_video_content 方法测试"""

    @pytest.mark.asyncio
    async def test_analyze_success(self, extractor, sample_video_info):
        """测试成功分析视频内容"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"app_name": "微信", "feature_name": "发送图片", "difficulty_level": "easy"}'
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        result = await extractor._analyze_video_content(sample_video_info)
        
        assert result["app_name"] == "微信"
        assert result["feature_name"] == "发送图片"

    @pytest.mark.asyncio
    async def test_analyze_invalid_json(self, extractor, sample_video_info):
        """测试返回无效JSON"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "这不是JSON格式"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        result = await extractor._analyze_video_content(sample_video_info)
        
        assert result == {}


# ============ _extract_steps 测试 ============

class TestExtractSteps:
    """_extract_steps 方法测试"""

    @pytest.mark.asyncio
    async def test_extract_steps_success(self, extractor):
        """测试成功提取步骤"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "1. 打开应用\n2. 点击按钮\n3. 输入内容\n4. 确认提交"
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        content_analysis = {"app_name": "测试应用", "feature_name": "测试功能"}
        steps = await extractor._extract_steps(content_analysis, "测试转录内容")
        
        assert len(steps) == 4
        assert steps[0] == "打开应用"
        assert steps[3] == "确认提交"

    @pytest.mark.asyncio
    async def test_extract_steps_with_dash_format(self, extractor):
        """测试提取带破折号格式的步骤"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "- 第一步操作\n- 第二步操作\n- 第三步操作"
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        steps = await extractor._extract_steps({}, "")
        
        assert len(steps) == 3


# ============ _generate_friendly_steps 测试 ============

class TestGenerateFriendlySteps:
    """_generate_friendly_steps 方法测试"""

    @pytest.mark.asyncio
    async def test_generate_friendly_success(self, extractor):
        """测试成功生成友好步骤"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "1. 找到微信图标点一下\n2. 找到朋友的名字点进去"
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        original_steps = ["打开微信", "选择联系人"]
        friendly = await extractor._generate_friendly_steps(original_steps)
        
        assert len(friendly) == 2
        assert "微信" in friendly[0]

    @pytest.mark.asyncio
    async def test_generate_friendly_count_mismatch(self, extractor):
        """测试步骤数量不匹配时返回原始步骤"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "1. 只有一个步骤"
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        original_steps = ["步骤1", "步骤2", "步骤3"]
        friendly = await extractor._generate_friendly_steps(original_steps)
        
        # 数量不匹配，应返回原始步骤
        assert friendly == original_steps

    @pytest.mark.asyncio
    async def test_generate_friendly_empty_input(self, extractor):
        """测试空输入"""
        extractor._client = AsyncMock()
        
        result = await extractor._generate_friendly_steps([])
        
        assert result == []


# ============ _extract_faq 测试 ============

class TestExtractFaq:
    """_extract_faq 方法测试"""

    @pytest.mark.asyncio
    async def test_extract_faq_success(self, extractor):
        """测试成功提取FAQ"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "问：如何找到设置？\n答：点击右上角的三个点\n问：忘记密码怎么办？\n答：点击忘记密码链接"
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        extractor._client = mock_client
        
        faq = await extractor._extract_faq({"app_name": "微信", "feature_name": "设置"})
        
        assert len(faq) == 2
        assert "如何找到设置？" in faq
        assert faq["如何找到设置？"] == "点击右上角的三个点"


# ============ _calculate_quality_score 测试 ============

class TestCalculateQualityScore:
    """_calculate_quality_score 方法测试"""

    def test_perfect_score(self, extractor):
        """测试完美分数"""
        steps = ["步骤" + "x" * 20 for _ in range(5)]  # 5个步骤，每个约22字符
        friendly_steps = steps.copy()
        
        score = extractor._calculate_quality_score(steps, friendly_steps)
        
        assert score == 1.0

    def test_no_steps(self, extractor):
        """测试无步骤"""
        score = extractor._calculate_quality_score([], [])
        
        assert score == 0.0

    def test_too_many_steps(self, extractor):
        """测试步骤过多"""
        steps = ["步骤" for _ in range(20)]  # 20个步骤，超过15
        
        score = extractor._calculate_quality_score(steps, steps)
        
        assert score < 1.0

    def test_mismatched_friendly_steps(self, extractor):
        """测试友好步骤数量不匹配"""
        steps = ["步骤1", "步骤2", "步骤3"]
        friendly_steps = ["友好步骤1"]  # 数量不匹配
        
        score = extractor._calculate_quality_score(steps, friendly_steps)
        
        # 友好步骤不匹配，只能得到部分分数
        assert score < 1.0


# ============ 初始化和关闭测试 ============

class TestInitializeAndClose:
    """初始化和关闭方法测试"""

    @pytest.mark.asyncio
    async def test_initialize(self, extractor):
        """测试初始化"""
        assert extractor._client is None
        
        await extractor.initialize()
        
        assert extractor._client is not None

    @pytest.mark.asyncio
    async def test_close(self, extractor):
        """测试关闭"""
        await extractor.initialize()
        assert extractor._client is not None
        
        await extractor.close()
        
        # close后client应该被关闭（但对象可能还在）


# ============ VideoInfo 数据类测试 ============

class TestVideoInfo:
    """VideoInfo 数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        info = VideoInfo()
        
        assert info.video_id == ""
        assert info.title == ""
        assert info.duration_seconds == 0
        assert info.view_count == 0

    def test_custom_values(self):
        """测试自定义值"""
        info = VideoInfo(
            video_id="test123",
            title="测试视频",
            duration_seconds=300
        )
        
        assert info.video_id == "test123"
        assert info.title == "测试视频"
        assert info.duration_seconds == 300
