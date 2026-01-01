import asyncio
import sys
from loguru import logger

# 确保能导入 src 目录
sys.path.append(".")

from src.knowledge.video_extractor import VideoKnowledgeExtractor, VideoInfo
from src.config import config

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")

async def test_search_and_extract_fallback():
    # --- 配置区域 ---
    # 这里我们直接提供一个真实的 B站 链接，跳过 DuckDuckGo 搜索
    # 这是一个关于 "如何重命名文件" 的简短教程
    FIXED_URL = "https://www.bilibili.com/video/BV1KP4y1Y7U1/?spm_id_from=333.337.search-card.all.click&vd_source=60e3da169867ed0e99de040aaa2130f4" 
    
    print(f"🚀 启动测试 (跳过不稳定搜索，直接测试模型链路)...")
    print(f"🔗 目标视频: {FIXED_URL}")
    print(f"🧠 使用模型: {config.api.vl_model}") # 打印一下确认读到了 Qwen3-VL

    # 1. 初始化
    extractor = VideoKnowledgeExtractor()
    await extractor.initialize()

    try:
        # --- 第一阶段：获取元数据 (使用 yt-dlp) ---
        print("\n⏳ [Phase 1] 正在解析视频元数据 (yt-dlp)...")
        # 既然 search_videos 跑不通，我们直接调用内部方法 _fetch_metadata
        # 注意：这是私有方法，但在测试脚本里调用是为了调试方便
        video_info = await extractor._fetch_metadata(FIXED_URL)

        if not video_info:
            print("❌ yt-dlp 解析失败。请检查：")
            print("1. 服务器能否访问 www.bilibili.com？")
            print("2. 你的 yt-dlp 版本是否太旧？(pip install -U yt-dlp)")
            return

        print(f"✅ 元数据获取成功！")
        print(f"   标题: {video_info.title}")
        print(f"   时长: {video_info.duration_seconds}秒")
        print(f"   简介预览: {video_info.transcript[:50].replace('\n', ' ')}...")

        # --- 第二阶段：模型提取 (Qwen3-VL) ---
        print(f"\n🤖 [Phase 2] 正在发送给 Sophnet Qwen3-VL 模型 (请耐心等待)...")
        
        guide = await extractor.extract_from_video(video_info)

        if guide:
            print("\n🎉🎉🎉 测试通过！模型成功返回了结构化数据：")
            print("="*60)
            print(f"📝 指南标题: {guide.title}")
            print(f"📱 涉及应用: {guide.app_name}")
            print(f"🔧 功能名称: {guide.feature_name}")
            print(f"👴 适老化步骤:")
            for i, step in enumerate(guide.friendly_steps):
                print(f"   {i+1}. {step}")
            print("-" * 30)
            print(f"❓ 自动生成 FAQ:")
            for q, a in guide.faq.items():
                print(f"   Q: {q}\n   A: {a}")
            print("="*60)
        else:
            print("⚠️ 提取失败 (Guide is None)。请检查 Config 中的模型名称是否正确。")

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await extractor.close()

if __name__ == "__main__":
    asyncio.run(test_search_and_extract_fallback())