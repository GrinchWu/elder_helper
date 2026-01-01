import asyncio
import sys
from loguru import logger

# 确保能导入 src 目录
sys.path.append(".")

from src.knowledge.video_extractor import VideoKnowledgeExtractor

# 配置日志输出，过滤掉太多杂讯，只保留 INFO
logger.remove()
logger.add(sys.stderr, level="INFO")

async def test_full_flow():
    # --- 测试参数 ---
    # 你可以随意修改这个问题，比如 "怎么用电脑发微信", "如何清理手机垃圾"
    QUERY = "微信怎么放大字体" 
    
    print(f"🚀 启动全链路测试 (Search -> Extract -> Rewrite)")
    print(f"❓ 模拟用户提问: [{QUERY}]")
    print("-" * 50)

    # 1. 初始化提取器
    extractor = VideoKnowledgeExtractor()
    await extractor.initialize()

    try:
        # --- 步骤 1: 搜索视频 ---
        print(f"\n🔍 [Step 1] 正在调用 Bilibili API 搜索视频...")
        
        # 搜索前 3 个结果
        videos = await extractor.search_videos(QUERY, platform="bilibili", max_results=3)

        if not videos:
            print("❌ 搜索失败：未找到任何相关视频。")
            return

        # 默认自动选择第一个视频作为“最佳匹配”
        target_video = videos[0]
        
        print(f"✅ 搜索成功！选中排位第一的视频：")
        print(f"   📺 标题: {target_video.title}")
        print(f"   🔗 链接: {target_video.url}")
        print(f"   ⏱️ 时长: {target_video.duration_seconds} 秒")
        # 打印一点简介证明拿到数据了
        preview = target_video.description[:30].replace('\n', ' ') + "..." if target_video.description else "无简介"
        print(f"   📝 简介: {preview}")

        # --- 步骤 2: 提取知识 ---
        print(f"\n🧠 [Step 2] 正在请求 Qwen3-VL 提取知识并进行适老化重写...")
        print("   (此过程涉及视频理解和多步推理，通常需要 15-30 秒，请耐心等待...)")

        guide = await extractor.extract_from_video(target_video)

        # --- 步骤 3: 展示结果 ---
        if guide:
            print("\n🎉🎉🎉 全流程测试通过！以下是生成的回答：")
            print("=" * 60)
            print(f"📘 指南标题: {guide.title}")
            print(f"📱 识别应用: {guide.app_name}")
            print(f"🔧 功能点: {guide.feature_name}")
            print(f"📊 质量评分: {guide.quality_score:.2f}")
            print("-" * 30)
            print(f"👴 给老年人的操作步骤:")
            for i, step in enumerate(guide.friendly_steps):
                print(f"   {i+1}. {step}")
            print("-" * 30)
            print(f"❓ 猜您可能遇到的问题 (FAQ):")
            for q, a in guide.faq.items():
                print(f"   Q: {q}")
                print(f"   A: {a}")
            print("=" * 60)
        else:
            print("⚠️ 提取失败 (返回为空)。")
            print("可能原因：视频内容无效、字幕缺失或模型调用超时。")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await extractor.close()

if __name__ == "__main__":
    asyncio.run(test_full_flow())