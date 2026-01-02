import asyncio
import os
import sys
from loguru import logger

# 确保能导入 src 模块
sys.path.append(os.getcwd())

from src.config import config
from src.services.vision_service import VisionService

# 配置日志输出到控制台
logger.remove()
logger.add(sys.stderr, level="INFO")

async def test_ocr_vl_parallel():
    """测试 OCR 和 VL 的并行调用逻辑"""
    print("\n" + "="*50)
    print("🚀 开始测试 VisionService (OCR + VL 并行模式)")
    print("="*50)

    # 1. 初始化服务
    vision = VisionService()
    try:
        await vision.initialize()
        print("\n✅ VisionService 初始化成功")
        print(f"   - API URL: {vision._config.base_url}")
        print(f"   - OCR Model: {vision._config.model_ocr}")
        print(f"   - VL Model: {vision._config.model_light}")

        # 2. 截取屏幕
        print("\n📸 正在截取当前屏幕...")
        screenshot, original_size = await vision.capture_screen()
        
        if not screenshot:
            logger.error("❌ 截屏失败，无法继续测试")
            return

        print(f"✅ 截屏成功，大小: {len(screenshot)} bytes, 原始尺寸: {original_size}")
        
        # 3. 并行执行 OCR 和 VL 分析
        print("\n⚡ 正在并行请求 DeepSeek-OCR 和 VL 分析...")
        print("   (请耐心等待，取决于网络速度...)")
        
        # 定义任务
        # task1: 调用 extract_text_from_bytes (OCR)
        ocr_task = vision.extract_text_from_bytes(screenshot)
        
        # task2: 调用 analyze_screen_state (VL)
        # 模拟一个用户意图，比如"查找微信"
        vl_task = vision.analyze_screen_state(screenshot, user_intent="查找屏幕上的文本信息")

        # 记录开始时间
        start_time = asyncio.get_event_loop().time()
        
        # 并行执行
        ocr_result, vl_result = await asyncio.gather(ocr_task, vl_task)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # 4. 输出结果
        print("\n" + "="*50)
        print(f"🎉 测试完成！总耗时: {duration:.2f} 秒")
        print("="*50)

        print("\n📝 [DeepSeek-OCR 结果]:")
        if ocr_result:
            # 只打印前200个字符，避免刷屏
            preview = ocr_result[:200].replace('\n', ' ')
            print(f"📄 内容预览: {preview}...")
            print(f"📊 总字符数: {len(ocr_result)}")
        else:
            print("❌ OCR 返回为空 (可能接口报错或屏幕无文字)")

        print("\n👁️ [VL 视觉分析结果]:")
        if vl_result:
            print(f"📱 应用名称: {vl_result.app_name}")
            print(f"🖥️ 页面状态: {vl_result.screen_state}")
            print(f"📝 描述信息: {vl_result.description[:100]}...")
            if vl_result.available_elements:
                print(f"🔍 发现元素: {', '.join(vl_result.available_elements[:5])}...")
        else:
            print("❌ VL 分析返回为空")

    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 5. 清理资源
        print("\n🧹 正在关闭服务...")
        await vision.close()
        print("✅ 服务已关闭")

if __name__ == "__main__":
    # Windows 下 asyncio 的兼容性设置
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(test_ocr_vl_parallel())