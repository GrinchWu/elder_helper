"""Vision服务测试脚本 - 测试屏幕分析功能"""

from __future__ import annotations

import asyncio
import sys
import base64
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_screen_capture():
    """测试屏幕截图功能"""
    from src.services.vision_service import VisionService, VLConfig
    
    print("=" * 70)
    print("测试1: 屏幕截图")
    print("=" * 70)
    
    config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(config)
    await vision.initialize()
    
    try:
        print("正在截取屏幕...")
        screenshot, original_size = await vision.capture_screen()
        
        if screenshot:
            print(f"✅ 截图成功! 大小: {len(screenshot) / 1024:.1f} KB")
            print(f"   原始屏幕尺寸: {original_size[0]}x{original_size[1]}")
            
            # 保存截图到文件
            output_path = Path(__file__).parent / "test_screenshot.png"
            with open(output_path, "wb") as f:
                f.write(screenshot)
            print(f"✅ 截图已保存到: {output_path}")
            
            return screenshot, original_size
        else:
            print("❌ 截图失败!")
            return None, (0, 0)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, (0, 0)
    finally:
        await vision.close()


async def test_screen_analysis(screenshot: bytes = None, original_size: tuple[int, int] = (0, 0)):
    """测试屏幕分析功能"""
    from src.services.vision_service import VisionService, VLConfig
    
    print("\n" + "=" * 70)
    print("测试2: 屏幕分析")
    print("=" * 70)
    
    config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(config)
    await vision.initialize()
    
    try:
        # 如果没有传入截图，先截取
        if not screenshot:
            print("正在截取屏幕...")
            screenshot, original_size = await vision.capture_screen()
        
        if not screenshot:
            print("❌ 无法获取截图")
            return
        
        print(f"截图大小: {len(screenshot) / 1024:.1f} KB")
        print(f"原始屏幕尺寸: {original_size[0]}x{original_size[1]}")
        print("正在分析屏幕内容...")
        
        # 分析屏幕（传递原始尺寸以便坐标映射）
        analysis = await vision.analyze_screen(screenshot, original_size=original_size)
        
        print("\n📊 分析结果:")
        print(f"  应用名称: {analysis.app_name or '未识别'}")
        print(f"  屏幕类型: {analysis.screen_type or '未识别'}")
        print(f"  屏幕描述: {analysis.description or '无'}")
        
        if analysis.elements:
            print(f"\n  ✅ 识别到 {len(analysis.elements)} 个元素:")
            for i, elem in enumerate(analysis.elements[:15], 1):  # 显示前15个
                clickable = "🖱️" if elem.is_clickable else ""
                input_mark = "⌨️" if elem.is_input else ""
                print(f"    {i}. [{elem.element_type}] {elem.text or elem.description} {clickable}{input_mark}")
                if elem.bbox != (0, 0, 0, 0):
                    print(f"       位置: {elem.bbox}")
            
            if len(analysis.elements) > 15:
                print(f"    ... 还有 {len(analysis.elements) - 15} 个元素")
        
        if analysis.suggested_actions:
            print(f"\n  💡 建议操作:")
            for action in analysis.suggested_actions[:3]:
                print(f"    - {action}")
        
        if analysis.warnings:
            print(f"\n  ⚠️ 安全警告:")
            for warning in analysis.warnings:
                print(f"    - {warning}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await vision.close()


async def test_screen_analysis_with_intent():
    """测试带意图的屏幕分析"""
    from src.services.vision_service import VisionService, VLConfig
    
    print("\n" + "=" * 70)
    print("测试3: 带意图的屏幕分析")
    print("=" * 70)
    
    config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(config)
    await vision.initialize()
    
    try:
        print("正在截取屏幕...")
        screenshot, original_size = await vision.capture_screen()
        
        if not screenshot:
            print("❌ 无法获取截图")
            return
        
        # 测试不同的用户意图
        intents = [
            "我想打开浏览器上网",
            "我想找到微信",
            "我想写个文档",
        ]
        
        for intent in intents:
            print(f"\n{'='*50}")
            print(f"用户意图: {intent}")
            print("-" * 50)
            
            analysis = await vision.analyze_screen(screenshot, user_intent=intent, original_size=original_size)
            
            print(f"屏幕描述: {analysis.description[:100] if analysis.description else '无'}...")
            
            if analysis.suggested_actions:
                print("建议操作:")
                for action in analysis.suggested_actions[:2]:
                    print(f"  - {action}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await vision.close()


async def test_find_element():
    """测试查找特定元素"""
    from src.services.vision_service import VisionService, VLConfig
    
    print("\n" + "=" * 70)
    print("测试4: 查找特定元素")
    print("=" * 70)
    
    config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(config)
    await vision.initialize()
    
    try:
        print("正在截取屏幕...")
        screenshot, original_size = await vision.capture_screen()
        
        if not screenshot:
            print("❌ 无法获取截图")
            return
        
        # 测试查找不同元素
        elements_to_find = [
            "开始按钮",
            "浏览器图标",
            "搜索框",
        ]
        
        for desc in elements_to_find:
            print(f"\n查找: {desc}")
            element = await vision.find_element(screenshot, desc, original_size=original_size)
            
            if element:
                print(f"  ✅ 找到: [{element.element_type}] {element.text or element.description}")
                print(f"     位置: {element.bbox}")
            else:
                print(f"  ❌ 未找到")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await vision.close()


async def test_with_image_file(image_path: str):
    """使用指定图片文件测试"""
    from src.services.vision_service import VisionService, VLConfig
    
    print("\n" + "=" * 70)
    print(f"测试: 分析图片文件 - {image_path}")
    print("=" * 70)
    
    config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(config)
    await vision.initialize()
    
    try:
        # 读取图片文件
        with open(image_path, "rb") as f:
            screenshot = f.read()
        
        print(f"图片大小: {len(screenshot) / 1024:.1f} KB")
        print("正在分析...")
        
        analysis = await vision.analyze_screen(screenshot)
        
        print("\n📊 分析结果:")
        print(f"  应用名称: {analysis.app_name or '未识别'}")
        print(f"  屏幕类型: {analysis.screen_type or '未识别'}")
        print(f"  屏幕描述: {analysis.description or '无'}")
        
        if analysis.elements:
            print(f"\n  识别到 {len(analysis.elements)} 个元素:")
            for i, elem in enumerate(analysis.elements[:10], 1):
                print(f"    {i}. [{elem.element_type}] {elem.text or elem.description}")
        
        if analysis.warnings:
            print(f"\n  ⚠️ 安全警告:")
            for warning in analysis.warnings:
                print(f"    - {warning}")
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {image_path}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await vision.close()


async def interactive_test():
    """交互式测试"""
    from src.services.vision_service import VisionService, VLConfig
    
    print("=" * 70)
    print("交互式Vision测试")
    print("输入 'quit' 退出")
    print("输入 'capture' 截取并分析当前屏幕")
    print("输入其他内容作为用户意图进行分析")
    print("=" * 70)
    
    config = VLConfig(
        api_key="CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ",
        model="Qwen3-VL-235B-A22B-Instruct",
    )
    
    vision = VisionService(config)
    await vision.initialize()
    
    try:
        while True:
            user_input = input("\n请输入 (capture/意图/quit): ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if not user_input:
                continue
            
            print("\n正在截取屏幕...")
            screenshot, original_size = await vision.capture_screen()
            
            if not screenshot:
                print("❌ 截图失败")
                continue
            
            intent = "" if user_input.lower() == "capture" else user_input
            
            print("正在分析...")
            analysis = await vision.analyze_screen(screenshot, user_intent=intent, original_size=original_size)
            
            print(f"\n📊 分析结果:")
            print(f"  应用: {analysis.app_name or '未识别'}")
            print(f"  描述: {analysis.description[:200] if analysis.description else '无'}...")
            
            if analysis.suggested_actions:
                print(f"\n  建议操作:")
                for action in analysis.suggested_actions[:3]:
                    print(f"    - {action}")
            
            if analysis.warnings:
                print(f"\n  ⚠️ 警告: {analysis.warnings}")
                
    except KeyboardInterrupt:
        pass
    finally:
        await vision.close()
        print("\n再见！")


def main():
    """主函数"""
    print("Vision服务测试")
    print("=" * 70)
    print("1. 屏幕截图测试")
    print("2. 屏幕分析测试")
    print("3. 带意图的屏幕分析")
    print("4. 查找特定元素")
    print("5. 交互式测试")
    print("6. 分析指定图片文件")
    print("7. 运行所有测试")
    print("=" * 70)
    
    choice = input("请选择测试项 (1-7): ").strip()
    
    if choice == "1":
        asyncio.run(test_screen_capture())
    elif choice == "2":
        asyncio.run(test_screen_analysis())
    elif choice == "3":
        asyncio.run(test_screen_analysis_with_intent())
    elif choice == "4":
        asyncio.run(test_find_element())
    elif choice == "5":
        asyncio.run(interactive_test())
    elif choice == "6":
        image_path = input("请输入图片路径: ").strip()
        if image_path:
            asyncio.run(test_with_image_file(image_path))
    elif choice == "7":
        async def run_all():
            screenshot, original_size = await test_screen_capture()
            if screenshot:
                await test_screen_analysis(screenshot, original_size)
            await test_screen_analysis_with_intent()
            await test_find_element()
        asyncio.run(run_all())
    else:
        print("无效选择，运行屏幕分析测试...")
        asyncio.run(test_screen_analysis())


if __name__ == "__main__":
    main()
