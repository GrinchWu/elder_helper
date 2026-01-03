"""测试 OCR 定位能力 - 在截图中定位指定文字并标红"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
from src.services.vision_service import VisionService


async def test_locate_text(target_text: str = "Run"):
    """
    测试定位指定文字
    
    Args:
        target_text: 要定位的文字，默认为 "Run"
    """
    print(f"=== 测试 OCR 定位: '{target_text}' ===\n")
    
    # 初始化服务
    vision = VisionService()
    await vision.initialize()
    
    try:
        # 1. 截取屏幕
        print("1. 截取屏幕...")
        screenshot_bytes, original_size = await vision.capture_screen()
        
        if not screenshot_bytes:
            print("❌ 截屏失败")
            return
        
        print(f"   截图尺寸: {original_size[0]}x{original_size[1]}")
        
        # 2. 使用 locate_element 定位文字
        print(f"\n2. 定位文字 '{target_text}'...")
        element = await vision.locate_element(
            screenshot=screenshot_bytes,
            element_description=target_text,
            original_size=original_size
        )
        
        if not element:
            print(f"❌ 未找到 '{target_text}'")
            return
        
        # 获取中心点坐标
        center_x, center_y = element.get_center()
        bbox = element.bbox
        
        print(f"   ✅ 找到元素!")
        print(f"   - 类型: {element.element_type}")
        print(f"   - 文字: {element.text}")
        print(f"   - 描述: {element.description}")
        print(f"   - 边界框: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
        print(f"   - 中心点: ({center_x}, {center_y})")
        print(f"   - 置信度: {element.confidence}")
        
        # 3. 在截图上标记红点
        print("\n3. 在截图上标记红点...")
        
        # 重新截图获取原始尺寸图片用于标记
        import mss
        import io
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            img = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.bgra,
                "raw",
                "BGRX",
            )
        
        draw = ImageDraw.Draw(img)
        
        # 画红色圆点（中心点）
        dot_radius = 15
        draw.ellipse(
            [center_x - dot_radius, center_y - dot_radius,
             center_x + dot_radius, center_y + dot_radius],
            fill='red',
            outline='darkred',
            width=3
        )
        
        # 画边界框（红色矩形）
        x, y, w, h = bbox
        draw.rectangle(
            [x, y, x + w, y + h],
            outline='red',
            width=3
        )
        
        # 添加文字标签
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        label = f"'{target_text}' @ ({center_x}, {center_y})"
        draw.text((x, y - 25), label, fill='red', font=font)
        
        # 4. 保存结果
        output_path = Path("tests/ocr_locate_result.png")
        img.save(output_path)
        print(f"   ✅ 结果已保存到: {output_path.absolute()}")
        
        # 打开图片查看
        print("\n4. 打开结果图片...")
        import os
        os.startfile(str(output_path.absolute()))
        
    finally:
        await vision.close()
    
    print("\n=== 测试完成 ===")


async def main():
    """主函数"""
    # 可以通过命令行参数指定要查找的文字
    target = sys.argv[1] if len(sys.argv) > 1 else "Run"
    await test_locate_text(target)


if __name__ == "__main__":
    asyncio.run(main())
