"""屏幕元素标注测试 - 在屏幕上实时显示识别到的元素位置"""

from __future__ import annotations

import asyncio
import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class ScreenOverlay:
    """屏幕覆盖层 - 用于在屏幕上标注元素"""
    
    def __init__(self):
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.labels: list[tk.Label] = []
    
    def create_overlay(self):
        """创建透明覆盖层窗口"""
        self.root = tk.Tk()
        self.root.title("元素标注")
        
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 设置窗口属性
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.overrideredirect(True)  # 无边框
        self.root.attributes('-topmost', True)  # 置顶
        self.root.attributes('-transparentcolor', 'white')  # 白色透明
        self.root.configure(bg='white')
        
        # 创建画布
        self.canvas = tk.Canvas(
            self.root,
            width=screen_width,
            height=screen_height,
            bg='white',
            highlightthickness=0
        )
        self.canvas.pack()
        
        # 绑定ESC键关闭
        self.root.bind('<Escape>', lambda e: self.close())
        # 绑定点击关闭
        self.root.bind('<Button-1>', lambda e: self.close())
        
        return self
    
    def draw_element(self, x: int, y: int, width: int, height: int, 
                     text: str = "", color: str = "red", index: int = 0):
        """在屏幕上绘制元素标注"""
        if not self.canvas:
            return
        
        # 绘制矩形边框
        self.canvas.create_rectangle(
            x, y, x + width, y + height,
            outline=color,
            width=3
        )
        
        # 绘制序号圆圈
        circle_radius = 12
        circle_x = x - circle_radius
        circle_y = y - circle_radius
        self.canvas.create_oval(
            circle_x, circle_y,
            circle_x + circle_radius * 2,
            circle_y + circle_radius * 2,
            fill=color,
            outline=color
        )
        
        # 绘制序号文字
        self.canvas.create_text(
            circle_x + circle_radius,
            circle_y + circle_radius,
            text=str(index),
            fill="white",
            font=("Arial", 10, "bold")
        )
        
        # 如果有文本，显示标签
        if text:
            # 创建标签背景
            label_y = y + height + 5
            self.canvas.create_rectangle(
                x, label_y,
                x + len(text) * 8 + 10, label_y + 20,
                fill=color,
                outline=color
            )
            self.canvas.create_text(
                x + 5, label_y + 10,
                text=text[:30],  # 限制长度
                fill="white",
                anchor="w",
                font=("Arial", 9)
            )
    
    def show(self, duration: int = 5000):
        """显示覆盖层"""
        if self.root:
            # 设置自动关闭
            self.root.after(duration, self.close)
            self.root.mainloop()
    
    def close(self):
        """关闭覆盖层"""
        if self.root:
            self.root.destroy()
            self.root = None


async def test_overlay_with_vision_ai():
    """使用视觉AI获取元素并标注"""
    from src.services.vision_service import VisionService, VLConfig
    
    print("=" * 70)
    print("测试: 视觉AI分析 + 屏幕标注")
    print("=" * 70)
    print("将截取屏幕，使用AI分析后显示标注")
    print("-" * 70)
    
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
            print("❌ 截图失败")
            return
        
        print(f"截图大小: {len(screenshot) / 1024:.1f} KB")
        print(f"原始屏幕尺寸: {original_size[0]}x{original_size[1]}")
        print("正在使用AI分析屏幕...")
        
        # 传递原始尺寸以便坐标映射
        analysis = await vision.analyze_screen(screenshot, original_size=original_size)
        
        if not analysis.elements:
            print("❌ 未识别到元素")
            print(f"描述: {analysis.description[:200] if analysis.description else '无'}")
            return
        
        print(f"✅ 识别到 {len(analysis.elements)} 个元素")
        
        # 过滤有效元素
        valid_elements = [
            e for e in analysis.elements 
            if e.bbox != (0, 0, 0, 0) and e.bbox[2] > 5 and e.bbox[3] > 5
        ]
        
        print(f"有坐标的元素: {len(valid_elements)} 个")
        
        if not valid_elements:
            print("⚠️ 没有元素有有效坐标")
            print("\n所有元素:")
            for i, elem in enumerate(analysis.elements[:10], 1):
                print(f"  {i}. [{elem.element_type}] {elem.text or elem.description}")
                print(f"      bbox: {elem.bbox}")
            return
        
        # 显示元素列表
        print("\n将标注以下元素:")
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]
        for i, elem in enumerate(valid_elements[:15], 1):
            color = colors[(i-1) % len(colors)]
            text = elem.text or elem.description
            print(f"  {i}. [{elem.element_type}] {text[:30] if text else '无'}")
            print(f"      位置: ({elem.bbox[0]}, {elem.bbox[1]}) 大小: {elem.bbox[2]}x{elem.bbox[3]}")
        
        print("\n3秒后显示标注...")
        await asyncio.sleep(3)
        
        # 创建覆盖层并标注
        overlay = ScreenOverlay()
        overlay.create_overlay()
        
        for i, elem in enumerate(valid_elements[:15], 1):
            color = colors[(i-1) % len(colors)]
            text = elem.text or elem.description
            overlay.draw_element(
                x=elem.bbox[0],
                y=elem.bbox[1],
                width=elem.bbox[2],
                height=elem.bbox[3],
                text=text[:20] if text else "",
                color=color,
                index=i
            )
        
        print("标注已显示！按ESC或点击屏幕关闭")
        overlay.show(duration=10000)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await vision.close()


def main():
    """主函数"""
    print("屏幕元素标注测试")
    print("=" * 70)
    print("使用视觉AI分析屏幕并标注识别到的元素")
    print("=" * 70)
    
    asyncio.run(test_overlay_with_vision_ai())


if __name__ == "__main__":
    main()
