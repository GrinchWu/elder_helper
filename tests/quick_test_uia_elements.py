"""UIA快速测试脚本 - 直接获取UI元素坐标并在截图上标红点

使用方法:
    cd agnet_help_elder
    python tests/quick_test_uia_elements.py

可选参数:
    --backend pywinauto|uiautomation
    --max-elements 50
    --title-regex ".*Chrome.*"
    --output "uia_marked.png"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass(frozen=True)
class UiElementRect:
    name: str
    control_type: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)


def _set_dpi_aware() -> None:
    try:
        import ctypes

        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass

    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _capture_screen_pil():
    import mss
    from PIL import Image

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
        return img


def _collect_rects_pywinauto(
    title_regex: str | None,
    max_elements: int,
) -> tuple[str, list[UiElementRect]]:
    import ctypes

    from pywinauto import Application, Desktop

    desktop = Desktop(backend="uia")
    if title_regex:
        win = desktop.window(title_re=title_regex)
        win.wait("exists", timeout=3)
    else:
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow())
        if not hwnd:
            raise RuntimeError("未获取到前台窗口句柄")
        app = Application(backend="uia").connect(handle=hwnd)
        win = app.window(handle=hwnd)

    win_text = ""
    try:
        win_text = win.window_text()
    except Exception:
        win_text = "<unknown>"

    rects: list[UiElementRect] = []
    try:
        items = win.descendants()
    except Exception:
        items = []

    for item in items:
        if len(rects) >= max_elements:
            break
        try:
            rect = item.rectangle()
            left, top, right, bottom = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
            if right <= left or bottom <= top:
                continue
            name = ""
            control_type = ""
            try:
                name = item.window_text() or ""
            except Exception:
                name = ""
            try:
                if hasattr(item, "friendly_class_name"):
                    control_type = item.friendly_class_name() or ""
                else:
                    control_type = item.element_info.control_type or ""
            except Exception:
                control_type = ""

            rects.append(
                UiElementRect(
                    name=name.strip(),
                    control_type=control_type.strip(),
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )
            )
        except Exception:
            continue

    return win_text, rects


def _collect_rects_uiautomation(
    title_regex: str | None,
    max_elements: int,
) -> tuple[str, list[UiElementRect]]:
    import re

    import uiautomation as auto

    auto.SetGlobalSearchTimeout(2.0)

    if title_regex:
        pattern = re.compile(title_regex)
        target = None
        for w in auto.GetRootControl().GetChildren():
            try:
                n = w.Name or ""
            except Exception:
                n = ""
            if pattern.search(n):
                target = w
                break
        if not target:
            raise RuntimeError(f"未找到匹配窗口: {title_regex}")
        win = target
    else:
        win = auto.GetForegroundControl()

    win_text = ""
    try:
        win_text = win.Name or ""
    except Exception:
        win_text = "<unknown>"

    rects: list[UiElementRect] = []
    for control, _depth in auto.WalkControl(win, maxDepth=8):
        if len(rects) >= max_elements:
            break
        try:
            br = control.BoundingRectangle
            left, top, right, bottom = int(br.left), int(br.top), int(br.right), int(br.bottom)
            if right <= left or bottom <= top:
                continue
            name = ""
            control_type = ""
            try:
                name = control.Name or ""
            except Exception:
                name = ""
            try:
                control_type = control.ControlTypeName or ""
            except Exception:
                control_type = ""
            rects.append(
                UiElementRect(
                    name=name.strip(),
                    control_type=control_type.strip(),
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                )
            )
        except Exception:
            continue

    return win_text, rects


def _draw_red_dots(img, rects: list[UiElementRect]) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    r = 6
    for rect in rects:
        cx, cy = rect.center
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 0, 0), outline=(255, 0, 0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="pywinauto", choices=["pywinauto", "uiautomation"])
    parser.add_argument("--max-elements", type=int, default=50)
    parser.add_argument("--title-regex", default=None)
    parser.add_argument("--output", default="uia_marked.png")
    args = parser.parse_args()

    _set_dpi_aware()

    win_title = ""
    rects: list[UiElementRect] = []
    if args.backend == "pywinauto":
        try:
            win_title, rects = _collect_rects_pywinauto(args.title_regex, args.max_elements)
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            print("   请安装: pip install pywinauto")
            return 2
    else:
        try:
            win_title, rects = _collect_rects_uiautomation(args.title_regex, args.max_elements)
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            print("   请安装: pip install uiautomation")
            return 2

    if not rects:
        print("⚠️ 未获取到任何元素坐标")
        return 1

    img = _capture_screen_pil()
    _draw_red_dots(img, rects)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path
    img.save(out_path, format="PNG", optimize=True)

    print("=" * 60)
    print(f"窗口: {win_title}")
    print(f"backend: {args.backend}")
    print(f"元素数量: {len(rects)}")
    print(f"输出: {out_path}")
    print("-" * 60)
    for i, r in enumerate(rects[: min(len(rects), 30)], start=1):
        print(f"{i:02d}. ({r.left},{r.top})-({r.right},{r.bottom})  {r.control_type}  {r.name}")
    if len(rects) > 30:
        print(f"... 其余 {len(rects) - 30} 个未打印")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
