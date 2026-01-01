"""任务执行器测试脚本 - 测试完整的任务执行流程"""

from __future__ import annotations

import asyncio
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
from typing import Optional
from queue import Queue
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class FeedbackWindow:
    """用户反馈悬浮窗口"""
    
    def __init__(self):
        self.root: Optional[tk.Tk] = None
        self.status_label: Optional[tk.Label] = None
        self.step_label: Optional[tk.Label] = None
        self.progress_label: Optional[tk.Label] = None
        self.log_text: Optional[scrolledtext.ScrolledText] = None
        self.feedback_entry: Optional[tk.Entry] = None
        self.feedback_queue: Queue[str] = Queue()
        self.message_queue: Queue[tuple[str, str]] = Queue()  # (type, message)
        self._is_running = False
    
    def create_window(self):
        """创建窗口"""
        self.root = tk.Tk()
        self.root.title("任务助手")
        self.root.geometry("450x500+50+50")
        self.root.attributes('-topmost', True)
        self.root.configure(bg='#f0f0f0')
        
        # 标题
        title_frame = tk.Frame(self.root, bg='#4a90d9', pady=10)
        title_frame.pack(fill='x')
        tk.Label(
            title_frame, 
            text="🤖 老年人电脑助手", 
            font=("微软雅黑", 16, "bold"),
            bg='#4a90d9',
            fg='white'
        ).pack()
        
        # 进度显示
        progress_frame = tk.Frame(self.root, bg='#f0f0f0', pady=5)
        progress_frame.pack(fill='x', padx=10)
        self.progress_label = tk.Label(
            progress_frame,
            text="进度: 0/0",
            font=("微软雅黑", 10),
            bg='#f0f0f0'
        )
        self.progress_label.pack(anchor='w')
        
        # 当前步骤
        step_frame = tk.LabelFrame(self.root, text="当前步骤", font=("微软雅黑", 10), bg='#f0f0f0', pady=5)
        step_frame.pack(fill='x', padx=10, pady=5)
        self.step_label = tk.Label(
            step_frame,
            text="等待开始...",
            font=("微软雅黑", 12),
            bg='#f0f0f0',
            fg='#333',
            wraplength=400,
            justify='left'
        )
        self.step_label.pack(anchor='w', padx=5)
        
        # 状态显示
        status_frame = tk.LabelFrame(self.root, text="状态", font=("微软雅黑", 10), bg='#f0f0f0')
        status_frame.pack(fill='x', padx=10, pady=5)
        self.status_label = tk.Label(
            status_frame,
            text="准备就绪",
            font=("微软雅黑", 10),
            bg='#f0f0f0',
            fg='#666'
        )
        self.status_label.pack(anchor='w', padx=5, pady=2)
        
        # 日志区域
        log_frame = tk.LabelFrame(self.root, text="执行日志", font=("微软雅黑", 10), bg='#f0f0f0')
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Consolas", 9),
            wrap='word'
        )
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 反馈输入区域
        feedback_frame = tk.LabelFrame(self.root, text="遇到问题？告诉我", font=("微软雅黑", 10), bg='#f0f0f0')
        feedback_frame.pack(fill='x', padx=10, pady=5)
        
        input_frame = tk.Frame(feedback_frame, bg='#f0f0f0')
        input_frame.pack(fill='x', padx=5, pady=5)
        
        self.feedback_entry = tk.Entry(input_frame, font=("微软雅黑", 10))
        self.feedback_entry.pack(side='left', fill='x', expand=True)
        self.feedback_entry.bind('<Return>', self._on_submit_feedback)
        
        submit_btn = tk.Button(
            input_frame,
            text="发送",
            font=("微软雅黑", 10),
            command=self._on_submit_feedback,
            bg='#4a90d9',
            fg='white'
        )
        submit_btn.pack(side='right', padx=(5, 0))
        
        # 关闭按钮
        close_btn = tk.Button(
            self.root,
            text="关闭助手",
            font=("微软雅黑", 10),
            command=self.close,
            bg='#d9534f',
            fg='white'
        )
        close_btn.pack(pady=10)
        
        self._is_running = True
        
        # 启动消息处理
        self._process_messages()
        
        return self
    
    def _process_messages(self):
        """处理消息队列"""
        if not self._is_running or not self.root:
            return
        
        try:
            while not self.message_queue.empty():
                msg_type, message = self.message_queue.get_nowait()
                if msg_type == "step":
                    self.step_label.config(text=message)
                elif msg_type == "status":
                    self.status_label.config(text=message)
                elif msg_type == "progress":
                    self.progress_label.config(text=message)
                elif msg_type == "log":
                    self.log_text.insert('end', message + '\n')
                    self.log_text.see('end')
        except:
            pass
        
        # 继续处理
        if self._is_running and self.root:
            self.root.after(100, self._process_messages)
    
    def _on_submit_feedback(self, event=None):
        """提交反馈"""
        if self.feedback_entry:
            feedback = self.feedback_entry.get().strip()
            if feedback:
                self.feedback_queue.put(feedback)
                self.feedback_entry.delete(0, 'end')
                self.add_log(f"📝 已发送反馈: {feedback}")
    
    def update_step(self, step_text: str):
        """更新当前步骤"""
        if self._is_running:
            self.message_queue.put(("step", step_text))
    
    def update_status(self, status: str):
        """更新状态"""
        if self._is_running:
            self.message_queue.put(("status", status))
    
    def update_progress(self, current: int, total: int):
        """更新进度"""
        if self._is_running:
            self.message_queue.put(("progress", f"进度: {current}/{total}"))
    
    def add_log(self, message: str):
        """添加日志"""
        if self._is_running:
            self.message_queue.put(("log", message))
    
    def get_feedback(self) -> Optional[str]:
        """获取用户反馈"""
        try:
            return self.feedback_queue.get_nowait()
        except:
            return None
    
    def run(self):
        """运行窗口"""
        if self.root:
            self.root.mainloop()
    
    def close(self):
        """关闭窗口"""
        self._is_running = False
        if self.root:
            self.root.quit()
            self.root.destroy()
            self.root = None


async def run_executor_task(window: FeedbackWindow, user_input: str):
    """在后台运行执行器任务"""
    from src.services.executor_service import ExecutorService
    from src.models.intent import Intent, IntentType
    from src.models.task import TaskStep
    
    # 初始化执行器
    executor = ExecutorService()
    await executor.initialize()
    
    # 设置回调
    def on_step_start(step: TaskStep):
        window.update_step(f"📋 {step.friendly_instruction or step.description}")
        current, total = executor.get_progress()
        window.update_progress(current, total)
    
    def on_step_complete(step: TaskStep, success: bool):
        status = "✅ 完成" if success else "❌ 失败"
        window.add_log(f"步骤 {step.step_number}: {status}")
    
    def on_status_update(message: str):
        window.update_status(message)
        window.add_log(message)
    
    def on_need_replan(reason: str):
        window.add_log(f"🔄 重新规划: {reason}")
    
    executor.set_callbacks(
        on_step_start=on_step_start,
        on_step_complete=on_step_complete,
        on_status_update=on_status_update,
        on_need_replan=on_need_replan,
    )
    
    try:
        window.add_log(f"📝 任务: {user_input}")
        
        # 创建意图
        intent = Intent(
            raw_text=user_input,
            normalized_text=user_input,
            intent_type=IntentType.NAVIGATION,
        )
        
        # 启动反馈检查
        async def check_feedback():
            while window._is_running:
                feedback = window.get_feedback()
                if feedback:
                    executor.submit_user_feedback(feedback)
                await asyncio.sleep(0.5)
        
        feedback_task = asyncio.create_task(check_feedback())
        
        # 执行任务
        task = await executor.execute_task(intent)
        
        # 取消反馈检查
        feedback_task.cancel()
        
        # 显示结果
        window.add_log(f"\n{'='*50}")
        window.add_log(f"任务状态: {task.status.value}")
        
    except Exception as e:
        window.add_log(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await executor.close()


def test_executor_with_ui():
    """带UI的执行器测试"""
    print("=" * 70)
    print("任务执行器测试（带反馈窗口）")
    print("=" * 70)
    
    # 获取用户输入
    print("\n请输入您的需求:")
    user_input = input(">>> ").strip()
    
    if not user_input:
        user_input = "打开浏览器"
    
    # 创建反馈窗口
    window = FeedbackWindow()
    window.create_window()
    
    # 在后台线程运行异步任务
    def run_async_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_executor_task(window, user_input))
        finally:
            loop.close()
    
    task_thread = threading.Thread(target=run_async_task, daemon=True)
    task_thread.start()
    
    # 在主线程运行Tkinter
    window.run()
    
    print("\n测试结束")


async def test_executor_simple():
    """简单的执行器测试（无UI）"""
    from src.services.executor_service import ExecutorService
    from src.models.intent import Intent, IntentType
    
    print("=" * 70)
    print("任务执行器测试（控制台模式）")
    print("=" * 70)
    
    executor = ExecutorService()
    await executor.initialize()
    
    # 设置回调
    def on_status_update(message: str):
        print(message)
    
    executor.set_callbacks(on_status_update=on_status_update)
    
    try:
        print("\n请输入您的需求（例如：打开微信、打开浏览器搜索天气）:")
        user_input = input(">>> ").strip()
        
        if not user_input:
            print("未输入需求，退出")
            return
        
        # 创建意图
        intent = Intent(
            raw_text=user_input,
            normalized_text=user_input,
            intent_type=IntentType.NAVIGATION,
        )
        
        # 执行任务
        task = await executor.execute_task(intent)
        
        print(f"\n{'='*70}")
        print(f"任务完成，状态: {task.status.value}")
        
    except KeyboardInterrupt:
        print("\n已中断")
    finally:
        await executor.close()


async def test_plan_only():
    """只测试计划生成（不执行）"""
    from src.services.vision_service import VisionService, VLConfig
    from src.services.planner_service import PlannerService
    from src.models.intent import Intent, IntentType
    from src.config import config
    
    print("=" * 70)
    print("任务计划生成测试")
    print("=" * 70)
    
    # 初始化服务
    vl_config = VLConfig(
        api_key=config.api.api_key,
        model=config.api.vl_model,
    )
    vision = VisionService(vl_config)
    await vision.initialize()
    
    planner = PlannerService()
    await planner.initialize()
    
    try:
        print("\n请输入您的需求:")
        user_input = input(">>> ").strip()
        
        if not user_input:
            user_input = "打开微信"
        
        print(f"\n任务: {user_input}")
        
        # 截取屏幕
        print("\n📸 截取屏幕...")
        screenshot, original_size = await vision.capture_screen()
        
        print("🔍 分析屏幕...")
        screen_analysis = await vision.analyze_screen(
            screenshot,
            user_intent=user_input,
            original_size=original_size
        )
        
        print(f"   应用: {screen_analysis.app_name or '未识别'}")
        print(f"   元素数: {len(screen_analysis.elements)}")
        
        # 创建意图
        intent = Intent(
            raw_text=user_input,
            normalized_text=user_input,
            intent_type=IntentType.NAVIGATION,
        )
        
        # 生成计划
        print("\n🧠 生成任务计划...")
        plan = await planner.create_plan(
            intent=intent,
            screen_analysis=screen_analysis,
        )
        
        # 显示计划
        print(f"\n{'='*70}")
        print(f"📋 任务计划: {user_input}")
        print(f"{'='*70}")
        
        if plan.steps:
            for step in plan.steps:
                print(f"\n【步骤 {step.step_number}】")
                print(f"  📝 {step.description}")
                print(f"  👴 {step.friendly_instruction}")
                if step.action:
                    print(f"  🎯 动作: {step.action.action_type.value}")
                    if step.action.element_description:
                        print(f"     目标: {step.action.element_description}")
                if step.expected_result:
                    print(f"  ✅ 预期: {step.expected_result}")
                if step.error_recovery_hint:
                    print(f"  ⚠️ 出错处理: {step.error_recovery_hint}")
        else:
            print("❌ 未能生成计划")
        
    finally:
        await vision.close()
        await planner.close()


def main():
    """主函数"""
    print("任务执行器测试")
    print("=" * 70)
    print("1. 带反馈窗口的完整测试（推荐）")
    print("2. 控制台模式测试")
    print("3. 只测试计划生成")
    print("=" * 70)
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        test_executor_with_ui()
    elif choice == "2":
        asyncio.run(test_executor_simple())
    elif choice == "3":
        asyncio.run(test_plan_only())
    else:
        print("默认运行只测试计划生成...")
        asyncio.run(test_plan_only())


if __name__ == "__main__":
    main()
