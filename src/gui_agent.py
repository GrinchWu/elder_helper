"""带GUI界面的老年人助手Agent"""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
from typing import Optional
import sys

from loguru import logger

from .config import config
from .models.intent import Intent, IntentType
from .models.task import Task, TaskStatus, TaskPlan
from .models.session import UserProfile
from .models.knowledge import KnowledgeGraph
from .services.llm_service import LLMService
from .services.vision_service import VisionService, VLConfig, ScreenAnalysis
from .services.planner_service import PlannerService
from .services.safety_service import SafetyService
from .services.executor_service import ExecutorService
from .services.embedding_service import EmbeddingService
from .knowledge.rag_service import RAGService


class AgentGUI:
    """老年人助手GUI界面"""
    
    def __init__(self):
        self._root: Optional[tk.Tk] = None
        self._agent: Optional[GUIElderlyAgent] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._agent_thread: Optional[threading.Thread] = None
        
        # UI组件
        self._input_entry: Optional[tk.Entry] = None
        self._output_text: Optional[scrolledtext.ScrolledText] = None
        self._status_label: Optional[tk.Label] = None
        self._send_btn: Optional[tk.Button] = None
        self._feedback_entry: Optional[tk.Entry] = None
        self._feedback_btn: Optional[tk.Button] = None
        
        # 状态
        self._is_processing = False
        self._current_task: Optional[Task] = None
    
    def run(self):
        """启动GUI"""
        self._create_window()
        self._start_agent_thread()
        self._root.mainloop()
    
    def _create_window(self):
        """创建主窗口"""
        self._root = tk.Tk()
        self._root.title("🤖 老年人电脑助手")
        self._root.geometry("500x600")
        self._root.resizable(True, True)
        
        # 设置窗口始终在最前面
        self._root.attributes('-topmost', True)
        
        # 设置样式
        style = ttk.Style()
        style.configure('TButton', font=('Microsoft YaHei', 11))
        style.configure('TLabel', font=('Microsoft YaHei', 10))
        
        # 主框架
        main_frame = ttk.Frame(self._root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(
            main_frame, 
            text="🤖 老年人电脑助手", 
            font=('Microsoft YaHei', 16, 'bold')
        )
        title_label.pack(pady=(0, 10))
        
        # 状态栏
        self._status_label = ttk.Label(
            main_frame, 
            text="⏳ 正在初始化...", 
            font=('Microsoft YaHei', 10)
        )
        self._status_label.pack(pady=(0, 5))
        
        # 输出区域
        output_frame = ttk.LabelFrame(main_frame, text="对话记录", padding="5")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self._output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap=tk.WORD,
            font=('Microsoft YaHei', 11),
            height=15,
            state=tk.DISABLED,
        )
        self._output_text.pack(fill=tk.BOTH, expand=True)
        
        # 配置文本标签样式
        self._output_text.tag_configure('user', foreground='#2196F3', font=('Microsoft YaHei', 11, 'bold'))
        self._output_text.tag_configure('agent', foreground='#4CAF50')
        self._output_text.tag_configure('system', foreground='#9E9E9E', font=('Microsoft YaHei', 9))
        self._output_text.tag_configure('warning', foreground='#FF9800')
        self._output_text.tag_configure('error', foreground='#F44336')
        self._output_text.tag_configure('success', foreground='#4CAF50', font=('Microsoft YaHei', 11, 'bold'))
        
        # 输入区域
        input_frame = ttk.LabelFrame(main_frame, text="请告诉我您想做什么", padding="5")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        input_inner = ttk.Frame(input_frame)
        input_inner.pack(fill=tk.X)
        
        self._input_entry = ttk.Entry(input_inner, font=('Microsoft YaHei', 12))
        self._input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self._input_entry.bind('<Return>', lambda e: self._on_send())
        
        self._send_btn = ttk.Button(input_inner, text="发送", command=self._on_send)
        self._send_btn.pack(side=tk.RIGHT)
        
        # 反馈区域
        feedback_frame = ttk.LabelFrame(main_frame, text="💬 反馈（如果操作不对，告诉我）", padding="5")
        feedback_frame.pack(fill=tk.X, pady=(0, 10))
        
        feedback_inner = ttk.Frame(feedback_frame)
        feedback_inner.pack(fill=tk.X)
        
        self._feedback_entry = ttk.Entry(feedback_inner, font=('Microsoft YaHei', 11))
        self._feedback_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self._feedback_entry.bind('<Return>', lambda e: self._on_feedback())
        
        self._feedback_btn = ttk.Button(feedback_inner, text="反馈", command=self._on_feedback)
        self._feedback_btn.pack(side=tk.RIGHT)
        
        # 快捷按钮
        quick_frame = ttk.Frame(main_frame)
        quick_frame.pack(fill=tk.X)
        
        quick_btns = [
            ("打开微信", "帮我打开微信"),
            ("打开浏览器", "帮我打开浏览器"),
            ("关闭弹窗", "屏幕上有个东西关不掉"),
        ]
        
        for text, cmd in quick_btns:
            btn = ttk.Button(
                quick_frame, 
                text=text, 
                command=lambda c=cmd: self._quick_command(c)
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # 退出按钮
        exit_btn = ttk.Button(quick_frame, text="退出", command=self._on_exit)
        exit_btn.pack(side=tk.RIGHT)
        
        # 窗口关闭事件
        self._root.protocol("WM_DELETE_WINDOW", self._on_exit)
        
        # 初始欢迎消息
        self._append_output("🤖 助手", "您好！我是您的电脑助手。\n请告诉我您想做什么，比如：\n• 帮我打开微信\n• 我想看看新闻\n• 屏幕上有个东西关不掉", 'agent')
    
    def _start_agent_thread(self):
        """启动Agent线程"""
        def run_agent():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            self._agent = GUIElderlyAgent(self)
            
            try:
                self._loop.run_until_complete(self._agent.initialize())
                self._update_status("✅ 准备就绪")
                self._loop.run_forever()
            except Exception as e:
                logger.error(f"Agent线程错误: {e}")
                self._update_status(f"❌ 初始化失败: {e}")
            finally:
                if self._agent:
                    self._loop.run_until_complete(self._agent.close())
                self._loop.close()
        
        self._agent_thread = threading.Thread(target=run_agent, daemon=True)
        self._agent_thread.start()
    
    def _on_send(self):
        """发送按钮点击"""
        if self._is_processing:
            return
        
        user_input = self._input_entry.get().strip()
        if not user_input:
            return
        
        self._input_entry.delete(0, tk.END)
        self._process_input(user_input)
    
    def _on_feedback(self):
        """反馈按钮点击"""
        feedback = self._feedback_entry.get().strip()
        if not feedback:
            return
        
        self._feedback_entry.delete(0, tk.END)
        self._append_output("👤 您的反馈", feedback, 'user')
        
        # 提交反馈给Agent
        if self._agent and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._agent.process_feedback(feedback),
                self._loop
            )
    
    def _quick_command(self, command: str):
        """快捷命令"""
        self._input_entry.delete(0, tk.END)
        self._input_entry.insert(0, command)
        self._on_send()
    
    def _process_input(self, user_input: str):
        """处理用户输入"""
        self._append_output("👤 您", user_input, 'user')
        self._set_processing(True)
        
        if self._agent and self._loop:
            asyncio.run_coroutine_threadsafe(
                self._agent.process_input(user_input),
                self._loop
            )
    
    def _on_exit(self):
        """退出"""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._root.destroy()
    
    # ========== UI更新方法（线程安全）==========
    
    def _append_output(self, sender: str, message: str, tag: str = 'agent'):
        """添加输出（线程安全）"""
        def update():
            self._output_text.config(state=tk.NORMAL)
            timestamp = datetime.now().strftime("%H:%M")
            self._output_text.insert(tk.END, f"\n[{timestamp}] {sender}:\n", tag)
            self._output_text.insert(tk.END, f"{message}\n", tag if tag != 'user' else 'agent')
            self._output_text.see(tk.END)
            self._output_text.config(state=tk.DISABLED)
        
        if self._root:
            self._root.after(0, update)
    
    def _update_status(self, status: str):
        """更新状态（线程安全）"""
        def update():
            if self._status_label:
                self._status_label.config(text=status)
        
        if self._root:
            self._root.after(0, update)
    
    def _set_processing(self, processing: bool):
        """设置处理状态（线程安全）"""
        def update():
            self._is_processing = processing
            if self._send_btn:
                self._send_btn.config(state=tk.DISABLED if processing else tk.NORMAL)
            if self._input_entry:
                self._input_entry.config(state=tk.DISABLED if processing else tk.NORMAL)
        
        if self._root:
            self._root.after(0, update)
    
    def show_message(self, message: str, tag: str = 'agent'):
        """显示消息"""
        self._append_output("🤖 助手", message, tag)
    
    def show_system(self, message: str):
        """显示系统消息"""
        def update():
            self._output_text.config(state=tk.NORMAL)
            self._output_text.insert(tk.END, f"  {message}\n", 'system')
            self._output_text.see(tk.END)
            self._output_text.config(state=tk.DISABLED)
        
        if self._root:
            self._root.after(0, update)
    
    def ask_confirmation(self, question: str, callback):
        """询问确认（弹窗）"""
        def ask():
            from tkinter import messagebox
            result = messagebox.askyesno("确认", question)
            if self._loop:
                asyncio.run_coroutine_threadsafe(callback(result), self._loop)
        
        if self._root:
            self._root.after(0, ask)
    
    def done_processing(self):
        """处理完成"""
        self._set_processing(False)


class GUIElderlyAgent:
    """带GUI的老年人助手Agent"""
    
    def __init__(self, gui: AgentGUI):
        self._gui = gui
        
        self._llm: Optional[LLMService] = None
        self._vision: Optional[VisionService] = None
        self._planner: Optional[PlannerService] = None
        self._safety: Optional[SafetyService] = None
        self._executor: Optional[ExecutorService] = None
        self._embedding: Optional[EmbeddingService] = None
        self._rag: Optional[RAGService] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        
        self._user_profile: Optional[UserProfile] = None
        self._current_plan: Optional[TaskPlan] = None
        self._current_intent: Optional[Intent] = None

    async def initialize(self):
        """初始化所有服务"""
        self._gui._update_status("⏳ 初始化意图理解服务...")
        self._llm = LLMService()
        await self._llm.initialize()
        
        self._gui._update_status("⏳ 初始化视觉服务...")
        vl_config = VLConfig(
            api_key=config.api.api_key,
            model_light=config.api.vl_model_light,
            model_heavy=config.api.vl_model_heavy,
        )
        self._vision = VisionService(vl_config)
        await self._vision.initialize()
        
        self._gui._update_status("⏳ 初始化规划服务...")
        self._planner = PlannerService()
        await self._planner.initialize()
        
        self._gui._update_status("⏳ 初始化安全服务...")
        self._safety = SafetyService()
        
        self._gui._update_status("⏳ 初始化知识检索服务...")
        self._embedding = EmbeddingService()
        await self._embedding.initialize()
        
        self._knowledge_graph = KnowledgeGraph()
        
        self._rag = RAGService()
        await self._rag.initialize(
            embedding_service=self._embedding,
            knowledge_graph=self._knowledge_graph,
        )
        
        self._planner.set_rag_service(self._rag)
        
        self._gui._update_status("⏳ 初始化执行服务...")
        self._executor = ExecutorService()
        self._executor.set_vision_service(self._vision)
        self._executor.set_planner_service(self._planner)
        await self._executor.initialize()
        
        # 设置执行器回调
        self._executor.set_callbacks(
            on_status_update=lambda msg: self._gui.show_system(msg),
            on_ask_user=lambda q: self._gui.show_message(f"❓ {q}", 'warning'),
        )
        
        # 设置默认用户画像
        self._user_profile = UserProfile(
            name="用户",
            family_mapping={"老二": "张小明", "闺女": "张小红"},
            frequent_contacts=["张小明", "张小红"],
        )
        
        logger.info("GUI Agent 初始化完成")
    
    async def close(self):
        """关闭所有服务"""
        if self._llm:
            await self._llm.close()
        if self._vision:
            await self._vision.close()
        if self._planner:
            await self._planner.close()
        if self._executor:
            await self._executor.close()
        if self._embedding:
            await self._embedding.close()
    
    async def process_input(self, user_input: str):
        """处理用户输入"""
        try:
            # 1. 安全检查
            self._gui._update_status("🛡️ 安全检查中...")
            safety_result = self._safety.check_text_safety(user_input)
            if not safety_result.is_safe:
                if safety_result.blocked_reason:
                    self._gui.show_message(f"⚠️ 安全警告：{safety_result.blocked_reason}", 'warning')
                    self._gui.done_processing()
                    return
                else:
                    self._gui.show_message(f"⚠️ 提醒：{', '.join(safety_result.warnings)}", 'warning')
            
            # 2. 意图理解
            self._gui._update_status("🧠 理解您的意图...")
            intent = await self._llm.understand_intent(
                user_input=user_input,
                user_profile=self._user_profile,
            )
            self._current_intent = intent
            
            self._gui.show_system(f"📌 意图：{intent.normalized_text}")
            self._gui.show_system(f"🎯 目标应用：{intent.target_app or '未指定'}")
            
            if intent.confidence.is_low:
                self._gui.show_message("🤔 我不太确定您想做什么，能再说详细一点吗？", 'warning')
                self._gui.done_processing()
                return
            
            # 3. 截屏分析
            self._gui._update_status("👁️ 分析当前屏幕...")
            screenshot, original_size = await self._vision.capture_screen()
            
            if not screenshot:
                self._gui.show_message("❌ 截屏失败，请重试", 'error')
                self._gui.done_processing()
                return
            
            screen_state = await self._vision.analyze_screen_state(
                screenshot,
                user_intent=intent.normalized_text or user_input,
            )
            
            self._gui.show_system(f"📱 当前应用：{screen_state.app_name}")
            self._gui.show_system(f"📄 页面状态：{screen_state.screen_state}")
            
            screen_analysis = ScreenAnalysis(
                app_name=screen_state.app_name,
                screen_type=screen_state.screen_state,
                description=screen_state.description,
                suggested_actions=[screen_state.suggested_action] if screen_state.suggested_action else [],
                warnings=screen_state.warnings,
            )
            
            # 4. 任务规划
            self._gui._update_status("📋 生成任务计划...")
            plan = await self._planner.create_plan(
                intent=intent,
                screen_analysis=screen_analysis,
            )
            self._current_plan = plan
            
            if not plan.steps:
                self._gui.show_message("🤔 抱歉，我不太确定该怎么帮您完成这个操作。您能再说详细一点吗？", 'warning')
                self._gui.done_processing()
                return
            
            # 显示计划
            steps_text = "\n".join([f"  {i+1}. {s.friendly_instruction or s.description}" for i, s in enumerate(plan.steps)])
            self._gui.show_message(f"📋 我准备这样帮您操作：\n{steps_text}\n\n请按照提示操作，我会在旁边指导您。", 'agent')
            
            # 5. 询问确认后执行
            self._gui._update_status("⏳ 等待确认...")
            self._gui.ask_confirmation(
                "是否开始执行？",
                self._on_confirm_execute
            )
            
        except Exception as e:
            logger.error(f"处理输入时出错: {e}")
            import traceback
            traceback.print_exc()
            self._gui.show_message(f"❌ 抱歉，出了点问题：{e}", 'error')
            self._gui.done_processing()
    
    async def _on_confirm_execute(self, confirmed: bool):
        """确认执行回调"""
        if not confirmed:
            self._gui.show_message("⏹️ 已取消执行", 'system')
            self._gui.done_processing()
            return
        
        try:
            self._gui._update_status("⚡ 执行任务中...")
            self._gui.show_message("▶️ 开始执行，请按照提示操作...", 'agent')
            
            task = await self._executor.execute_task(
                self._current_intent, 
                plan=self._current_plan
            )
            
            if task.status == TaskStatus.COMPLETED:
                self._gui.show_message("🎉 太棒了！任务完成！", 'success')
                self._gui._update_status("✅ 任务完成")
            else:
                self._gui.show_message(f"⚠️ 任务未完成，状态：{task.status.value}\n如果遇到问题，请在下方反馈框告诉我。", 'warning')
                self._gui._update_status("⚠️ 任务未完成")
            
        except Exception as e:
            logger.error(f"执行任务时出错: {e}")
            self._gui.show_message(f"❌ 执行出错：{e}", 'error')
            self._gui._update_status("❌ 执行出错")
        
        finally:
            self._gui.done_processing()
    
    async def process_feedback(self, feedback: str):
        """处理用户反馈"""
        self._gui._update_status("🔄 处理反馈中...")
        
        try:
            # 使用LLM理解反馈内容
            response = await self._llm.generate_response(
                user_input=f"用户在操作过程中给出了反馈：'{feedback}'。请理解用户的意思，并给出简短的回应和建议。",
                context=f"当前任务：{self._current_intent.normalized_text if self._current_intent else '无'}",
            )
            
            self._gui.show_message(f"💡 {response.content}", 'agent')
            
            # 如果执行器正在运行，提交反馈
            if self._executor:
                self._executor.submit_user_feedback(feedback)
            
            self._gui._update_status("✅ 已收到反馈")
            
        except Exception as e:
            logger.error(f"处理反馈时出错: {e}")
            self._gui.show_message("收到您的反馈，我会尝试调整。", 'agent')


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        level="WARNING",
        format="<dim>{time:HH:mm:ss}</dim> | <level>{message}</level>",
    )
    
    gui = AgentGUI()
    gui.run()


if __name__ == "__main__":
    main()
