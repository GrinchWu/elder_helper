"""
智能助手桌面应用 - 简洁版
集成语音输入/输出、任务执行功能
"""
import sys
import asyncio
import threading
from PyQt5.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                             QLabel, QPushButton, QLineEdit, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPainter, QPainterPath, QBrush, QLinearGradient

from src.config import config
from src.models.intent import Intent
from src.models.task import Task, TaskStatus, TaskPlan
from src.models.session import UserProfile
from src.models.knowledge import KnowledgeGraph
from src.services.llm_service import LLMService
from src.services.vision_service import VisionService, VLConfig, ScreenAnalysis
from src.services.planner_service import PlannerService
from src.services.safety_service import SafetyService
from src.services.executor_service import ExecutorService
from src.services.embedding_service import EmbeddingService
from src.services.tts_service import TTSService
from src.services.asr_service import ASRService, ASRConfig, AudioCapture
from src.knowledge.rag_service import RAGService
from loguru import logger


class SignalBridge(QObject):
    """Qt信号桥接器，用于线程间通信"""
    status_changed = pyqtSignal(str)
    message_received = pyqtSignal(str)
    recording_done = pyqtSignal(str)
    processing_done = pyqtSignal()


class ElderlyAgent:
    """老年人助手Agent"""
    
    def __init__(self, signals: SignalBridge):
        self._signals = signals
        self._llm = None
        self._vision = None
        self._planner = None
        self._safety = None
        self._executor = None
        self._embedding = None
        self._rag = None
        self._knowledge_graph = None
        self._tts = None
        self._asr = None
        self._audio_capture = None
        self._user_profile = None
        self._current_plan = None
        self._current_intent = None
        self._idle_timeout = 30  # 无动作超时秒数
        self._last_action_time = None
        self._idle_check_task = None

    async def initialize(self):
        """初始化所有服务"""
        self._signals.status_changed.emit("初始化语音服务...")
        self._tts = TTSService()
        await self._tts.initialize()
        
        self._signals.status_changed.emit("初始化语音识别...")
        asr_config = ASRConfig(
            project_id=config.asr.project_id,
            easyllm_id=config.asr.easyllm_id,
            api_key=config.asr.api_key,
        )
        self._asr = ASRService(asr_config)
        await self._asr.initialize()
        
        self._signals.status_changed.emit("初始化意图理解...")
        self._llm = LLMService()
        await self._llm.initialize()
        
        self._signals.status_changed.emit("初始化视觉服务...")
        vl_config = VLConfig(
            api_key=config.api.api_key,
            model_light=config.api.vl_model_light,
            model_heavy=config.api.vl_model_heavy,
        )
        self._vision = VisionService(vl_config)
        await self._vision.initialize()
        
        self._signals.status_changed.emit("初始化规划服务...")
        self._planner = PlannerService()
        await self._planner.initialize()
        
        self._safety = SafetyService()
        
        self._signals.status_changed.emit("初始化知识服务...")
        self._embedding = EmbeddingService()
        await self._embedding.initialize()
        
        self._knowledge_graph = KnowledgeGraph()
        self._rag = RAGService()
        await self._rag.initialize(
            embedding_service=self._embedding,
            knowledge_graph=self._knowledge_graph,
        )
        self._planner.set_rag_service(self._rag)
        
        self._signals.status_changed.emit("初始化执行服务...")
        self._executor = ExecutorService()
        self._executor.set_vision_service(self._vision)
        self._executor.set_planner_service(self._planner)
        await self._executor.initialize()
        
        self._user_profile = UserProfile(
            name="用户",
            family_mapping={"老二": "张小明", "闺女": "张小红"},
            frequent_contacts=["张小明", "张小红"],
        )
        
        self._signals.status_changed.emit("准备就绪")
        await self._tts.speak_welcome()
        
        # 启动空闲检测
        self._last_action_time = asyncio.get_event_loop().time()
        self._idle_check_task = asyncio.create_task(self._check_idle())

    async def _check_idle(self):
        """检测用户是否长时间无动作"""
        while True:
            await asyncio.sleep(5)  # 每5秒检查一次
            if self._last_action_time:
                elapsed = asyncio.get_event_loop().time() - self._last_action_time
                if elapsed >= self._idle_timeout:
                    await self._tts.speak("您好，需要我帮您做什么吗？")
                    self._last_action_time = asyncio.get_event_loop().time()  # 重置计时
    
    def _reset_idle_timer(self):
        """重置空闲计时器"""
        self._last_action_time = asyncio.get_event_loop().time()

    async def close(self):
        """关闭服务"""
        if self._idle_check_task:
            self._idle_check_task.cancel()
        for svc in [self._asr, self._tts, self._llm, self._vision, 
                    self._planner, self._executor, self._embedding]:
            if svc:
                await svc.close()


    async def voice_input(self, duration: float = 5.0):
        """语音输入"""
        self._reset_idle_timer()
        try:
            await self._tts.speak("请说话")
            self._audio_capture = AudioCapture(sample_rate=config.asr.sample_rate)
            self._audio_capture.start()
            
            audio_data = b""
            start_time = asyncio.get_event_loop().time()
            async for chunk in self._audio_capture.get_audio_stream():
                audio_data += chunk
                if asyncio.get_event_loop().time() - start_time >= duration:
                    break
            
            self._audio_capture.stop()
            
            if audio_data:
                self._signals.status_changed.emit("识别中...")
                result = await self._asr.recognize_audio(audio_data)
                text = result.text.strip() if result.text else ""
                if text:
                    await self._tts.speak(f"您说的是：{text}")
                self._signals.recording_done.emit(text)
            else:
                self._signals.recording_done.emit("")
        except Exception as e:
            logger.error(f"语音输入失败: {e}")
            self._signals.recording_done.emit("")

    async def process_input(self, user_input: str):
        """处理用户输入"""
        self._reset_idle_timer()
        try:
            self._signals.status_changed.emit("安全检查...")
            
            safety_result = self._safety.check_text_safety(user_input)
            if not safety_result.is_safe and safety_result.blocked_reason:
                await self._tts.speak(f"安全警告：{safety_result.blocked_reason}")
                self._signals.processing_done.emit()
                return
            
            self._signals.status_changed.emit("理解意图...")
            intent = await self._llm.understand_intent(user_input, self._user_profile)
            self._current_intent = intent
            
            if intent.confidence.is_low:
                await self._tts.speak("我不太确定您想做什么，能再说详细一点吗？")
                self._signals.processing_done.emit()
                return
            
            self._signals.status_changed.emit("分析屏幕...")
            screenshot, original_size = await self._vision.capture_screen()
            
            if not screenshot:
                await self._tts.speak("截屏失败")
                self._signals.processing_done.emit()
                return
            
            screen_state = await self._vision.analyze_screen_state(
                screenshot, user_intent=intent.normalized_text or user_input
            )
            
            screen_analysis = ScreenAnalysis(
                app_name=screen_state.app_name,
                screen_type=screen_state.screen_state,
                description=screen_state.description,
                suggested_actions=[screen_state.suggested_action] if screen_state.suggested_action else [],
                warnings=screen_state.warnings,
            )
            
            self._signals.status_changed.emit("生成计划...")
            plan = await self._planner.create_plan(intent=intent, screen_analysis=screen_analysis)
            self._current_plan = plan
            
            if not plan.steps:
                await self._tts.speak("抱歉，我不确定该怎么帮您")
                self._signals.processing_done.emit()
                return
            
            from src.models.action import ActionType
            if len(plan.steps) == 1 and plan.steps[0].action and plan.steps[0].action.action_type == ActionType.DONE:
                msg = plan.steps[0].friendly_instruction or "任务已完成"
                await self._tts.speak_success(msg)
                self._signals.status_changed.emit("完成")
                self._signals.processing_done.emit()
                return
            
            # 不播报整体计划，直接开始执行
            self._signals.status_changed.emit("执行中...")
            await self._tts.speak("好的，我来帮您操作")
            
            # 逐步执行并播报每一步
            total_steps = len(plan.steps)
            for i, step in enumerate(plan.steps):
                self._reset_idle_timer()
                step_msg = step.friendly_instruction or step.description
                # 播报当前步骤
                await self._tts.speak(f"第{i+1}步，{step_msg}")
                self._signals.status_changed.emit(f"步骤 {i+1}/{total_steps}")
            
            task = await self._executor.execute_task(self._current_intent, plan=self._current_plan)
            
            if task.status == TaskStatus.COMPLETED:
                await self._tts.speak_success("任务完成！")
                self._signals.status_changed.emit("完成")
            else:
                await self._tts.speak("任务未完成，您可以告诉我遇到了什么问题")
                self._signals.status_changed.emit("未完成")
            
        except Exception as e:
            logger.error(f"处理出错: {e}")
            await self._tts.speak_error(str(e))
            self._signals.status_changed.emit("出错")
        finally:
            self._reset_idle_timer()
            self._signals.processing_done.emit()


class SimpleAssistantUI(QWidget):
    """简洁助手界面"""
    
    def __init__(self):
        super().__init__()
        self._drag_pos = QPoint()
        self._is_recording = False
        self._is_processing = False
        self._signals = SignalBridge()
        self._agent = None
        self._loop = None
        self._agent_thread = None
        
        self._signals.status_changed.connect(self._on_status_changed)
        self._signals.recording_done.connect(self._on_recording_done)
        self._signals.processing_done.connect(self._on_processing_done)
        
        self.initUI()
        self._start_agent()
    
    def initUI(self):
        """初始化界面"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(400, 70)
        
        # 屏幕顶部居中
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 400) // 2, 20)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(10)
        
        # 语音按钮
        self._voice_btn = QPushButton("🎤")
        self._voice_btn.setFixedSize(50, 50)
        self._voice_btn.setCursor(Qt.PointingHandCursor)
        self._voice_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFB75E, stop:1 #ED8F03);
                border: none; border-radius: 25px; color: white; font-size: 24px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFC988, stop:1 #FF9D00); }
        """)
        self._voice_btn.clicked.connect(self._on_voice_click)
        layout.addWidget(self._voice_btn)
        
        # 输入框
        self._input = QLineEdit()
        self._input.setPlaceholderText("输入或点击麦克风说话...")
        self._input.setStyleSheet("""
            QLineEdit {
                background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
                border-radius: 20px; color: white; font-size: 16px; padding: 10px 15px;
            }
        """)
        self._input.returnPressed.connect(self._on_send)
        layout.addWidget(self._input)
        
        # 发送按钮
        self._send_btn = QPushButton("➤")
        self._send_btn.setFixedSize(50, 50)
        self._send_btn.setCursor(Qt.PointingHandCursor)
        self._send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4ade80, stop:1 #22c55e);
                border: none; border-radius: 25px; color: white; font-size: 20px;
            }
            QPushButton:hover { background: #16a34a; }
        """)
        self._send_btn.clicked.connect(self._on_send)
        layout.addWidget(self._send_btn)
        
        # 关闭按钮
        close_btn = QPushButton("×")
        close_btn.setFixedSize(30, 30)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.1); border: none; border-radius: 15px;
                color: white; font-size: 18px;
            }
            QPushButton:hover { background: #ef4444; }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        # 阴影
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
    
    def paintEvent(self, event):
        """绘制背景"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 35, 35)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#164270"))
        gradient.setColorAt(1, QColor("#24548C"))
        painter.fillPath(path, QBrush(gradient))
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)

    
    def _start_agent(self):
        """启动Agent线程"""
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._agent = ElderlyAgent(self._signals)
            try:
                self._loop.run_until_complete(self._agent.initialize())
                self._loop.run_forever()
            except Exception as e:
                logger.error(f"Agent错误: {e}")
            finally:
                if self._agent:
                    self._loop.run_until_complete(self._agent.close())
                self._loop.close()
        
        self._agent_thread = threading.Thread(target=run, daemon=True)
        self._agent_thread.start()
    
    def _on_voice_click(self):
        """语音按钮点击"""
        if self._is_processing or self._is_recording:
            return
        self._is_recording = True
        self._voice_btn.setText("🔴")
        self._voice_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF5F6D, stop:1 #FFC371);
                border: none; border-radius: 25px; color: white; font-size: 24px;
            }
        """)
        self._input.setPlaceholderText("正在录音...")
        
        if self._agent and self._loop:
            asyncio.run_coroutine_threadsafe(self._agent.voice_input(), self._loop)
    
    def _on_recording_done(self, text: str):
        """录音完成"""
        self._is_recording = False
        self._voice_btn.setText("🎤")
        self._voice_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFB75E, stop:1 #ED8F03);
                border: none; border-radius: 25px; color: white; font-size: 24px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFC988, stop:1 #FF9D00); }
        """)
        self._input.setPlaceholderText("输入或点击麦克风说话...")
        
        if text:
            self._input.setText(text)
            self._on_send()
    
    def _on_send(self):
        """发送"""
        if self._is_processing:
            return
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._is_processing = True
        self._input.setEnabled(False)
        self._send_btn.setEnabled(False)
        
        if self._agent and self._loop:
            asyncio.run_coroutine_threadsafe(self._agent.process_input(text), self._loop)
    
    def _on_status_changed(self, status: str):
        """状态变化"""
        self._input.setPlaceholderText(status)
    
    def _on_processing_done(self):
        """处理完成"""
        self._is_processing = False
        self._input.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._input.setPlaceholderText("输入或点击麦克风说话...")
    
    def closeEvent(self, event):
        """关闭"""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        event.accept()


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<dim>{time:HH:mm:ss}</dim> | <level>{message}</level>")
    
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 12))
    
    window = SimpleAssistantUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
