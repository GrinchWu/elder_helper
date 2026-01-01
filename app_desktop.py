"""
智能助手桌面应用 - 简洁版
集成语音输入/输出、任务执行功能
支持：需求录音、提问录音、重新开始流程
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
from src.knowledge.video_extractor import VideoKnowledgeExtractor
from src.models.action import ActionType
from loguru import logger


class SignalBridge(QObject):
    """Qt信号桥接器，用于线程间通信"""
    status_changed = pyqtSignal(str)
    message_received = pyqtSignal(str)
    recording_done = pyqtSignal(str, str)  # (text, input_type)
    processing_done = pyqtSignal()
    reset_done = pyqtSignal()


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
        self._video_extractor = None
        self._tts = None
        self._asr = None
        self._audio_capture = None
        self._user_profile = None
        self._current_plan = None
        self._current_intent = None
        self._idle_timeout = 30
        self._last_action_time = None
        self._idle_check_task = None
        self._is_recording = False  # 录音状态

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
        
        # 构建知识库（从B站搜索或使用预置数据）
        self._signals.status_changed.emit("构建知识库...")
        self._video_extractor = VideoKnowledgeExtractor()
        await self._video_extractor.initialize()
        
        try:
            # 使用带回退的构建方法（如果B站搜索失败则使用预置数据）
            kb_stats = await self._video_extractor.build_knowledge_base_with_fallback(self._rag)
            logger.info(f"知识库构建完成: {kb_stats}")
        except Exception as e:
            logger.warning(f"知识库构建失败，使用预置数据: {e}")
            await self._video_extractor._load_preset_knowledge(self._rag)
        
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
        
        self._last_action_time = asyncio.get_event_loop().time()
        self._idle_check_task = asyncio.create_task(self._check_idle())

    async def _check_idle(self):
        """检测用户是否长时间无动作"""
        while True:
            await asyncio.sleep(5)
            if self._last_action_time:
                elapsed = asyncio.get_event_loop().time() - self._last_action_time
                if elapsed >= self._idle_timeout:
                    await self._tts.speak("您好，需要我帮您做什么吗？")
                    self._last_action_time = asyncio.get_event_loop().time()
    
    def _reset_idle_timer(self):
        """重置空闲计时器"""
        self._last_action_time = asyncio.get_event_loop().time()

    async def close(self):
        """关闭服务"""
        if self._idle_check_task:
            self._idle_check_task.cancel()
        for svc in [self._asr, self._tts, self._llm, self._vision, 
                    self._planner, self._executor, self._embedding, self._video_extractor]:
            if svc:
                await svc.close()

    async def start_recording(self):
        """开始录音"""
        if self._is_recording:
            return
        self._is_recording = True
        self._reset_idle_timer()
        await self._tts.speak("开始录音，请说话")
        self._audio_capture = AudioCapture(sample_rate=config.asr.sample_rate)
        self._audio_capture.start()
        logger.info("录音已开始")

    async def stop_recording(self, input_type: str = "requirement"):
        """停止录音并识别"""
        if not self._is_recording:
            self._signals.recording_done.emit("", input_type)
            return
        
        self._is_recording = False
        try:
            if self._audio_capture:
                audio_data = self._audio_capture.get_all_audio()
                self._audio_capture.stop()
                self._audio_capture = None
                
                if audio_data:
                    self._signals.status_changed.emit("识别中...")
                    logger.info(f"音频数据大小: {len(audio_data)} bytes")
                    result = await self._asr.recognize_audio(audio_data)
                    text = result.text.strip() if result.text else ""
                    if text:
                        logger.info(f"识别结果: {text}")
                        await self._tts.speak(f"您说的是：{text}")
                    self._signals.recording_done.emit(text, input_type)
                else:
                    self._signals.recording_done.emit("", input_type)
        except Exception as e:
            logger.error(f"停止录音失败: {e}")
            self._signals.recording_done.emit("", input_type)

    async def reset_flow(self):
        """重新开始流程"""
        self._current_plan = None
        self._current_intent = None
        self._reset_idle_timer()
        await self._tts.speak("好的，我们重新开始，请告诉我您需要什么帮助")
        self._signals.reset_done.emit()
        logger.info("流程已重置")

    async def process_requirement(self, user_input: str):
        """处理用户需求（主流程）- 优化版：并行化 + 混合规划模式"""
        self._reset_idle_timer()
        try:
            self._signals.status_changed.emit("安全检查...")
            
            safety_result = self._safety.check_text_safety(user_input)
            if not safety_result.is_safe and safety_result.blocked_reason:
                await self._tts.speak(f"安全警告：{safety_result.blocked_reason}")
                self._signals.processing_done.emit()
                return
            
            # ========== 并行执行：意图理解 + 屏幕截图 + RAG搜索 ==========
            self._signals.status_changed.emit("分析中...")
            logger.info("=" * 50)
            logger.info("[并行处理] 开始并行执行：意图理解 + 屏幕截图 + RAG搜索")
            
            import time
            start_time = time.time()
            
            # 创建并行任务
            intent_task = asyncio.create_task(
                self._llm.understand_intent(user_input, self._user_profile)
            )
            screenshot_task = asyncio.create_task(
                self._vision.capture_screen()
            )
            rag_task = asyncio.create_task(
                self._rag.retrieve(user_input, top_k=3)
            )
            
            # 等待所有任务完成
            intent, (screenshot, original_size), rag_result = await asyncio.gather(
                intent_task, screenshot_task, rag_task,
                return_exceptions=True
            )
            
            parallel_time = time.time() - start_time
            logger.info(f"[并行处理] 完成，耗时: {parallel_time:.2f}s")
            
            # 处理可能的异常
            if isinstance(intent, Exception):
                logger.error(f"意图理解失败: {intent}")
                await self._tts.speak("抱歉，我没有理解您的意思")
                self._signals.processing_done.emit()
                return
            
            if isinstance(screenshot, Exception) or not screenshot:
                logger.error(f"截屏失败: {screenshot}")
                await self._tts.speak("截屏失败")
                self._signals.processing_done.emit()
                return
            
            self._current_intent = intent
            logger.info(f"[意图理解] 原始输入: {user_input}")
            logger.info(f"[意图理解] 规范化文本: {intent.normalized_text}")
            logger.info(f"[意图理解] 置信度: {intent.confidence}")
            
            if intent.confidence.is_low:
                await self._tts.speak("我不太确定您想做什么，能再说详细一点吗？")
                self._signals.processing_done.emit()
                return
            
            # RAG结果日志
            if not isinstance(rag_result, Exception) and (rag_result.guides or rag_result.nodes):
                logger.info(f"[RAG搜索] 找到 {len(rag_result.guides)} 条指南, {len(rag_result.nodes)} 个知识节点")
            else:
                logger.info("[RAG搜索] 未找到相关结果")
            
            # ========== 屏幕分析（需要intent结果）==========
            self._signals.status_changed.emit("分析屏幕...")
            screen_state = await self._vision.analyze_screen_state(
                screenshot, user_intent=intent.normalized_text or user_input
            )
            logger.info(f"[屏幕分析] 应用: {screen_state.app_name}")
            logger.info(f"[屏幕分析] 状态: {screen_state.screen_state}")
            logger.info("=" * 50)
            
            screen_analysis = ScreenAnalysis(
                app_name=screen_state.app_name,
                screen_type=screen_state.screen_state,
                description=screen_state.description,
                suggested_actions=[screen_state.suggested_action] if screen_state.suggested_action else [],
                warnings=screen_state.warnings,
            )
            
            # ========== 混合模式：ReAct循环执行 ==========
            self._signals.status_changed.emit("执行中...")
            await self._tts.speak("好的，我来帮您操作")
            
            # 使用ReAct循环模式
            await self._react_execution_loop(intent, screen_analysis, screenshot)
            
        except Exception as e:
            logger.error(f"处理出错: {e}")
            import traceback
            traceback.print_exc()
            await self._tts.speak_error(str(e))
            self._signals.status_changed.emit("出错")
        finally:
            self._reset_idle_timer()
            self._signals.processing_done.emit()
    
    async def _react_execution_loop(self, intent: Intent, screen_analysis: ScreenAnalysis, screenshot: bytes):
        """ReAct循环执行模式 - 观察->规划->执行->观察..."""
        max_steps = 10
        history = []
        current_screen = screen_analysis
        current_screenshot = screenshot
        
        for step_num in range(max_steps):
            self._reset_idle_timer()
            
            # 1. 快速规划下一步（使用Qwen3-14B，目标<3s）
            logger.info(f"[ReAct] 步骤 {step_num + 1}: 规划中...")
            import time
            plan_start = time.time()
            
            next_step = await self._planner.plan_next_step(
                intent=intent,
                screen_analysis=current_screen,
                history=history,
            )
            
            plan_time = time.time() - plan_start
            logger.info(f"[ReAct] 规划耗时: {plan_time:.2f}s")
            
            # 检查是否完成
            if next_step.action and next_step.action.action_type == ActionType.DONE:
                await self._tts.speak_success("任务完成！")
                self._signals.status_changed.emit("完成")
                return
            
            # 2. 播报当前步骤
            step_msg = next_step.friendly_instruction or next_step.description
            await self._tts.speak(f"第{step_num + 1}步，{step_msg}")
            self._signals.status_changed.emit(f"步骤 {step_num + 1}: {step_msg[:20]}...")
            
            # 3. 执行动作（这里等待用户操作或自动执行）
            # TODO: 实现自动执行，目前等待用户手动操作
            logger.info(f"[ReAct] 等待用户执行: {step_msg}")
            
            # 4. 动作后延迟0.5s再分析
            await asyncio.sleep(0.5)
            
            # 5. 观察新屏幕状态
            new_screenshot, _ = await self._vision.capture_screen()
            if new_screenshot:
                new_state = await self._vision.analyze_screen_state(
                    new_screenshot, 
                    user_intent=intent.normalized_text
                )
                current_screen = ScreenAnalysis(
                    app_name=new_state.app_name,
                    screen_type=new_state.screen_state,
                    description=new_state.description,
                )
                current_screenshot = new_screenshot
                logger.info(f"[ReAct] 新屏幕状态: {new_state.app_name} - {new_state.screen_state}")
            
            # 记录历史
            history.append(f"{step_num + 1}. {step_msg}")
            
            # 简单等待让用户有时间操作
            await asyncio.sleep(2)
        
        # 达到最大步骤
        await self._tts.speak("操作步骤较多，请告诉我是否需要继续")
        self._signals.status_changed.emit("等待确认")

    async def process_question(self, question: str):
        """处理用户提问（简单问答，不执行任务）"""
        self._reset_idle_timer()
        try:
            self._signals.status_changed.emit("思考中...")
            logger.info(f"[提问] 用户问题: {question}")
            
            # RAG搜索相关知识
            logger.info("=" * 50)
            logger.info("[RAG搜索] 搜索问题相关知识...")
            try:
                rag_result = await self._rag.retrieve(question, top_k=5)
                if rag_result.guides or rag_result.nodes:
                    logger.info(f"[RAG搜索] 找到 {len(rag_result.guides)} 条指南, {len(rag_result.nodes)} 个知识节点")
                    logger.info(f"[RAG搜索] 置信度: {rag_result.confidence:.3f}")
                    for i, guide in enumerate(rag_result.guides):
                        logger.info(f"  [指南{i+1}] {guide.title}")
                        logger.info(f"      应用: {guide.app_name}, 功能: {guide.feature_name}")
                        logger.info(f"      步骤: {' -> '.join(guide.steps[:3])}...")
                    for i, node in enumerate(rag_result.nodes):
                        logger.info(f"  [节点{i+1}] {node.name}")
                        logger.info(f"      描述: {node.description[:100]}...")
                    context = rag_result.context
                    if context:
                        logger.info(f"[RAG上下文]\n{context}")
                else:
                    logger.info("[RAG搜索] 未找到相关结果")
                    context = ""
            except Exception as e:
                logger.warning(f"[RAG搜索] 搜索失败: {e}")
                context = ""
            logger.info("=" * 50)
            
            # 使用LLM回答问题
            if context:
                prompt = f"根据以下知识回答用户问题：\n\n知识：{context}\n\n问题：{question}\n\n请用简洁易懂的语言回答："
            else:
                prompt = f"请用简洁易懂的语言回答以下问题：{question}"
            
            response = await self._llm.chat(prompt)
            logger.info(f"[回答] {response}")
            
            await self._tts.speak(response)
            self._signals.status_changed.emit("回答完成")
            
        except Exception as e:
            logger.error(f"回答问题出错: {e}")
            await self._tts.speak("抱歉，我无法回答这个问题")
            self._signals.status_changed.emit("出错")
        finally:
            self._signals.processing_done.emit()


class SimpleAssistantUI(QWidget):
    """简洁助手界面"""
    
    def __init__(self):
        super().__init__()
        self._drag_pos = QPoint()
        self._is_recording = False
        self._is_processing = False
        self._current_input_type = "requirement"  # requirement 或 question
        self._signals = SignalBridge()
        self._agent = None
        self._loop = None
        self._agent_thread = None
        
        self._signals.status_changed.connect(self._on_status_changed)
        self._signals.recording_done.connect(self._on_recording_done)
        self._signals.processing_done.connect(self._on_processing_done)
        self._signals.reset_done.connect(self._on_reset_done)
        
        self.initUI()
        self._start_agent()
    
    def initUI(self):
        """初始化界面"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(550, 70)
        
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 550) // 2, 20)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(8)
        
        # 需求录音按钮（开始/停止）
        self._req_btn = QPushButton("🎤需求")
        self._req_btn.setFixedSize(70, 50)
        self._req_btn.setCursor(Qt.PointingHandCursor)
        self._req_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFB75E, stop:1 #ED8F03);
                border: none; border-radius: 10px; color: white; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFC988, stop:1 #FF9D00); }
        """)
        self._req_btn.clicked.connect(self._on_req_click)
        layout.addWidget(self._req_btn)
        
        # 提问录音按钮
        self._ask_btn = QPushButton("❓提问")
        self._ask_btn.setFixedSize(70, 50)
        self._ask_btn.setCursor(Qt.PointingHandCursor)
        self._ask_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #667eea, stop:1 #764ba2);
                border: none; border-radius: 10px; color: white; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c94f4, stop:1 #8b5fbf); }
        """)
        self._ask_btn.clicked.connect(self._on_ask_click)
        layout.addWidget(self._ask_btn)
        
        # 输入框
        self._input = QLineEdit()
        self._input.setPlaceholderText("输入需求或问题...")
        self._input.setStyleSheet("""
            QLineEdit {
                background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.3);
                border-radius: 15px; color: white; font-size: 14px; padding: 8px 12px;
            }
        """)
        self._input.returnPressed.connect(self._on_send)
        layout.addWidget(self._input)
        
        # 发送按钮
        self._send_btn = QPushButton("➤")
        self._send_btn.setFixedSize(45, 45)
        self._send_btn.setCursor(Qt.PointingHandCursor)
        self._send_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #4ade80, stop:1 #22c55e);
                border: none; border-radius: 22px; color: white; font-size: 18px;
            }
            QPushButton:hover { background: #16a34a; }
        """)
        self._send_btn.clicked.connect(self._on_send)
        layout.addWidget(self._send_btn)
        
        # 重新开始按钮
        self._reset_btn = QPushButton("🔄")
        self._reset_btn.setFixedSize(45, 45)
        self._reset_btn.setCursor(Qt.PointingHandCursor)
        self._reset_btn.setToolTip("重新开始")
        self._reset_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f97316, stop:1 #ea580c);
                border: none; border-radius: 22px; color: white; font-size: 18px;
            }
            QPushButton:hover { background: #c2410c; }
        """)
        self._reset_btn.clicked.connect(self._on_reset_click)
        layout.addWidget(self._reset_btn)
        
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
        
        # 注意：在Windows上透明窗口使用阴影效果可能导致UpdateLayeredWindowIndirect错误
        # 如需阴影效果，可取消下面注释
        # shadow = QGraphicsDropShadowEffect()
        # shadow.setBlurRadius(20)
        # shadow.setColor(QColor(0, 0, 0, 100))
        # shadow.setOffset(0, 4)
        # self.setGraphicsEffect(shadow)

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

    def _on_req_click(self):
        """需求按钮点击 - 开始/停止录音"""
        if self._is_processing:
            return
        
        if not self._is_recording:
            # 开始录音
            self._is_recording = True
            self._current_input_type = "requirement"
            self._req_btn.setText("⏹停止")
            self._req_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF5F6D, stop:1 #FFC371);
                    border: none; border-radius: 10px; color: white; font-size: 14px; font-weight: bold;
                }
            """)
            self._ask_btn.setEnabled(False)
            self._input.setPlaceholderText("正在录音...点击停止结束")
            if self._agent and self._loop:
                asyncio.run_coroutine_threadsafe(self._agent.start_recording(), self._loop)
        else:
            # 停止录音
            if self._agent and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._agent.stop_recording("requirement"), self._loop
                )

    def _on_ask_click(self):
        """提问按钮点击 - 开始/停止录音"""
        if self._is_processing:
            return
        
        if not self._is_recording:
            # 开始录音
            self._is_recording = True
            self._current_input_type = "question"
            self._ask_btn.setText("⏹停止")
            self._ask_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FF5F6D, stop:1 #FFC371);
                    border: none; border-radius: 10px; color: white; font-size: 14px; font-weight: bold;
                }
            """)
            self._req_btn.setEnabled(False)
            self._input.setPlaceholderText("正在录音...点击停止结束")
            if self._agent and self._loop:
                asyncio.run_coroutine_threadsafe(self._agent.start_recording(), self._loop)
        else:
            # 停止录音
            if self._agent and self._loop:
                asyncio.run_coroutine_threadsafe(
                    self._agent.stop_recording("question"), self._loop
                )

    def _on_recording_done(self, text: str, input_type: str):
        """录音完成"""
        self._is_recording = False
        
        # 恢复按钮状态
        self._req_btn.setText("🎤需求")
        self._req_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFB75E, stop:1 #ED8F03);
                border: none; border-radius: 10px; color: white; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #FFC988, stop:1 #FF9D00); }
        """)
        self._req_btn.setEnabled(True)
        
        self._ask_btn.setText("❓提问")
        self._ask_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #667eea, stop:1 #764ba2);
                border: none; border-radius: 10px; color: white; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7c94f4, stop:1 #8b5fbf); }
        """)
        self._ask_btn.setEnabled(True)
        
        self._input.setPlaceholderText("输入需求或问题...")
        
        if text:
            self._input.setText(text)
            # 根据输入类型处理
            if input_type == "requirement":
                self._process_requirement()
            else:
                self._process_question()

    def _on_send(self):
        """发送按钮 - 默认作为需求处理"""
        self._process_requirement()

    def _process_requirement(self):
        """处理需求"""
        if self._is_processing:
            return
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._is_processing = True
        self._set_buttons_enabled(False)
        
        if self._agent and self._loop:
            asyncio.run_coroutine_threadsafe(self._agent.process_requirement(text), self._loop)

    def _process_question(self):
        """处理提问"""
        if self._is_processing:
            return
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self._is_processing = True
        self._set_buttons_enabled(False)
        
        if self._agent and self._loop:
            asyncio.run_coroutine_threadsafe(self._agent.process_question(text), self._loop)

    def _on_reset_click(self):
        """重新开始按钮点击"""
        if self._is_recording:
            return
        self._input.clear()
        if self._agent and self._loop:
            asyncio.run_coroutine_threadsafe(self._agent.reset_flow(), self._loop)

    def _on_reset_done(self):
        """重置完成"""
        self._is_processing = False
        self._set_buttons_enabled(True)
        self._input.setPlaceholderText("请说出您的需求...")

    def _set_buttons_enabled(self, enabled: bool):
        """设置按钮启用状态"""
        self._req_btn.setEnabled(enabled)
        self._ask_btn.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        self._input.setEnabled(enabled)

    def _on_status_changed(self, status: str):
        """状态变化"""
        self._input.setPlaceholderText(status)
    
    def _on_processing_done(self):
        """处理完成"""
        self._is_processing = False
        self._set_buttons_enabled(True)
        self._input.setPlaceholderText("输入需求或问题...")
    
    def closeEvent(self, event):
        """关闭"""
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        event.accept()


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 12))
    
    window = SimpleAssistantUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
