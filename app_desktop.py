"""
智能助手桌面应用 - 使用PyQt5
类似classland的桌面应用形式，保留顶部灵动岛和可移动窗口
"""
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QPoint, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QPainterPath, QBrush, QLinearGradient


class StatusBar(QWidget):
    """顶部固定状态栏（灵动岛样式）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 灵动岛：始终置顶且不抢焦点；独立于主窗口
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        
        # 设置窗口位置（屏幕顶部中央）
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(
            (screen.width() - 400) // 2,
            20,
            400,
            60
        )
        
        # 设置窗口属性
        self.setWindowTitle("StatusBar")
        # 不接受焦点，避免影响其他程序
        self.setFocusPolicy(Qt.NoFocus)
        
        self.initUI()
    
    def initUI(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(12)
        
        # 图标
        icon_label = QLabel("🎤")
        icon_label.setFont(QFont("Arial", 20))
        layout.addWidget(icon_label)
        
        # 文本区域
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        
        self.title_label = QLabel("等待您的指令...")
        self.title_label.setStyleSheet("""
            color: white;
            font-size: 16px;
            font-weight: 600;
        """)
        
        self.subtitle_label = QLabel("请说出您想要做的事情")
        self.subtitle_label.setStyleSheet("""
            color: #aaa;
            font-size: 12px;
        """)
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.subtitle_label)
        layout.addWidget(text_widget)
        
        # 状态指示器
        self.indicator = QLabel("●")
        self.indicator.setStyleSheet("""
            color: #4ade80;
            font-size: 12px;
        """)
        layout.addWidget(self.indicator)
        
        # 设置背景样式
        self.setStyleSheet("""
            QWidget {
                background: rgba(30, 30, 30, 230);
                border-radius: 30px;
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
    
    def paintEvent(self, event):
        """绘制圆角背景"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 30, 30)
        
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(30, 30, 30, 240))
        gradient.setColorAt(1, QColor(30, 30, 30, 230))
        
        painter.fillPath(path, QBrush(gradient))
        painter.setPen(QColor(255, 255, 255, 25))
        painter.drawPath(path)
    
    def updateStatus(self, title, subtitle, color="#4ade80"):
        """更新状态"""
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.indicator.setStyleSheet(f"""
            color: {color};
            font-size: 12px;
        """)


class DraggableWindow(QMainWindow):
    """可拖拽的主窗口"""
    def __init__(self):
        super().__init__()
        self.drag_position = QPoint()
        self.initUI()
    
    def initUI(self):
        # 使用标准窗口标志，保持与其他应用同层级，避免置顶和焦点问题
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("智能助手")
        
        # 设置窗口大小和位置
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(
            (screen.width() - 800) // 2,
            100,
            800,
            600
        )
        
        # 设置窗口背景色
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # 创建中央部件
        central_widget = QWidget()
        central_widget.setObjectName("mainWindow")
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 窗口标题栏
        self.createHeader(main_layout)
        
        # 内容区域
        self.createContent(main_layout)
        
        # 设置样式
        self.setStyleSheet("""
            QWidget#mainWindow {
                background: #1e1e1e;
                border-radius: 16px;
            }
        """)
        
        # 添加阴影
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 10)
        central_widget.setGraphicsEffect(shadow)
    
    def createHeader(self, parent_layout):
        """创建标题栏"""
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("""
            QFrame {
                background: #2a2a2a;
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
            }
        """)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # 标题
        title_layout = QHBoxLayout()
        title_icon = QLabel("📺")
        title_icon.setFont(QFont("Arial", 16))
        self.title_text = QLabel("操作指导")
        self.title_text.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 600;
        """)
        title_layout.addWidget(title_icon)
        title_layout.addWidget(self.title_text)
        title_layout.setSpacing(10)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # 控制按钮
        minimize_btn = QPushButton("−")
        minimize_btn.setFixedSize(32, 32)
        minimize_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 6px;
                color: white;
                font-size: 18px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
        """)
        # 最小化按钮：切换最小化 / 还原
        minimize_btn.clicked.connect(self.toggleMinimize)
        
        close_btn = QPushButton("×")
        close_btn.setFixedSize(32, 32)
        close_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 6px;
                color: white;
                font-size: 18px;
            }
            QPushButton:hover {
                background: #ef4444;
            }
        """)
        close_btn.clicked.connect(self.close)
        
        header_layout.addWidget(minimize_btn)
        header_layout.addWidget(close_btn)
        
        parent_layout.addWidget(header)
    
    def createContent(self, parent_layout):
        """创建内容区域"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: #1e1e1e;
            }
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 8px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555;
            }
        """)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(24)
        
        # 任务卡片
        self.createTaskCard(content_layout)
        
        # 操作步骤
        self.createSteps(content_layout)
        
        # 视频区域
        self.createVideoSection(content_layout)
        
        # 反馈区域
        self.createFeedbackSection(content_layout)
        
        # 语音按钮
        self.createVoiceButton(content_layout)
        
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        parent_layout.addWidget(scroll)
    
    def createTaskCard(self, parent_layout):
        """创建任务卡片"""
        section_title = QLabel("当前任务")
        section_title.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: 600;
        """)
        parent_layout.addWidget(section_title)
        
        task_card = QFrame()
        task_card.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 12px;
                padding: 20px;
            }
        """)
        
        task_layout = QHBoxLayout(task_card)
        task_layout.setContentsMargins(20, 20, 20, 20)
        task_layout.setSpacing(16)
        
        task_icon = QLabel("🎯")
        task_icon.setFont(QFont("Arial", 24))
        task_layout.addWidget(task_icon)
        
        task_info = QVBoxLayout()
        self.task_name = QLabel("准备就绪")
        self.task_name.setStyleSheet("""
            color: white;
            font-size: 18px;
            font-weight: 600;
        """)
        self.task_desc = QLabel("等待您的语音指令...")
        self.task_desc.setStyleSheet("""
            color: rgba(255, 255, 255, 0.9);
            font-size: 14px;
        """)
        task_info.addWidget(self.task_name)
        task_info.addWidget(self.task_desc)
        task_layout.addLayout(task_info)
        
        parent_layout.addWidget(task_card)
    
    def createSteps(self, parent_layout):
        """创建操作步骤"""
        section_title = QLabel("操作步骤")
        section_title.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: 600;
        """)
        parent_layout.addWidget(section_title)
        
        self.steps_layout = QVBoxLayout()
        self.steps_layout.setSpacing(12)
        
        # 示例步骤
        steps = [
            ("等待指令", "请说出您想要做的事情", "pending"),
        ]
        
        for i, (title, desc, status) in enumerate(steps, 1):
            self.addStep(i, title, desc, status)
        
        steps_widget = QWidget()
        steps_widget.setLayout(self.steps_layout)
        parent_layout.addWidget(steps_widget)
    
    def addStep(self, num, title, desc, status="pending"):
        """添加步骤"""
        step_frame = QFrame()
        
        if status == "active":
            step_frame.setStyleSheet("""
                QFrame {
                    background: #2a3a4a;
                    border-left: 4px solid #3b82f6;
                    border-radius: 12px;
                }
            """)
        elif status == "completed":
            step_frame.setStyleSheet("""
                QFrame {
                    background: #2a2a2a;
                    border-left: 4px solid #4ade80;
                    border-radius: 12px;
                    opacity: 0.7;
                }
            """)
        else:
            step_frame.setStyleSheet("""
                QFrame {
                    background: #2a2a2a;
                    border-left: 4px solid #444;
                    border-radius: 12px;
                }
            """)
        
        step_layout = QHBoxLayout(step_frame)
        step_layout.setContentsMargins(16, 16, 16, 16)
        step_layout.setSpacing(16)
        
        # 步骤编号
        step_num = QLabel(str(num))
        step_num.setFixedSize(32, 32)
        step_num.setAlignment(Qt.AlignCenter)
        if status == "active":
            step_num.setStyleSheet("""
                background: #3b82f6;
                border-radius: 16px;
                color: white;
                font-weight: 600;
            """)
        elif status == "completed":
            step_num.setStyleSheet("""
                background: #4ade80;
                border-radius: 16px;
                color: white;
                font-weight: 600;
            """)
        else:
            step_num.setStyleSheet("""
                background: #444;
                border-radius: 16px;
                color: white;
                font-weight: 600;
            """)
        
        # 步骤内容
        step_content = QVBoxLayout()
        step_title = QLabel(title)
        step_title.setStyleSheet("""
            color: white;
            font-size: 16px;
            font-weight: 600;
        """)
        step_desc = QLabel(desc)
        step_desc.setStyleSheet("""
            color: #aaa;
            font-size: 14px;
        """)
        step_desc.setWordWrap(True)
        step_content.addWidget(step_title)
        step_content.addWidget(step_desc)
        
        step_layout.addWidget(step_num)
        step_layout.addLayout(step_content)
        
        self.steps_layout.addWidget(step_frame)
    
    def createVideoSection(self, parent_layout):
        """创建视频区域"""
        self.video_section = QWidget()
        video_layout = QVBoxLayout(self.video_section)
        video_layout.setContentsMargins(0, 0, 0, 0)
        
        section_title = QLabel("相关视频教程")
        section_title.setStyleSheet("""
            color: white;
            font-size: 20px;
            font-weight: 600;
        """)
        video_layout.addWidget(section_title)
        
        self.video_section.setVisible(False)
        parent_layout.addWidget(self.video_section)
    
    def createFeedbackSection(self, parent_layout):
        """创建反馈区域"""
        self.feedback_section = QFrame()
        self.feedback_section.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #4ade80, stop:1 #22c55e);
                border-radius: 12px;
                padding: 20px;
            }
        """)
        self.feedback_section.setVisible(False)
        
        feedback_layout = QHBoxLayout(self.feedback_section)
        feedback_layout.setContentsMargins(20, 20, 20, 20)
        
        feedback_icon = QLabel("👏")
        feedback_icon.setFont(QFont("Arial", 24))
        self.feedback_text = QLabel("您做得很好！继续加油！")
        self.feedback_text.setStyleSheet("""
            color: white;
            font-size: 16px;
            font-weight: 600;
        """)
        
        feedback_layout.addWidget(feedback_icon)
        feedback_layout.addWidget(self.feedback_text)
        
        parent_layout.addWidget(self.feedback_section)
    
    def createVoiceButton(self, parent_layout):
        """创建语音按钮"""
        voice_widget = QWidget()
        voice_layout = QVBoxLayout(voice_widget)
        voice_layout.setAlignment(Qt.AlignCenter)
        
        voice_btn = QPushButton("🎤\n点击开始语音输入")
        voice_btn.setFixedSize(200, 200)
        voice_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #667eea, stop:1 #764ba2);
                border: none;
                border-radius: 100px;
                color: white;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #7c8ef0, stop:1 #8a5fb8);
            }
        """)
        
        voice_layout.addWidget(voice_btn)
        parent_layout.addWidget(voice_widget)
    
    def toggleMinimize(self):
        """单击即最小化"""
        self.showMinimized()

    def mousePressEvent(self, event):
        """鼠标按下事件 - 用于拖拽"""
        if event.button() == Qt.LeftButton:
            # 检查是否点击在标题栏区域
            if event.y() < 50:  # 标题栏高度
                self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
                event.accept()
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """鼠标移动事件 - 实现拖拽"""
        if event.buttons() == Qt.LeftButton and hasattr(self, 'drag_position'):
            self.move(event.globalPos() - self.drag_position)
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def paintEvent(self, event):
        """绘制圆角窗口"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 16, 16)
        painter.fillPath(path, QBrush(QColor(30, 30, 30)))
    
    def closeEvent(self, event):
        """窗口关闭事件 - 通知主应用关闭状态栏"""
        # 通过应用程序对象找到MainApp并关闭状态栏
        app = QApplication.instance()
        if hasattr(app, 'main_app') and app.main_app:
            app.main_app.closeStatusBar()
        event.accept()


class MainApp:
    """主应用类"""
    def __init__(self):
        self.app = QApplication(sys.argv)
        # 将MainApp实例保存到app对象中，方便窗口访问
        self.app.main_app = self
        
        # 先创建主窗口（普通窗口）
        self.main_window = DraggableWindow()
        
        # 再创建状态栏（灵动岛），确保它独立存在
        self.status_bar = StatusBar()
        
        # 显示窗口 - 先显示主窗口，再显示状态栏
        self.main_window.show()
        self.main_window.raise_()
        
        # 显示状态栏
        self.status_bar.show()
        self.status_bar.raise_()
        # 启动定时器，确保灵动岛始终置顶且不抢焦点
        self.timer = QTimer()
        self.timer.timeout.connect(self.keepStatusBarOnTop)
        self.timer.start(500)  # 每0.5秒检查一次
    
    def keepStatusBarOnTop(self):
        """保持状态栏在最上层（定时执行，不抢焦点）"""
        if self.status_bar and self.status_bar.isVisible():
            self.status_bar.raise_()
    
    def closeStatusBar(self):
        """关闭状态栏"""
        if self.timer:
            self.timer.stop()
        if self.status_bar:
            self.status_bar.close()
    
    def run(self):
        """运行应用"""
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    app = MainApp()
    app.run()

