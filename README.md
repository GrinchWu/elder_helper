# 老年人电脑助手 Agent

帮助不会使用电子产品的中老年人使用电脑的AI助手。

## 核心特性

### 1. SimToM心智理论意图理解
- 基于论文 "Think Twice: Perspective-Taking Improves Large Language Models' Theory-of-Mind Capabilities"
- 两阶段框架：视角转换 (Perspective-Taking) + 意图推理 (Question-Answering)
- BDI模型分析：Beliefs（信念）、Desires（愿望）、Intentions（意图）
- 理解老年人的模糊表述（如"手机吃钱"="流量超标"）

### 2. 语义到操作映射 (Semantic-to-Procedural Mapping)
- 将生活化语言转换为具体操作步骤
- 支持家庭成员称呼映射（如"老二"="张三"）
- 自动检测系统已安装应用，智能推荐目标应用

### 3. 两层视觉架构
- **第一层（轻量级）**：使用 `Qwen2.5-VL-72B-Instruct` 进行页面状态分析
  - 识别当前应用和页面状态
  - 列出可用元素（不含坐标）
  - 快速响应，适合频繁调用
- **第二层（重量级）**：使用 `Qwen3-VL-235B-A22B-Instruct` 进行精确元素定位
  - 返回元素的精确 bbox 坐标
  - 仅在需要点击操作时调用

### 4. 智能任务规划 (ReAct模式)
- 自动分解复杂任务为简单步骤
- 每一步给予清晰的指导
- 错误时自动重新规划，提供恢复方案
- 支持用户反馈实时调整

### 5. 智能步骤完成判断
- 基于用户输入事件（鼠标/键盘）触发验证
- 区分用户操作与页面动态效果（广告、动画）
- 30秒无操作超时提醒
- 自动检测整体任务目标是否达成

### 6. 知识库与RAG
- 从短视频平台提取操作知识
- 使用思维导图方式压缩上下文
- 语义搜索相关操作指南
- 老年人语言映射（如"那个绿色的"="微信"）

### 7. GUI交互界面
- 简洁的 tkinter 界面，始终置顶
- 对话记录显示
- 快捷按钮（打开微信、打开浏览器、关闭弹窗）
- 实时反馈输入框

### 8. 安全防护
- 诈骗信息检测与警告
- 敏感操作确认机制
- 隐私数据保护

## 技术栈

| 组件 | 技术 |
|------|------|
| 页面状态分析 | Qwen2.5-VL-72B-Instruct (Sophnet API) |
| 精确元素定位 | Qwen3-VL-235B-A22B-Instruct (Sophnet API) |
| 语言理解 | Qwen2.5-72B-Instruct (Sophnet API) |
| 意图理解 | SimToM + BDI模型 |
| 语音识别 | Sophnet WebSocket ASR |
| 语音合成 | CosyVoice |
| 向量模型 | BGE-M3 |
| 后端框架 | FastAPI |
| GUI框架 | tkinter |
| 输入监听 | pynput |

## 项目结构

```
elderly-assistant-agent/
├── src/
│   ├── agent/              # Agent核心
│   │   ├── elderly_agent.py    # 主Agent类
│   │   └── executor.py         # 动作执行器
│   ├── api/                # API接口
│   │   └── routes.py           # FastAPI路由
│   ├── models/             # 数据模型
│   │   ├── intent.py           # 意图模型
│   │   ├── action.py           # 动作模型
│   │   ├── task.py             # 任务模型
│   │   ├── knowledge.py        # 知识图谱模型
│   │   └── session.py          # 会话模型
│   ├── services/           # 服务层
│   │   ├── asr_service.py      # 语音识别
│   │   ├── tts_service.py      # 语音合成
│   │   ├── vision_service.py   # 视觉服务（两层架构）
│   │   ├── llm_service.py      # 大模型服务
│   │   ├── planner_service.py  # 任务规划（ReAct模式）
│   │   ├── executor_service.py # 任务执行服务
│   │   ├── safety_service.py   # 安全服务
│   │   └── embedding_service.py # 向量嵌入
│   ├── knowledge/          # 知识库
│   │   ├── video_extractor.py  # 视频知识提取
│   │   └── rag_service.py      # RAG服务
│   ├── utils/              # 工具类
│   │   ├── rate_limiter.py     # 速率限制
│   │   └── validators.py       # 数据验证
│   ├── config.py           # 配置管理
│   ├── gui_agent.py        # GUI界面入口
│   └── main.py             # API服务入口
├── tests/                  # 测试
├── pyproject.toml          # 项目配置
└── .env.example            # 环境变量示例
```

## 快速开始

### 1. 安装依赖

```bash
cd elderly-assistant-agent
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Windows下安装额外依赖
pip install pynput mss pillow
conda install pyaudio  # 用于麦克风采集（可选）
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，配置API地址
```

环境变量说明：
```bash
# Sophnet API配置 (OpenAI兼容格式)
SOPHNET_API_KEY=your_api_key_here
LLM_MODEL=Qwen2.5-72B-Instruct
VL_MODEL_LIGHT=Qwen2.5-VL-72B-Instruct
VL_MODEL_HEAVY=Qwen3-VL-235B-A22B-Instruct

# ASR配置 (WebSocket) - 可选
ASR_PROJECT_ID=your_project_id
ASR_EASYLLM_ID=your_easyllm_id
```

### 3. 启动GUI界面（推荐）

```bash
cd elderly-assistant-agent
python -m src.gui_agent
```

GUI界面功能：
- 输入框：输入您想做的事情
- 反馈框：操作过程中可以随时反馈
- 快捷按钮：常用操作一键触发
- 对话记录：查看历史交互

### 4. 启动API服务（可选）

```bash
python -m src.main
```

### 5. API使用

```bash
# 文本输入
curl -X POST http://localhost:8080/api/input/text \
  -H "Content-Type: application/json" \
  -d '{"text": "我想给女儿打个电话"}'

# 获取会话状态
curl http://localhost:8080/api/session/state
```

## 使用示例

### GUI界面使用

1. 启动程序后，会出现一个小窗口（始终置顶）
2. 在输入框输入您想做的事情，如：
   - "帮我打开微信"
   - "我想看看新闻"
   - "屏幕上有个东西关不掉"
3. 点击"发送"或按回车
4. 系统会分析屏幕并生成操作计划
5. 确认后，按照提示一步步操作
6. 如果操作不对，在反馈框告诉助手

### 支持的操作类型

- 打开应用（微信、浏览器、QQ等）
- 发送消息
- 关闭弹窗/广告
- 查看信息
- 基本的电脑操作

## Sophnet API配置

本项目使用Sophnet API（OpenAI兼容格式）：

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://www.sophnet.com/api/open-apis/v1"
)

response = client.chat.completions.create(
    model="Qwen2.5-72B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个帮助老年人使用电脑的助手"},
        {"role": "user", "content": "我想给老同事发个信"},
    ]
)
```

支持的模型：
- 纯文本：`Qwen2.5-72B-Instruct`, `Qwen2.5-32B-Instruct`, `DeepSeek-v3`
- 多模态（轻量）：`Qwen2.5-VL-72B-Instruct`
- 多模态（重量）：`Qwen3-VL-235B-A22B-Instruct`

## API接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/input/text` | POST | 文本输入 |
| `/api/input/audio` | POST | 语音输入 |
| `/api/user/profile` | POST | 设置用户画像 |
| `/api/session/state` | GET | 获取会话状态 |
| `/ws` | WebSocket | 实时通信 |

## 设计原则

### 无过错假设 (No-Fault Assumption)
任何错误都是系统的责任，不是用户的错。错误提示使用温和的语言，避免让用户产生焦虑。

### 老年人友好
- 使用简单的日常用语
- 避免技术术语
- 每一步都给予鼓励
- 支持模糊表述

### 安全第一
- 检测诈骗信息
- 敏感操作需要确认
- 保护用户隐私

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码检查

```bash
ruff check src/
mypy src/
```

## 更新日志

### v0.2.0 (2026-01)
- 新增 GUI 交互界面
- 实现两层视觉架构（轻量级状态分析 + 重量级元素定位）
- 改进步骤完成判断逻辑（基于用户输入事件）
- 添加任务目标自动检测
- RAG服务与Planner集成
- VL API增加重试机制

### v0.1.0
- 初始版本
- SimToM意图理解
- ReAct任务规划
- 基础视觉服务

## License

MIT
