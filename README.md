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

### 3. 多模态交互
- 语音输入：使用Sophnet流式WebSocket ASR
- 语音输出：使用CosyVoice语音合成，语速可调
- 屏幕理解：使用Qwen2.5-VL分析当前屏幕内容

### 4. 智能任务规划 (ReAct模式)
- 自动分解复杂任务为简单步骤
- 每一步给予清晰的语音指导
- 错误时自动重新规划，提供恢复方案

### 5. 知识库与RAG
- 从短视频平台提取操作知识
- 使用思维导图方式压缩上下文
- 语义搜索相关操作指南

### 6. 安全防护
- 诈骗信息检测与警告
- 敏感操作确认机制
- 隐私数据保护

## 技术栈

| 组件 | 技术 |
|------|------|
| 多模态理解 | Qwen2.5-VL-72B-Instruct (Sophnet API) |
| 语言理解 | Qwen2.5-72B-Instruct (Sophnet API) |
| 意图理解 | SimToM + BDI模型 |
| 语音识别 | Sophnet WebSocket ASR |
| 语音合成 | CosyVoice |
| 向量模型 | BGE-M3 |
| 后端框架 | FastAPI |
| 知识图谱 | NetworkX |

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
│   │   ├── vision_service.py   # 视觉服务
│   │   ├── llm_service.py      # 大模型服务
│   │   ├── planner_service.py  # 任务规划
│   │   ├── safety_service.py   # 安全服务
│   │   └── embedding_service.py # 向量嵌入
│   ├── knowledge/          # 知识库
│   │   ├── video_extractor.py  # 视频知识提取
│   │   └── rag_service.py      # RAG服务
│   ├── utils/              # 工具类
│   │   ├── rate_limiter.py     # 速率限制
│   │   └── validators.py       # 数据验证
│   ├── config.py           # 配置管理
│   └── main.py             # 主程序入口
├── tests/                  # 测试
├── pyproject.toml          # 项目配置
└── .env.example            # 环境变量示例
```

## 快速开始

### 1. 安装依赖

```bash
cd elderly-assistant-agent
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# Windows下安装pyaudio（用于麦克风采集）
conda install pyaudio
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
VL_MODEL=Qwen2.5-VL-72B-Instruct

# ASR配置 (WebSocket)
ASR_PROJECT_ID=your_project_id
ASR_EASYLLM_ID=your_easyllm_id
```

### 3. 测试LLM服务

```bash
cd elderly-assistant-agent
python tests/test_llm_service.py
```

测试场景包括：
- 发邮件：`"我想给老同事发个信"`
- Word问题：`"我写的东西找不到了，刚才还在的"`
- 上网看新闻：`"我想看看人民网上有什么新消息"`

### 4. 启动服务

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
- 多模态：`Qwen2.5-VL-72B-Instruct`, `Qwen2.5-VL-32B-Instruct`

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
- 语速可调，默认较慢
- 每一步都给予鼓励

### 安全第一
- 检测诈骗信息
- 敏感操作需要确认
- 保护用户隐私


## 实现细节

### RAG服务实现
`src/knowledge/rag_service.py`

- 混合检索机制 (Hybrid Retrieval)

向量召回：使用 BGE-M3 模型对用户查询进行语义向量化，检索相关的 OperationGuide (操作指南) 和 KnowledgeNode (知识节点)。

图谱路径推理：利用 NetworkX 知识图谱，基于 NEXT_STEP (下一步)、REQUIRES (前置条件) 等边关系，自动推导操作的逻辑路径，确保回答的连贯性。

- 老年人查询扩展 (Query Expansion)

内置老年人语言映射字典（例如：将“手机吃钱”扩展为“流量超标”、“扣费”；“那个绿色的”扩展为“微信”）。

采用多路召回策略，同时检索原始查询和扩展后的专业术语，大幅提高模糊意图的命中率。

- 智能上下文构建

语义压缩：自动过滤口语填充词（如“然后”、“那个”、“一下”），在保留核心语义的前提下压缩 Context 长度，降低 LLM 推理延迟与成本。

多模态增强：在构建上下文时保留截图索引 [截图]，为后续的多模态展示做准备。

- 动态置信度评估

综合计算向量相似度、图谱路径连贯性 (Path Coherence) 和召回覆盖率。

输出 confidence 分数，支持低置信度时自动回退到 Web 搜索模式。

### 视频知识提取
`src/knowledge/video_extractor.py`

- 轻量采集：集成 DuckDuckGo + yt-dlp，无需下载视频文件，秒级提取全网（B站/抖音）教程的元数据与字幕。

- 智能解析：利用 Qwen3-VL 多模态模型分析视频意图，从杂乱口语中提炼标准操作流。

- 适老化重写：核心特性。自动将“技术术语”翻译为老年人能听懂的“大白话”（例：点击汉堡菜单 → 点击左上角的三条横线），并自动生成配套 FAQ

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


## License

MIT
