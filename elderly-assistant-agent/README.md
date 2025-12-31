# 老年人电脑助手 Agent

帮助不会使用电子产品的中老年人使用电脑的AI助手。

## 核心特性

### 1. 语义到操作映射 (Semantic-to-Procedural Mapping)
- 理解老年人的模糊表述（如"手机吃钱"="流量超标"）
- 将生活化语言转换为具体操作步骤
- 支持家庭成员称呼映射（如"老二"="张三"）

### 2. 多模态交互
- 语音输入：使用FunASR实时语音识别
- 语音输出：使用CosyVoice语音合成，语速可调
- 屏幕理解：使用Qwen-VL分析当前屏幕内容

### 3. 智能任务规划 (ReAct模式)
- 自动分解复杂任务为简单步骤
- 每一步给予清晰的语音指导
- 错误时自动重新规划，提供恢复方案

### 4. 知识库与RAG
- 从短视频平台提取操作知识
- 使用思维导图方式压缩上下文
- 语义搜索相关操作指南

### 5. 安全防护
- 诈骗信息检测与警告
- 敏感操作确认机制
- 隐私数据保护

## 技术栈

| 组件 | 技术 |
|------|------|
| 多模态理解 | Qwen3-VL-235B-A22B-Instruct |
| 语言理解 | Qwen3-235B-A22B-Instruct-2507 |
| 语音识别 | FunASR (实时) |
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
pip install -e .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，配置API地址
```

### 3. 启动服务

```bash
python -m src.main
```

### 4. API使用

```bash
# 文本输入
curl -X POST http://localhost:8080/api/input/text \
  -H "Content-Type: application/json" \
  -d '{"text": "我想给女儿打个电话"}'

# 获取会话状态
curl http://localhost:8080/api/session/state
```

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
