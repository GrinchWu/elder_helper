# 安装指南

## 方法1: 使用Conda (推荐)

### 1. 创建虚拟环境

```bash
# 进入项目目录
cd elderly-assistant-agent

# 使用environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate elderly-assistant
```

### 2. 验证安装

```bash
# 检查Python版本
python --version  # 应该显示 Python 3.10.x

# 检查关键依赖
python -c "import pyaudio; print('PyAudio OK')"
python -c "import websockets; print('WebSockets OK')"
python -c "import fastapi; print('FastAPI OK')"
```

### 3. 配置环境变量

```bash
# 复制示例配置
copy .env.example .env

# 编辑.env文件，填写必要的配置
# 特别是 ASR_PROJECT_ID 和 ASR_EASYLLM_ID
```

---

## 方法2: 使用pip

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境 (Windows)
venv\Scripts\activate

# 激活环境 (Linux/Mac)
source venv/bin/activate
```

### 2. 安装PyAudio (Windows特殊处理)

PyAudio在Windows上安装可能遇到问题，推荐以下方法：

**方法A: 使用pipwin**
```bash
pip install pipwin
pipwin install pyaudio
```

**方法B: 下载预编译wheel**
1. 访问 https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
2. 下载对应Python版本的wheel文件
3. 安装: `pip install PyAudio‑0.2.14‑cp310‑cp310‑win_amd64.whl`

**方法C: 使用conda安装pyaudio**
```bash
conda install -c conda-forge pyaudio
```

### 3. 安装其他依赖

```bash
pip install -r requirements.txt
```

---

## 方法3: 开发模式安装

```bash
# 激活环境后
pip install -e .

# 或者包含开发依赖
pip install -e ".[dev]"
```

---

## 环境变量配置

创建 `.env` 文件，配置以下内容：

```env
# ASR语音识别配置 (必填)
ASR_PROJECT_ID=你的项目ID
ASR_EASYLLM_ID=你的EasyLLM ID
ASR_API_KEY=CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ

# LLM配置 (根据实际情况填写)
QWEN_VL_API_URL=http://localhost:8000/v1
QWEN_LLM_API_URL=http://localhost:8001/v1
QWEN_API_KEY=your_api_key

# TTS配置
COSYVOICE_API_URL=http://localhost:8003

# 向量模型配置
BGE_M3_API_URL=http://localhost:8004
```

---

## 运行测试

```bash
# 激活环境
conda activate elderly-assistant

# 运行ASR测试
python tests/quick_test_asr.py

# 运行所有单元测试
pytest tests/ -v
```

---

## 启动服务

```bash
# 启动API服务
python -m src.main

# 或者启动CLI交互模式
python -m src.cli
```

---

## 常见问题

### Q: PyAudio安装失败
A: Windows上推荐使用conda安装:
```bash
conda install -c conda-forge pyaudio
```

### Q: WebSocket连接失败
A: 检查网络连接，确保能访问 `wss://www.sophnet.com`

### Q: 麦克风无法使用
A: 
1. 检查系统麦克风权限
2. 确保没有其他程序占用麦克风
3. 在系统设置中检查默认录音设备

### Q: 模块导入错误
A: 确保在项目根目录运行，或使用 `pip install -e .` 安装
