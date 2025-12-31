"""ASR语音识别服务测试脚本

使用方法:
1. 确保已安装依赖: pip install pyaudio websockets
2. 配置环境变量或直接修改下面的配置
3. 运行: python -m tests.test_asr_service

测试模式:
- 模式1: 从麦克风实时录音识别
- 模式2: 从音频文件识别
"""

from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.asr_service import ASRService, ASRConfig, ASRResult, AudioCapture
from loguru import logger


# ============ 配置区域 ============
# 请填写你的项目ID和EasyLLM ID
ASR_PROJECT_ID = os.getenv("ASR_PROJECT_ID", "ellm_7asJ6QtG2wmknC3iBH7l4B")
ASR_EASYLLM_ID = os.getenv("ASR_EASYLLM_ID", "7asJ6QtG2wmknC3iBH7l4B")
ASR_API_KEY = os.getenv(
    "ASR_API_KEY",
    "CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ"
)
# =================================


def setup_logging():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


async def test_connection():
    """测试1: 测试WebSocket连接"""
    print("\n" + "=" * 50)
    print("测试1: WebSocket连接测试")
    print("=" * 50)
    
    config = ASRConfig(
        project_id=ASR_PROJECT_ID,
        easyllm_id=ASR_EASYLLM_ID,
        api_key=ASR_API_KEY,
        format="pcm",
        sample_rate=16000,
        heartbeat=True,
    )
    
    asr = ASRService(config)
    await asr.initialize()
    
    print(f"正在连接到ASR服务...")
    print(f"  Project ID: {ASR_PROJECT_ID}")
    print(f"  EasyLLM ID: {ASR_EASYLLM_ID}")
    
    try:
        connected = await asr.connect()
        if connected:
            print("✅ 连接成功!")
            await asyncio.sleep(1)
            await asr.disconnect()
            print("✅ 断开连接成功!")
            return True
        else:
            print("❌ 连接失败!")
            return False
    except Exception as e:
        print(f"❌ 连接异常: {e}")
        return False


async def test_microphone_recognition():
    """测试2: 麦克风实时识别"""
    print("\n" + "=" * 50)
    print("测试2: 麦克风实时语音识别")
    print("=" * 50)
    print("请对着麦克风说话，按 Ctrl+C 停止...")
    print("-" * 50)
    
    config = ASRConfig(
        project_id=ASR_PROJECT_ID,
        easyllm_id=ASR_EASYLLM_ID,
        api_key=ASR_API_KEY,
        format="pcm",
        sample_rate=16000,
        heartbeat=True,
    )
    
    asr = ASRService(config)
    await asr.initialize()
    
    # 设置回调
    def on_result(result: ASRResult):
        if result.is_sentence_end:
            print(f"🎯 [完整句子] {result.text}")
        else:
            print(f"   [识别中...] {result.text}", end="\r")
    
    asr.set_result_callback(on_result)
    
    # 连接ASR服务
    connected = await asr.connect()
    if not connected:
        print("❌ 无法连接到ASR服务")
        return
    
    # 启动音频采集
    try:
        audio_capture = AudioCapture(
            sample_rate=16000,
            channels=1,
            chunk_size=3200,
        )
        audio_capture.start()
        
        print("🎤 开始录音...")
        
        # 流式识别
        async for result in asr.stream_recognize(audio_capture.get_audio_stream()):
            pass  # 结果通过回调处理
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 停止录音")
    except ImportError:
        print("❌ 请安装 pyaudio: pip install pyaudio")
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        if 'audio_capture' in locals():
            audio_capture.stop()
        await asr.disconnect()


async def test_audio_file_recognition(audio_file: str):
    """测试3: 从音频文件识别"""
    print("\n" + "=" * 50)
    print(f"测试3: 音频文件识别")
    print(f"文件: {audio_file}")
    print("=" * 50)
    
    if not os.path.exists(audio_file):
        print(f"❌ 文件不存在: {audio_file}")
        return
    
    config = ASRConfig(
        project_id=ASR_PROJECT_ID,
        easyllm_id=ASR_EASYLLM_ID,
        api_key=ASR_API_KEY,
        format="wav",  # 根据文件格式调整
        sample_rate=16000,
        heartbeat=True,
    )
    
    asr = ASRService(config)
    await asr.initialize()
    
    # 读取音频文件
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    
    print(f"音频大小: {len(audio_data)} bytes")
    print("正在识别...")
    
    try:
        result = await asr.recognize_audio(audio_data)
        print(f"\n识别结果: {result.text}")
        print(f"是否句子结束: {result.is_sentence_end}")
        print(f"开始时间: {result.begin_time}ms")
        print(f"结束时间: {result.end_time}ms")
    except Exception as e:
        print(f"❌ 识别失败: {e}")
    finally:
        await asr.disconnect()


async def test_simple_send_receive():
    """测试4: 简单的发送接收测试（使用静音数据）"""
    print("\n" + "=" * 50)
    print("测试4: 简单发送接收测试")
    print("=" * 50)
    
    config = ASRConfig(
        project_id=ASR_PROJECT_ID,
        easyllm_id=ASR_EASYLLM_ID,
        api_key=ASR_API_KEY,
        format="pcm",
        sample_rate=16000,
        heartbeat=True,
    )
    
    asr = ASRService(config)
    await asr.initialize()
    
    try:
        connected = await asr.connect()
        if not connected:
            print("❌ 连接失败")
            return
        
        print("✅ 连接成功")
        
        # 发送一些静音数据（全0）
        silence = bytes(3200)  # 3200字节的静音
        
        print("发送静音数据...")
        for i in range(5):
            await asr.send_audio(silence)
            print(f"  发送第 {i+1} 个数据包")
            await asyncio.sleep(0.1)
        
        print("等待响应...")
        await asyncio.sleep(2)
        
        print("✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    finally:
        await asr.disconnect()


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 50)
    print("ASR语音识别服务测试")
    print("=" * 50)
    print("1. 测试WebSocket连接")
    print("2. 麦克风实时识别 (需要pyaudio)")
    print("3. 音频文件识别")
    print("4. 简单发送接收测试")
    print("5. 运行所有测试")
    print("0. 退出")
    print("-" * 50)


async def main():
    """主函数"""
    setup_logging()
    
    # 检查配置
    if ASR_PROJECT_ID == "your_project_id_here":
        print("⚠️  警告: 请先配置 ASR_PROJECT_ID")
        print("   可以通过环境变量设置，或直接修改脚本中的配置")
        print()
    
    while True:
        print_menu()
        choice = input("请选择测试项 [0-5]: ").strip()
        
        if choice == "0":
            print("再见!")
            break
        elif choice == "1":
            await test_connection()
        elif choice == "2":
            await test_microphone_recognition()
        elif choice == "3":
            audio_file = input("请输入音频文件路径: ").strip()
            if audio_file:
                await test_audio_file_recognition(audio_file)
        elif choice == "4":
            await test_simple_send_receive()
        elif choice == "5":
            await test_connection()
            await test_simple_send_receive()
            print("\n麦克风测试需要手动运行 (选项2)")
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")
