"""ASR快速测试脚本 - 直接运行测试麦克风识别

使用方法:
    cd elderly-assistant-agent
    python tests/quick_test_asr.py

需要先设置环境变量:
    set ASR_PROJECT_ID=你的项目ID
    set ASR_EASYLLM_ID=你的EasyLLM ID
"""

from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    # 导入
    from src.services.asr_service import ASRService, ASRConfig, ASRResult, AudioCapture
    
    # 配置
    config = ASRConfig(
        project_id=os.getenv("ASR_PROJECT_ID", ""),
        easyllm_id=os.getenv("ASR_EASYLLM_ID", ""),
        api_key=os.getenv(
            "ASR_API_KEY",
            "CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ"
        ),
        format="pcm",
        sample_rate=16000,
        heartbeat=True,
    )
    
    if not config.project_id or not config.easyllm_id:
        print("❌ 请设置环境变量:")
        print("   set ASR_PROJECT_ID=你的项目ID")
        print("   set ASR_EASYLLM_ID=你的EasyLLM ID")
        return
    
    print("=" * 50)
    print("ASR语音识别快速测试")
    print("=" * 50)
    print(f"Project ID: {config.project_id}")
    print(f"EasyLLM ID: {config.easyllm_id}")
    print("-" * 50)
    
    # 创建ASR服务
    asr = ASRService(config)
    await asr.initialize()
    
    # 设置回调 - 打印识别结果
    def on_result(result: ASRResult):
        if result.is_sentence_end:
            print(f"\n🎯 识别结果: {result.text}")
            print("-" * 50)
        else:
            # 实时显示识别中的文字
            print(f"\r   识别中: {result.text}          ", end="", flush=True)
    
    asr.set_result_callback(on_result)
    
    # 连接
    print("正在连接ASR服务...")
    connected = await asr.connect()
    
    if not connected:
        print("❌ 连接失败!")
        return
    
    print("✅ 连接成功!")
    print("\n🎤 请对着麦克风说话...")
    print("   按 Ctrl+C 停止\n")
    
    # 启动麦克风采集
    try:
        audio_capture = AudioCapture(
            sample_rate=16000,
            channels=1,
            chunk_size=3200,
        )
        audio_capture.start()
        
        # 流式识别
        async for _ in asr.stream_recognize(audio_capture.get_audio_stream()):
            pass
            
    except KeyboardInterrupt:
        print("\n\n⏹️ 停止录音")
    except ImportError as e:
        print(f"\n❌ 缺少依赖: {e}")
        print("   请安装: pip install pyaudio")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        if 'audio_capture' in locals():
            audio_capture.stop()
        await asr.disconnect()
        print("已断开连接")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已退出")
