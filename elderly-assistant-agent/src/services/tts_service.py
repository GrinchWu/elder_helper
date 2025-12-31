"""语音合成服务 - 使用CosyVoice"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from ..config import config


@dataclass
class TTSConfig:
    """TTS配置"""
    voice_id: str = "elderly_friendly"  # 老年人友好的声音
    speed: float = 0.8                   # 语速 (0.5-1.5)
    pitch: float = 1.0                   # 音调
    volume: float = 1.0                  # 音量


class TTSService:
    """语音合成服务"""
    
    def __init__(self) -> None:
        self._api_url = config.api.cosyvoice_url
        self._client: Optional[httpx.AsyncClient] = None
        self._config = TTSConfig()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info(f"TTS服务初始化完成，API地址: {self._api_url}")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_config(self, config: TTSConfig) -> None:
        """设置TTS配置"""
        self._config = config
    
    def set_speed(self, speed: float) -> None:
        """设置语速"""
        self._config.speed = max(0.5, min(1.5, speed))
    
    async def synthesize(self, text: str) -> bytes:
        """合成语音"""
        if not self._client:
            raise RuntimeError("TTS服务未初始化")
        
        if not text.strip():
            return b""
        
        try:
            # 对老年人友好的文本预处理
            processed_text = self._preprocess_text(text)
            
            response = await self._client.post(
                f"{self._api_url}/tts",
                json={
                    "text": processed_text,
                    "voice_id": self._config.voice_id,
                    "speed": self._config.speed,
                    "pitch": self._config.pitch,
                    "volume": self._config.volume,
                },
            )
            response.raise_for_status()
            
            return response.content
        except httpx.HTTPError as e:
            logger.error(f"TTS请求失败: {e}")
            return b""
    
    async def speak(self, text: str) -> None:
        """播放语音"""
        audio_data = await self.synthesize(text)
        if audio_data:
            await self._play_audio(audio_data)
    
    async def speak_step_instruction(
        self,
        step_number: int,
        total_steps: int,
        instruction: str,
    ) -> None:
        """播放步骤指令"""
        prefix = f"第{step_number}步，共{total_steps}步。"
        full_text = f"{prefix}{instruction}"
        await self.speak(full_text)
    
    async def speak_success(self, message: str = "做得好！") -> None:
        """播放成功提示"""
        await self.speak(message)
    
    async def speak_error_recovery(
        self,
        error_description: str,
        recovery_instruction: str,
    ) -> None:
        """播放错误恢复指导"""
        # 使用无过错假设的语言
        text = (
            f"没关系，这不是您的问题。"
            f"现在的情况是：{error_description}。"
            f"我们来这样做：{recovery_instruction}"
        )
        await self.speak(text)
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本，使其更适合老年人"""
        # 添加适当的停顿
        text = text.replace("，", "，，")  # 逗号处多停顿
        text = text.replace("。", "。。")  # 句号处多停顿
        
        # 替换技术术语为通俗表达
        replacements = {
            "点击": "用手指点一下",
            "双击": "快速点两下",
            "长按": "按住不放",
            "滑动": "用手指划一下",
            "下拉": "从上往下划",
            "上滑": "从下往上划",
            "返回": "点左上角的箭头回去",
            "确认": "点那个确定的按钮",
            "取消": "点那个不要的按钮",
            "输入框": "打字的地方",
            "搜索框": "找东西的地方",
            "菜单": "那个有很多选项的地方",
            "设置": "调整手机的地方",
            "图标": "那个小图片",
            "APP": "软件",
            "应用": "软件",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    async def _play_audio(self, audio_data: bytes) -> None:
        """播放音频"""
        try:
            import pyaudio
            import wave
            import io
            
            # 假设返回的是WAV格式
            audio_io = io.BytesIO(audio_data)
            
            try:
                with wave.open(audio_io, 'rb') as wf:
                    p = pyaudio.PyAudio()
                    stream = p.open(
                        format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                    )
                    
                    chunk_size = 1024
                    data = wf.readframes(chunk_size)
                    while data:
                        stream.write(data)
                        data = wf.readframes(chunk_size)
                        await asyncio.sleep(0)  # 让出控制权
                    
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
            except wave.Error:
                # 如果不是WAV格式，尝试直接播放
                logger.warning("音频格式不是WAV，尝试直接播放")
                
        except ImportError:
            logger.warning("PyAudio未安装，无法播放音频")
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
