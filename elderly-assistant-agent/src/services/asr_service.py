"""语音识别服务 - 使用FunASR实时识别"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

import httpx
from loguru import logger

from ..config import config


@dataclass
class ASRResult:
    """语音识别结果"""
    text: str
    is_final: bool
    confidence: float = 1.0
    language: str = "zh"


class ASRService:
    """语音识别服务"""
    
    def __init__(self) -> None:
        self._api_url = config.api.fun_asr_url
        self._client: Optional[httpx.AsyncClient] = None
        self._is_listening = False
        self._on_result_callback: Optional[Callable[[ASRResult], None]] = None
    
    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info(f"ASR服务初始化完成，API地址: {self._api_url}")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_result_callback(self, callback: Callable[[ASRResult], None]) -> None:
        """设置结果回调"""
        self._on_result_callback = callback
    
    async def recognize_audio(self, audio_data: bytes) -> ASRResult:
        """识别音频数据"""
        if not self._client:
            raise RuntimeError("ASR服务未初始化")
        
        try:
            response = await self._client.post(
                f"{self._api_url}/asr",
                files={"audio": ("audio.wav", audio_data, "audio/wav")},
                data={"language": "zh"},
            )
            response.raise_for_status()
            
            result = response.json()
            return ASRResult(
                text=result.get("text", ""),
                is_final=result.get("is_final", True),
                confidence=result.get("confidence", 1.0),
            )
        except httpx.HTTPError as e:
            logger.error(f"ASR请求失败: {e}")
            return ASRResult(text="", is_final=True, confidence=0.0)
    
    async def stream_recognize(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[ASRResult]:
        """流式识别"""
        if not self._client:
            raise RuntimeError("ASR服务未初始化")
        
        buffer: list[bytes] = []
        buffer_duration_ms = 0
        chunk_duration_ms = 100  # 每个chunk约100ms
        
        async for chunk in audio_stream:
            buffer.append(chunk)
            buffer_duration_ms += chunk_duration_ms
            
            # 每500ms发送一次识别请求
            if buffer_duration_ms >= 500:
                audio_data = b"".join(buffer)
                result = await self.recognize_audio(audio_data)
                
                if result.text:
                    yield result
                    if self._on_result_callback:
                        self._on_result_callback(result)
                
                if result.is_final:
                    buffer.clear()
                    buffer_duration_ms = 0
    
    async def start_listening(self) -> None:
        """开始监听"""
        self._is_listening = True
        logger.info("开始语音监听")
    
    async def stop_listening(self) -> None:
        """停止监听"""
        self._is_listening = False
        logger.info("停止语音监听")
    
    @property
    def is_listening(self) -> bool:
        return self._is_listening


class AudioCapture:
    """音频采集 (使用PyAudio)"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1600,  # 100ms at 16kHz
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._is_capturing = False
        self._audio = None
        self._stream = None
    
    def start(self) -> None:
        """开始采集"""
        try:
            import pyaudio
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._chunk_size,
            )
            self._is_capturing = True
            logger.info("音频采集已启动")
        except Exception as e:
            logger.error(f"启动音频采集失败: {e}")
            raise
    
    def stop(self) -> None:
        """停止采集"""
        self._is_capturing = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()
        logger.info("音频采集已停止")
    
    async def get_audio_stream(self) -> AsyncIterator[bytes]:
        """获取音频流"""
        while self._is_capturing and self._stream:
            try:
                data = self._stream.read(self._chunk_size, exception_on_overflow=False)
                yield data
                await asyncio.sleep(0.01)  # 避免阻塞
            except Exception as e:
                logger.error(f"读取音频数据失败: {e}")
                break
