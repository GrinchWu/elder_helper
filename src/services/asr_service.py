"""语音识别服务 - 使用Sophnet流式WebSocket API"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import websockets
from loguru import logger

from ..config import config


@dataclass
class ASRResult:
    """语音识别结果"""
    text: str
    is_sentence_end: bool = False
    begin_time: int = 0          # 句子开始时刻（毫秒）
    end_time: Optional[int] = None  # 句子结束时刻（毫秒）
    words: list[dict] = field(default_factory=list)  # 字级别结果
    confidence: float = 1.0
    language: str = "zh"


@dataclass
class ASRConfig:
    """ASR配置"""
    project_id: str = ""
    easyllm_id: str = ""
    api_key: str = "CL9TPTG2Qro1oto8pSyBq6bQpXFCRs8g-Yl2d7nuElQBr2HtqkA19yu7wC1Zy6DGWOe4BELfLoZXUfuhD3yIoQ"
    format: str = "pcm"          # pcm, wav, mp3, opus, speex, aac, amr
    sample_rate: int = 16000     # 16k效果更好
    heartbeat: bool = True       # 开启心跳保持连接
    chunk_size: int = 3200       # 每次发送的音频字节数


class ASRService:
    """流式语音识别服务 - WebSocket实现"""
    
    BASE_URL = "wss://www.sophnet.com/api/open-apis/projects"
    
    def __init__(self, asr_config: Optional[ASRConfig] = None) -> None:
        self._config = asr_config or ASRConfig()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._is_connected = False
        self._is_listening = False
        
        # 回调函数
        self._on_result_callback: Optional[Callable[[ASRResult], None]] = None
        self._on_sentence_end_callback: Optional[Callable[[ASRResult], None]] = None
        
        # 结果队列
        self._result_queue: asyncio.Queue[ASRResult] = asyncio.Queue()
    
    def set_config(self, asr_config: ASRConfig) -> None:
        """设置配置"""
        self._config = asr_config
    
    def _build_ws_url(self) -> str:
        """构建WebSocket连接URL"""
        base = f"{self.BASE_URL}/{self._config.project_id}/easyllms/stream-speech"
        params = (
            f"?easyllm_id={self._config.easyllm_id}"
            f"&apikey={self._config.api_key}"
            f"&format={self._config.format}"
            f"&sample_rate={self._config.sample_rate}"
            f"&heartbeat={str(self._config.heartbeat).lower()}"
        )
        return base + params
    
    async def initialize(self) -> None:
        """初始化服务（预热，不立即连接）"""
        logger.info("ASR服务初始化完成")
        logger.info(f"  - 采样率: {self._config.sample_rate}Hz")
        logger.info(f"  - 音频格式: {self._config.format}")
        logger.info(f"  - 心跳: {self._config.heartbeat}")
    
    async def connect(self) -> bool:
        """建立WebSocket连接"""
        if self._is_connected:
            return True
        
        url = self._build_ws_url()
        logger.info(f"正在连接ASR服务...")
        
        try:
            self._ws = await websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10,
            )
            
            # 等待连接确认
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            data = json.loads(response)
            
            if data.get("status") == "ok":
                self._is_connected = True
                logger.info("ASR WebSocket连接成功")
                
                # 启动接收任务
                asyncio.create_task(self._receive_loop())
                return True
            else:
                logger.error(f"ASR连接失败: {data}")
                return False
                
        except Exception as e:
            logger.error(f"ASR连接异常: {e}")
            self._is_connected = False
            return False
    
    async def disconnect(self) -> None:
        """断开连接"""
        if self._ws and self._is_connected:
            try:
                # 发送BYE关闭连接
                await self._ws.send("BYE")
                await self._ws.close()
                logger.info("ASR WebSocket已断开")
            except Exception as e:
                logger.warning(f"断开连接时出错: {e}")
            finally:
                self._is_connected = False
                self._ws = None
    
    async def close(self) -> None:
        """关闭服务"""
        await self.disconnect()
    
    async def _receive_loop(self) -> None:
        """接收识别结果的循环"""
        if not self._ws:
            return
        
        try:
            async for message in self._ws:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        result = self._parse_result(data)
                        
                        # 放入队列
                        await self._result_queue.put(result)
                        
                        # 触发回调
                        if self._on_result_callback:
                            self._on_result_callback(result)
                        
                        if result.is_sentence_end and self._on_sentence_end_callback:
                            self._on_sentence_end_callback(result)
                            
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析ASR响应: {message}")
                        
        except websockets.ConnectionClosed:
            logger.info("ASR WebSocket连接已关闭")
            self._is_connected = False
        except Exception as e:
            logger.error(f"ASR接收循环异常: {e}")
            self._is_connected = False
    
    def _parse_result(self, data: dict) -> ASRResult:
        """解析识别结果"""
        return ASRResult(
            text=data.get("text", ""),
            is_sentence_end=data.get("is_sentence_end", False),
            begin_time=data.get("begin_time", 0),
            end_time=data.get("end_time"),
            words=data.get("words", []),
        )
    
    async def send_audio(self, audio_data: bytes) -> None:
        """发送音频数据"""
        if not self._is_connected or not self._ws:
            # 自动重连
            connected = await self.connect()
            if not connected:
                raise RuntimeError("ASR服务未连接")
        
        try:
            # 按chunk_size分块发送
            chunk_size = self._config.chunk_size
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                await self._ws.send(chunk)
                await asyncio.sleep(0.01)  # 避免发送过快
                
        except Exception as e:
            logger.error(f"发送音频数据失败: {e}")
            self._is_connected = False
            raise
    
    async def recognize_audio(self, audio_data: bytes) -> ASRResult:
        """识别一段完整音频（非流式）"""
        # 确保连接
        if not self._is_connected:
            await self.connect()
        
        # 清空队列
        while not self._result_queue.empty():
            self._result_queue.get_nowait()
        
        # 发送音频
        await self.send_audio(audio_data)
        
        # 等待句子结束的结果
        final_result = ASRResult(text="")
        timeout = 10.0  # 最多等待10秒
        
        try:
            start_time = asyncio.get_event_loop().time()
            while True:
                remaining = timeout - (asyncio.get_event_loop().time() - start_time)
                if remaining <= 0:
                    break
                
                result = await asyncio.wait_for(
                    self._result_queue.get(),
                    timeout=remaining,
                )
                
                final_result = result
                if result.is_sentence_end:
                    break
                    
        except asyncio.TimeoutError:
            logger.warning("等待ASR结果超时")
        
        return final_result
    
    async def stream_recognize(
        self,
        audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[ASRResult]:
        """流式识别"""
        # 确保连接
        if not self._is_connected:
            await self.connect()
        
        # 启动发送任务
        async def send_task():
            async for chunk in audio_stream:
                if not self._is_connected:
                    break
                await self.send_audio(chunk)
        
        send_future = asyncio.create_task(send_task())
        
        try:
            while self._is_connected:
                try:
                    result = await asyncio.wait_for(
                        self._result_queue.get(),
                        timeout=1.0,
                    )
                    yield result
                    
                    if self._on_result_callback:
                        self._on_result_callback(result)
                        
                except asyncio.TimeoutError:
                    continue
                    
        finally:
            send_future.cancel()
    
    def set_result_callback(self, callback: Callable[[ASRResult], None]) -> None:
        """设置实时结果回调（每次有新识别结果时触发）"""
        self._on_result_callback = callback
    
    def set_sentence_end_callback(self, callback: Callable[[ASRResult], None]) -> None:
        """设置句子结束回调（当一句话识别完成时触发）"""
        self._on_sentence_end_callback = callback
    
    async def start_listening(self) -> None:
        """开始监听"""
        if not self._is_connected:
            await self.connect()
        self._is_listening = True
        logger.info("开始语音监听")
    
    async def stop_listening(self) -> None:
        """停止监听"""
        self._is_listening = False
        logger.info("停止语音监听")
    
    @property
    def is_listening(self) -> bool:
        return self._is_listening
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected


class AudioCapture:
    """音频采集 (使用PyAudio从麦克风采集)"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 3200,  # 对应ASR要求的3200字节
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
                frames_per_buffer=self._chunk_size // 2,  # 16bit = 2 bytes per sample
            )
            self._is_capturing = True
            logger.info(f"音频采集已启动 (采样率: {self._sample_rate}Hz)")
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
                # 读取音频数据
                data = self._stream.read(
                    self._chunk_size // 2,  # frames
                    exception_on_overflow=False,
                )
                yield data
                await asyncio.sleep(0.01)  # 让出控制权
            except Exception as e:
                logger.error(f"读取音频数据失败: {e}")
                break
    
    @property
    def is_capturing(self) -> bool:
        return self._is_capturing
