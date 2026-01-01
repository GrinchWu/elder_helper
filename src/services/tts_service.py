"""语音合成服务 - 使用Sophnet CosyVoice API"""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import os
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from ..config import config


@dataclass
class TTSConfig:
    """TTS配置"""
    model: str = "cosyvoice-v1"
    voice: str = "longxiaochun"
    format: str = "MP3_16000HZ_MONO_128KBPS"
    volume: int = 80
    speech_rate: float = 0.9
    pitch_rate: float = 1.0


class TTSService:
    """语音合成服务 - Sophnet API"""
    
    def __init__(self) -> None:
        self._project_id = "ellm_143peGYFl1Kh1ihRDWrE3f"
        self._easyllm_id = "143peGYFl1Kh1ihRDWrE3f"
        self._api_key = config.api.api_key
        self._base_url = "https://www.sophnet.com/api/open-apis"
        self._client: Optional[httpx.AsyncClient] = None
        self._config = TTSConfig()
        self._is_speaking = False

    async def initialize(self) -> None:
        """初始化服务"""
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info("TTS服务初始化完成")
    
    async def close(self) -> None:
        """关闭服务"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def set_speed(self, speed: float) -> None:
        """设置语速 (0.5-2.0)"""
        self._config.speech_rate = max(0.5, min(2.0, speed))
    
    async def synthesize(self, text: str) -> bytes:
        """合成语音（非流式）"""
        if not self._client:
            raise RuntimeError("TTS服务未初始化")
        
        if not text.strip():
            return b""
        
        url = f"{self._base_url}/projects/{self._project_id}/easyllms/voice/synthesize-audio"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        
        payload = {
            "easyllm_id": self._easyllm_id,
            "text": [text],
            "synthesis_param": {
                "model": self._config.model,
                "voice": self._config.voice,
                "format": self._config.format,
                "volume": self._config.volume,
                "speechRate": self._config.speech_rate,
                "pitchRate": self._config.pitch_rate,
            }
        }
        
        try:
            response = await self._client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as e:
            logger.error(f"TTS请求失败: {e}")
            return b""

    async def speak(self, text: str) -> None:
        """播放语音"""
        if self._is_speaking:
            return
        
        self._is_speaking = True
        try:
            audio_data = await self.synthesize(text)
            if audio_data:
                await self._play_audio(audio_data)
        finally:
            self._is_speaking = False
    
    async def speak_async(self, text: str) -> None:
        """异步播放语音（不阻塞）"""
        asyncio.create_task(self.speak(text))
    
    async def speak_status(self, status: str) -> None:
        """播放状态提示"""
        await self.speak(status)
    
    async def speak_step(self, step_num: int, total: int, instruction: str) -> None:
        """播放步骤指令"""
        text = f"第{step_num}步，共{total}步。{instruction}"
        await self.speak(text)
    
    async def speak_success(self, msg: str = "操作完成！") -> None:
        """播放成功提示"""
        await self.speak(msg)
    
    async def speak_error(self, msg: str) -> None:
        """播放错误提示"""
        await self.speak(f"出了点问题：{msg}")
    
    async def speak_welcome(self) -> None:
        """播放欢迎语"""
        await self.speak("您好！我是您的电脑助手，请告诉我您想做什么。")

    async def _play_audio(self, audio_data: bytes) -> None:
        """播放音频 - 使用Windows系统命令"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            # Windows: 使用PowerShell播放音频
            import subprocess
            import sys
            
            if sys.platform == "win32":
                # 使用Windows Media Player COM对象播放
                ps_cmd = f'''
                Add-Type -AssemblyName presentationCore
                $player = New-Object System.Windows.Media.MediaPlayer
                $player.Open("{temp_path}")
                $player.Play()
                Start-Sleep -Milliseconds 500
                while ($player.Position -lt $player.NaturalDuration.TimeSpan) {{
                    Start-Sleep -Milliseconds 100
                }}
                $player.Close()
                '''
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["powershell", "-Command", ps_cmd],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                )
            else:
                # Linux/Mac: 使用ffplay或aplay
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", temp_path],
                        capture_output=True
                    )
                )
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
