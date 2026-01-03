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
        """播放音频 - 多种方式回退"""
        if not audio_data:
            logger.warning("音频数据为空，跳过播放")
            return
            
        temp_path = None
        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            logger.debug(f"音频文件保存到: {temp_path}, 大小: {len(audio_data)} bytes")
            
            import subprocess
            import sys
            
            if sys.platform == "win32":
                # 方法1: 尝试使用 pygame（最可靠）
                played = await self._play_with_pygame(temp_path)
                
                # 方法2: 尝试使用 playsound
                if not played:
                    played = await self._play_with_playsound(temp_path)
                
                # 方法3: 使用 Windows Media Player (wmplayer)
                if not played:
                    played = await self._play_with_wmplayer(temp_path)
                
                # 方法4: 使用 PowerShell (最后的回退)
                if not played:
                    await self._play_with_powershell(temp_path)
            else:
                # Linux/Mac: 使用 ffplay 或 aplay
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
            # 延迟删除临时文件，确保播放完成
            if temp_path and os.path.exists(temp_path):
                await asyncio.sleep(0.5)
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    async def _play_with_pygame(self, file_path: str) -> bool:
        """使用 pygame 播放音频"""
        try:
            import pygame
            
            def play():
                pygame.mixer.init()
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.quit()
            
            await asyncio.get_event_loop().run_in_executor(None, play)
            logger.debug("pygame 播放成功")
            return True
        except ImportError:
            logger.debug("pygame 未安装")
            return False
        except Exception as e:
            logger.debug(f"pygame 播放失败: {e}")
            return False
    
    async def _play_with_playsound(self, file_path: str) -> bool:
        """使用 playsound 播放音频"""
        try:
            from playsound import playsound
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: playsound(file_path)
            )
            logger.debug("playsound 播放成功")
            return True
        except ImportError:
            logger.debug("playsound 未安装")
            return False
        except Exception as e:
            logger.debug(f"playsound 播放失败: {e}")
            return False
    
    async def _play_with_wmplayer(self, file_path: str) -> bool:
        """使用 Windows Media Player 播放"""
        try:
            import subprocess
            import time
            
            # 使用 start 命令打开默认播放器
            process = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.Popen(
                    ["cmd", "/c", "start", "/min", "", file_path],
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            )
            
            # 估算播放时间（根据文件大小）
            file_size = os.path.getsize(file_path)
            # MP3 128kbps ≈ 16KB/s
            duration = max(1, file_size / 16000)
            await asyncio.sleep(duration + 0.5)
            
            logger.debug("wmplayer 播放完成")
            return True
        except Exception as e:
            logger.debug(f"wmplayer 播放失败: {e}")
            return False
    
    async def _play_with_powershell(self, file_path: str) -> bool:
        """使用 PowerShell 播放（回退方案）"""
        try:
            import subprocess
            
            # 使用更简单的 PowerShell 命令
            ps_cmd = f'''
            $player = New-Object Media.SoundPlayer "{file_path}"
            $player.PlaySync()
            '''
            
            # 如果是 MP3，需要用不同的方法
            if file_path.endswith('.mp3'):
                ps_cmd = f'''
                Add-Type -AssemblyName PresentationCore
                $player = New-Object System.Windows.Media.MediaPlayer
                $player.Open([Uri]"{file_path}")
                Start-Sleep -Milliseconds 300
                $player.Play()
                $duration = $player.NaturalDuration.TimeSpan.TotalSeconds
                if ($duration -gt 0) {{
                    Start-Sleep -Seconds ($duration + 0.5)
                }} else {{
                    Start-Sleep -Seconds 3
                }}
                $player.Close()
                '''
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    timeout=30
                )
            )
            
            if result.returncode == 0:
                logger.debug("PowerShell 播放成功")
                return True
            else:
                logger.debug(f"PowerShell 播放失败: {result.stderr.decode('utf-8', errors='ignore')}")
                return False
        except Exception as e:
            logger.debug(f"PowerShell 播放异常: {e}")
            return False
