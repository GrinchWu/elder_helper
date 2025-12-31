"""API路由"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from ..agent.elderly_agent import ElderlyAssistantAgent, AgentConfig, AgentState
from ..models.session import UserProfile


# 全局Agent实例
_agent: Optional[ElderlyAssistantAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _agent
    
    # 启动时初始化Agent
    logger.info("正在启动应用...")
    _agent = ElderlyAssistantAgent(AgentConfig())
    await _agent.initialize()
    
    yield
    
    # 关闭时清理
    logger.info("正在关闭应用...")
    if _agent:
        await _agent.close()


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="老年人电脑助手API",
        description="帮助老年人使用电脑的AI助手",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    _register_routes(app)
    
    return app


def _register_routes(app: FastAPI) -> None:
    """注册路由"""
    
    # ===== 健康检查 =====
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "agent_state": _agent.state.value if _agent else "not_initialized"}
    
    # ===== 文本输入 =====
    class TextInput(BaseModel):
        text: str
    
    @app.post("/api/input/text")
    async def process_text(input_data: TextInput):
        """处理文本输入"""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent未初始化")
        
        await _agent.process_text_input(input_data.text)
        
        return {
            "status": "processing",
            "agent_state": _agent.state.value,
        }
    
    # ===== 语音输入 =====
    @app.post("/api/input/audio")
    async def process_audio(audio: UploadFile = File(...)):
        """处理语音输入"""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent未初始化")
        
        audio_data = await audio.read()
        await _agent.process_voice_input(audio_data)
        
        return {
            "status": "processing",
            "agent_state": _agent.state.value,
        }
    
    # ===== 用户配置 =====
    class UserProfileInput(BaseModel):
        name: str = ""
        family_mapping: dict[str, str] = {}
        frequent_contacts: list[str] = []
        preferred_voice_speed: float = 0.8
    
    @app.post("/api/user/profile")
    async def set_user_profile(profile_input: UserProfileInput):
        """设置用户画像"""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent未初始化")
        
        profile = UserProfile(
            name=profile_input.name,
            family_mapping=profile_input.family_mapping,
            frequent_contacts=profile_input.frequent_contacts,
            preferred_voice_speed=profile_input.preferred_voice_speed,
        )
        _agent.set_user_profile(profile)
        
        return {"status": "ok"}
    
    # ===== 会话状态 =====
    @app.get("/api/session/state")
    async def get_session_state():
        """获取会话状态"""
        if not _agent or not _agent.session:
            raise HTTPException(status_code=503, detail="会话未初始化")
        
        session = _agent.session
        return {
            "agent_state": _agent.state.value,
            "session_state": session.state.value,
            "current_task": {
                "status": session.current_task.status.value if session.current_task else None,
                "progress": session.current_task.plan.progress_percentage if session.current_task and session.current_task.plan else 0,
            } if session.current_task else None,
            "conversation_history": session.conversation_history[-5:],
        }
    
    # ===== WebSocket实时通信 =====
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket实时通信"""
        await websocket.accept()
        
        if not _agent:
            await websocket.close(code=1011, reason="Agent未初始化")
            return
        
        # 设置回调
        async def on_state_change(state: AgentState):
            try:
                await websocket.send_json({
                    "type": "state_change",
                    "state": state.value,
                })
            except Exception:
                pass
        
        async def on_speak(text: str):
            try:
                await websocket.send_json({
                    "type": "speak",
                    "text": text,
                })
            except Exception:
                pass
        
        async def on_step_complete(step: int, total: int, success: bool):
            try:
                await websocket.send_json({
                    "type": "step_complete",
                    "step": step,
                    "total": total,
                    "success": success,
                })
            except Exception:
                pass
        
        _agent.set_callbacks(
            on_state_change=lambda s: asyncio.create_task(on_state_change(s)),
            on_speak=lambda t: asyncio.create_task(on_speak(t)),
            on_step_complete=lambda s, t, r: asyncio.create_task(on_step_complete(s, t, r)),
        )
        
        try:
            while True:
                data = await websocket.receive_json()
                
                msg_type = data.get("type")
                
                if msg_type == "text":
                    text = data.get("text", "")
                    if text:
                        await _agent.process_text_input(text)
                
                elif msg_type == "audio":
                    # Base64编码的音频数据
                    import base64
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        audio_data = base64.b64decode(audio_b64)
                        await _agent.process_voice_input(audio_data)
                
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
        except WebSocketDisconnect:
            logger.info("WebSocket连接断开")
        except Exception as e:
            logger.error(f"WebSocket错误: {e}")
        finally:
            _agent.set_callbacks()  # 清除回调
