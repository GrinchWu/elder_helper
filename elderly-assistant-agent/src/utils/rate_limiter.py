"""速率限制器"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class RateLimiter:
    """速率限制器 - 滑动窗口算法"""
    
    max_requests: int = 30           # 最大请求数
    window_seconds: int = 60         # 时间窗口（秒）
    _timestamps: deque = field(default_factory=deque)
    _lock: Optional[asyncio.Lock] = None
    
    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """获取许可"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        async with self._lock:
            now = time.time()
            
            # 清理过期的时间戳
            while self._timestamps and self._timestamps[0] < now - self.window_seconds:
                self._timestamps.popleft()
            
            # 检查是否超过限制
            if len(self._timestamps) >= self.max_requests:
                return False
            
            # 记录当前请求
            self._timestamps.append(now)
            return True
    
    async def wait_and_acquire(self) -> None:
        """等待并获取许可"""
        while not await self.acquire():
            # 计算需要等待的时间
            if self._timestamps:
                wait_time = self._timestamps[0] + self.window_seconds - time.time()
                if wait_time > 0:
                    logger.debug(f"速率限制，等待 {wait_time:.2f} 秒")
                    await asyncio.sleep(wait_time)
    
    @property
    def remaining(self) -> int:
        """剩余可用请求数"""
        now = time.time()
        
        # 清理过期的时间戳
        while self._timestamps and self._timestamps[0] < now - self.window_seconds:
            self._timestamps.popleft()
        
        return max(0, self.max_requests - len(self._timestamps))
    
    def reset(self) -> None:
        """重置限制器"""
        self._timestamps.clear()


class TokenBucketLimiter:
    """令牌桶限流器"""
    
    def __init__(
        self,
        rate: float = 1.0,           # 每秒生成的令牌数
        capacity: int = 10,          # 桶容量
    ) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """获取令牌"""
        async with self._lock:
            now = time.time()
            
            # 添加新令牌
            elapsed = now - self._last_update
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate,
            )
            self._last_update = now
            
            # 检查是否有足够的令牌
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            
            return False
    
    async def wait_and_acquire(self, tokens: int = 1) -> None:
        """等待并获取令牌"""
        while not await self.acquire(tokens):
            # 计算需要等待的时间
            needed = tokens - self._tokens
            wait_time = needed / self._rate
            await asyncio.sleep(wait_time)
