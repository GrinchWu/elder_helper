"""主程序入口"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from .config import config
from .api import create_app


def setup_logging() -> None:
    """配置日志"""
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stderr,
        level=config.log.level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    # 添加文件输出
    log_file = config.log.file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_file),
        level=config.log.level,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )


def main() -> None:
    """主函数"""
    setup_logging()
    
    logger.info("启动老年人电脑助手...")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )


if __name__ == "__main__":
    main()
