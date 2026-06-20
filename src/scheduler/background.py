"""
FastAPI 后台调度器

管理盘中扫描和实时预警的后台定时任务。
使用 APScheduler 的 AsyncIOScheduler。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 占位实现，后续集成 APScheduler
_scheduler: Optional[object] = None


def start_scheduler() -> None:
    """启动后台调度器"""
    global _scheduler
    logger.info("Background scheduler placeholder started")


def stop_scheduler() -> None:
    """停止后台调度器"""
    global _scheduler
    _scheduler = None
    logger.info("Background scheduler placeholder stopped")
