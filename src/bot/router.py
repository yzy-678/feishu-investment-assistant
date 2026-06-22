"""
飞书事件回调路由

FastAPI 路由，处理飞书事件回调：
- URL 验证挑战
- 消息事件处理
- 健康检查
- 内部配置 API
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request

from src.bot.handler import get_handler
from src.config.manager import get_config
from src.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_handler_safely(raw: dict[str, Any]) -> None:
    """后台执行 handler，并吞掉异常避免影响飞书回调确认。"""
    try:
        handler = get_handler()
        handler.handle_event(raw)
    except Exception as exc:
        logger.warning("Handler error: %s", exc)


@router.post("/feishu/event")
async def feishu_event(
    request: Request, background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """飞书事件回调入口

    处理两种请求：
    1. URL 验证挑战（首次配置飞书事件回调时）
    2. 消息接收事件（im.message.receive_v1）
    """
    raw = await request.json()

    # ── URL 验证挑战 ────────────────────────────────────
    if raw.get("type") == "url_verification":
        challenge = raw.get("challenge", "")
        token = raw.get("token", "")

        # 可选：验证 token
        if settings.feishu_event_verify_token:
            if token != settings.feishu_event_verify_token:
                logger.warning("Feishu event verify token mismatch")
                return {"code": -1, "msg": "invalid token"}

        logger.info("Feishu URL verification successful")
        return {"challenge": challenge}

    # ── 事件回调 ────────────────────────────────────────
    event_type = raw.get("header", {}).get("event_type", "")
    logger.debug("Received Feishu event: %s", event_type)

    if event_type == "im.message.receive_v1":
        background_tasks.add_task(_run_handler_safely, raw)

    return {"code": 0, "msg": "ok"}


@router.get("/health")
async def health() -> dict[str, str]:
    """健康检查端点"""
    return {"status": "ok"}


@router.get("/api/config")
async def api_config() -> dict[str, Any]:
    """获取运行时配置（供 GitHub Actions 调度任务调用）"""
    cfg = get_config()
    return {
        "enabled": cfg.get_enabled(),
        "market": cfg.get_market(),
        "scan_interval": cfg.get_scan_interval(),
    }
