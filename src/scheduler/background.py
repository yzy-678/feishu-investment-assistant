"""
FastAPI 后台调度器

使用 APScheduler 在 Railway 常驻进程中自动推送每日早报：
- 08:30 早报
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config.settings import settings
from src.reports.morning_report_service import get_morning_report_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MorningReportJob:
    """每日早报任务配置。"""

    hour: int
    minute: int


MORNING_REPORT_JOB = MorningReportJob(hour=8, minute=30)

_scheduler: Optional[AsyncIOScheduler] = None


def start_scheduler() -> None:
    """启动后台调度器。

    该函数由 FastAPI startup 调用；在 Railway 上随 Web 服务进程长期运行。
    """
    global _scheduler  # noqa: PLW0603

    if not settings.daily_report_enabled:
        logger.info(
            "Daily report scheduler disabled: report_type=%s "
            "send_status=%s error_message=%s",
            "all",
            "disabled",
            "",
        )
        return

    if _scheduler is not None and getattr(_scheduler, "running", False):
        logger.info("Background scheduler already running")
        return

    timezone = _resolve_timezone(settings.timezone)
    scheduler = AsyncIOScheduler(timezone=timezone)
    _register_daily_report_jobs(scheduler)
    scheduler.start()
    _scheduler = scheduler

    logger.info(
        "Background scheduler started: timezone=%s jobs=%d",
        timezone.key,
        1,
    )


def stop_scheduler() -> None:
    """停止后台调度器。"""
    global _scheduler  # noqa: PLW0603

    if _scheduler is None:
        logger.info("Background scheduler not running")
        return

    _scheduler.shutdown(wait=False)
    _scheduler = None
    logger.info("Background scheduler stopped")


def send_morning_report() -> bool:
    """生成并推送单次早报。"""
    if not settings.daily_report_enabled:
        _log_report_send("morning", "disabled")
        return False

    try:
        sent = get_morning_report_service().send()
        if sent:
            _log_report_send("morning", "success")
        else:
            _log_report_send("morning", "skipped", "admin_user_open_id is empty")
        return sent
    except Exception as exc:
        _log_report_send("morning", "failed", str(exc))
        logger.exception(
            "Daily report send failed: report_type=%s error_message=%s",
            "morning",
            exc,
        )
        return False


def send_daily_report(report_type: str = "morning") -> bool:
    """Backward-compatible entrypoint for the Sprint1 morning report."""
    if report_type != "morning":
        _log_report_send(report_type, "skipped", "only morning report is enabled")
        return False
    return send_morning_report()


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """返回当前调度器实例，供测试/诊断使用。"""
    return _scheduler


def _register_daily_report_jobs(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        send_morning_report,
        "cron",
        hour=MORNING_REPORT_JOB.hour,
        minute=MORNING_REPORT_JOB.minute,
        id="daily_report_morning",
        name="daily_report_morning",
        replace_existing=True,
        coalesce=True,
        max_instances=1,
    )


def _resolve_timezone(timezone_name: str) -> ZoneInfo:
    name = (timezone_name or "Asia/Shanghai").strip() or "Asia/Shanghai"
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        logger.warning(
            "Invalid timezone configured: %s, fallback to Asia/Shanghai",
            timezone_name,
        )
        return ZoneInfo("Asia/Shanghai")


def _log_report_send(
    report_type: str,
    send_status: str,
    error_message: str = "",
) -> None:
    logger.info(
        "Daily report push: report_type=%s send_status=%s error_message=%s",
        report_type,
        send_status,
        error_message,
    )
