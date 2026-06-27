"""
FastAPI 后台调度器

使用 APScheduler 在 Railway 常驻进程中自动推送每日报告：
- 08:30 早报
- 12:00 午间观察
- 15:30 收盘复盘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.agents.report_agent import get_report_agent
from src.bot.client import get_feishu_client
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyReportJob:
    """每日自动报告任务配置。"""

    report_type: str
    hour: int
    minute: int
    generator_name: str


DAILY_REPORT_JOBS: tuple[DailyReportJob, ...] = (
    DailyReportJob(
        report_type="morning",
        hour=8,
        minute=30,
        generator_name="generate_morning_report",
    ),
    DailyReportJob(
        report_type="noon",
        hour=12,
        minute=0,
        generator_name="generate_noon_report",
    ),
    DailyReportJob(
        report_type="closing",
        hour=15,
        minute=30,
        generator_name="generate_closing_report",
    ),
)

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
        len(DAILY_REPORT_JOBS),
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


def send_daily_report(report_type: str) -> bool:
    """生成并推送单次日报。

    Args:
        report_type: morning / noon / closing

    Returns:
        True 表示已成功发送，False 表示跳过或失败。
    """
    if not settings.daily_report_enabled:
        _log_report_send(report_type, "disabled")
        return False

    admin_open_id = settings.admin_user_open_id.strip()
    if not admin_open_id:
        _log_report_send(report_type, "skipped", "admin_user_open_id is empty")
        return False

    try:
        report_text = _generate_report(report_type)
        get_feishu_client().send_markdown(admin_open_id, report_text)
        _log_report_send(report_type, "success")
        return True
    except Exception as exc:
        _log_report_send(report_type, "failed", str(exc))
        logger.exception(
            "Daily report send failed: report_type=%s error_message=%s",
            report_type,
            exc,
        )
        return False


def get_scheduler() -> Optional[AsyncIOScheduler]:
    """返回当前调度器实例，供测试/诊断使用。"""
    return _scheduler


def _register_daily_report_jobs(scheduler: AsyncIOScheduler) -> None:
    for job in DAILY_REPORT_JOBS:
        scheduler.add_job(
            send_daily_report,
            "cron",
            hour=job.hour,
            minute=job.minute,
            args=[job.report_type],
            id=f"daily_report_{job.report_type}",
            name=f"daily_report_{job.report_type}",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )


def _generate_report(report_type: str) -> str:
    generators: dict[str, str] = {
        job.report_type: job.generator_name
        for job in DAILY_REPORT_JOBS
    }
    generator_name = generators.get(report_type)
    if not generator_name:
        raise ValueError(f"Unsupported report_type: {report_type}")

    agent = get_report_agent()
    generator: Callable[[], str] = getattr(agent, generator_name)
    return generator()


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
