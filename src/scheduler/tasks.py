"""
定时任务 CLI 入口

供 GitHub Actions 调用，独立运行三大日报生成任务。

用法:
    python -m src.scheduler.tasks morning
    python -m src.scheduler.tasks noon
    python -m src.scheduler.tasks closing
"""

import argparse
import logging
import sys

from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def _resolve_admin_open_id(config) -> str:
    """解析报告推送接收人。

    优先使用运行时配置，未设置时回退到环境变量，
    这样 GitHub Actions 只配置 `ADMIN_USER_OPEN_ID` 也能正常推送。
    """
    runtime_value = config.get_value("admin_user_open_id")
    if runtime_value and runtime_value.strip():
        return runtime_value.strip()

    return settings.admin_user_open_id.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成定时报告")
    parser.add_argument(
        "report_type",
        choices=["morning", "noon", "closing", "scan"],
        help="任务类型",
    )

    args = parser.parse_args()

    # 初始化数据库
    from src.db import init_database
    init_database()

    # 初始化依赖
    from src.config.manager import get_config
    from src.bot.client import get_feishu_client
    from src.agents.report_agent import get_report_agent, ReportType

    config = get_config()

    # 检查系统状态
    if not config.get_enabled():
        logger.warning("System is disabled, skipping report")
        return 0

    market = config.get_market()
    logger.info("Running %s task (market=%s)...", args.report_type, market)

    if args.report_type == "scan":
        from src.agents.alert_agent import get_alert_agent

        alert_agent = get_alert_agent()
        result = alert_agent.scan_watchlist()
        logger.info(
            "Scan finished: scanned=%d triggered=%d deliverable=%d",
            result["scanned"], result["triggered"], result["deliverable"],
        )

        admin_open_id = _resolve_admin_open_id(config)
        deliverable = [item for item in result["alerts"] if item["should_send"]]
        if admin_open_id and deliverable:
            try:
                feishu = get_feishu_client()
                lines = [
                    "【盘中预警】",
                    f"扫描标的: {result['scanned']}",
                    f"触发预警: {result['triggered']}",
                ]
                for item in deliverable[:10]:
                    event = item["event"]
                    lines.append(
                        f"- {event.related_code or event.title} {event.title}（强度 {event.strength:.1f}）"
                    )
                feishu.send_text(admin_open_id, "\n".join(lines)[:1500])
                alert_agent.mark_delivered(
                    [item["event"].event_id for item in deliverable]
                )
                logger.info("Scan alerts sent to admin")
            except Exception as exc:
                logger.warning("Failed to send scan alerts via Feishu: %s", exc)
        elif not admin_open_id:
            logger.info("No admin open_id configured, skipping scan delivery")
        else:
            logger.info(result["message"])

        return 0

    # 生成报告
    report_type = ReportType(args.report_type)
    agent = get_report_agent()

    try:
        report_text = agent.generate_report(report_type)
    except Exception as exc:
        logger.error("Failed to generate report: %s", exc)
        return 1

    if not report_text:
        logger.warning("Empty report generated")
        return 0

    logger.info(
        "Report generated: %s (%d chars)", args.report_type, len(report_text)
    )

    # 推送到飞书
    admin_open_id = _resolve_admin_open_id(config)
    if admin_open_id:
        try:
            feishu = get_feishu_client()
            feishu.send_text(admin_open_id, report_text[:1500])
            logger.info("Report sent to admin")
        except Exception as exc:
            logger.warning("Failed to send report via Feishu: %s", exc)
    else:
        logger.info("No admin open_id configured, skipping Feishu delivery")

    return 0


if __name__ == "__main__":
    sys.exit(main())
