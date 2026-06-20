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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成定时报告")
    parser.add_argument(
        "report_type",
        choices=["morning", "noon", "closing"],
        help="报告类型",
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
    logger.info("Generating %s report (market=%s)...", args.report_type, market)

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
    admin_open_id = config.get_value("admin_user_open_id")
    if admin_open_id:
        try:
            feishu = get_feishu_client()
            feishu.send_text(admin_open_id, report_text[:1500])
            logger.info("Report sent to admin")
        except Exception as exc:
            logger.warning("Failed to send report via Feishu: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
