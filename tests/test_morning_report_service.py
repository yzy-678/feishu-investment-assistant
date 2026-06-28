"""MorningReportService tests."""

from unittest.mock import MagicMock

from src.reports import morning_report_service
from src.reports.morning_report_service import MorningReportService


def test_send_calls_report_agent_and_feishu_client(monkeypatch):
    agent = MagicMock()
    agent.generate_morning_report.return_value = (
        "## 【早报】2026-06-28\n\n"
        "### 市场概览\n市场整体震荡修复。\n强势股暂不推送。\n\n"
        "### 热点板块\nAI、半导体活跃。\n\n"
        "### 风险提示\n关注成交不足风险。\n\n"
        "### 强势股\n不应推送。\n\n"
        "### 自选股观察\n不应推送。"
    )
    feishu = MagicMock()
    screener = MagicMock()
    analyzer = MagicMock()
    pool = MagicMock()
    screener.screen_top_stocks.return_value = []
    analyzer.analyze_candidates.return_value = []
    pool.get_continuous_leaders.return_value = []
    pool.get_new_entries.return_value = []
    pool.get_dropped_stocks.return_value = []

    monkeypatch.setattr(morning_report_service.settings, "admin_user_open_id", "ou_admin")

    service = MorningReportService(
        report_agent=agent,
        feishu_client=feishu,
        stock_screener=screener,
        strong_stock_analyzer=analyzer,
        observation_pool=pool,
    )

    assert service.send() is True

    agent.generate_morning_report.assert_called_once_with()
    feishu.send_markdown.assert_called_once()
    receive_id, content = feishu.send_markdown.call_args.args
    assert receive_id == "ou_admin"
    assert "### ① 市场总览" in content
    assert "市场整体震荡修复" in content
    assert "### ② 今日主线板块" in content
    assert "### ③ 龙头池（Top3）" in content
    assert "### ④ 潜力接力池" in content
    assert "### ⑤ 连续跟踪池" in content
    assert "### ⑥ 淘汰池" in content
    assert "### ⑦ 风险提示" in content
    assert "关注成交不足风险" in content
    assert "不应推送" not in content
    assert "自选股观察" not in content


def test_send_skips_without_receive_id(monkeypatch):
    agent = MagicMock()
    feishu = MagicMock()

    monkeypatch.setattr(morning_report_service.settings, "admin_user_open_id", "")

    service = MorningReportService(
        report_agent=agent,
        feishu_client=feishu,
        stock_screener=MagicMock(),
        strong_stock_analyzer=MagicMock(),
        observation_pool=MagicMock(),
    )

    assert service.send() is False
    agent.generate_morning_report.assert_not_called()
    feishu.send_markdown.assert_not_called()


def test_get_strong_stocks_returns_empty_when_no_candidates():
    screener = MagicMock()
    analyzer = MagicMock()
    pool = MagicMock()
    screener.screen_top_stocks.return_value = []
    analyzer.analyze_candidates.return_value = []
    pool.update_daily_picks.return_value = []
    service = MorningReportService(
        report_agent=MagicMock(),
        feishu_client=MagicMock(),
        stock_screener=screener,
        strong_stock_analyzer=analyzer,
        observation_pool=pool,
    )

    assert service.get_strong_stocks() == []
