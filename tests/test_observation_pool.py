"""ObservationPoolManager tests."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from src.db import init_database
from src.db.models import ObservationStatus
from src.market.observation_pool import (
    ObservationPoolManager,
    get_observation_pool_manager,
)
from src.market.strong_stock_analyzer import StrongStockPick
from src.reports.morning_report_service import MorningReportService


@pytest.fixture(autouse=True)
def reset_observation_pool(monkeypatch):
    init_database()
    monkeypatch.setattr(
        "src.market.observation_pool.shanghai_today",
        lambda: date(2026, 6, 28),
    )
    manager = get_observation_pool_manager()
    manager.clear()
    return manager


def make_pick(
    symbol,
    name,
    rank=1,
    score=90.0,
    reason="量价齐升",
    industry="半导体",
):
    return StrongStockPick(
        symbol=symbol,
        name=name,
        industry=industry,
        score=score,
        rank=rank,
        reason=reason,
        risk="冲高回落风险",
        watch_points="观察成交额延续性",
        data_source="StockScreener",
        data_time="2026-06-28 08:30:00",
    )


def test_new_stock_first_enters_pool(reset_observation_pool):
    manager = reset_observation_pool

    manager.update_daily_picks([make_pick("300001", "测试科技")])

    active = manager.get_active_pool()
    assert len(active) == 1
    assert active[0].symbol == "300001"
    assert active[0].first_seen == "2026-06-28"
    assert active[0].last_seen == "2026-06-28"
    assert active[0].consecutive_days == 1
    assert active[0].status == ObservationStatus.ACTIVE


def test_consecutive_days_increase(reset_observation_pool, monkeypatch):
    manager = reset_observation_pool
    manager.update_daily_picks([make_pick("300001", "测试科技", score=88.0)])

    monkeypatch.setattr(
        "src.market.observation_pool.shanghai_today",
        lambda: date(2026, 6, 29),
    )
    manager.update_daily_picks([
        make_pick("300001", "测试科技", score=92.0, reason="平台突破"),
    ])

    entry = manager.get_active_pool()[0]
    assert entry.consecutive_days == 2
    assert entry.highest_score == 92.0
    assert entry.latest_score == 92.0
    assert entry.latest_reason == "平台突破"


def test_dropped_stock_when_missing_today(reset_observation_pool, monkeypatch):
    manager = reset_observation_pool
    manager.update_daily_picks([
        make_pick("300001", "测试科技"),
        make_pick("300002", "测试材料", rank=2),
    ])

    monkeypatch.setattr(
        "src.market.observation_pool.shanghai_today",
        lambda: date(2026, 6, 29),
    )
    manager.update_daily_picks([make_pick("300001", "测试科技")])

    dropped = manager.get_dropped_stocks()
    assert [item.symbol for item in dropped] == ["300002"]
    assert dropped[0].status == ObservationStatus.DROPPED
    assert dropped[0].consecutive_days == 1


def test_get_active_pool(reset_observation_pool, monkeypatch):
    manager = reset_observation_pool
    manager.update_daily_picks([
        make_pick("300001", "测试科技"),
        make_pick("300002", "测试材料", rank=2),
    ])

    monkeypatch.setattr(
        "src.market.observation_pool.shanghai_today",
        lambda: date(2026, 6, 29),
    )
    manager.update_daily_picks([make_pick("300001", "测试科技")])

    active = manager.get_active_pool()
    assert [item.symbol for item in active] == ["300001"]


def test_get_continuous_leaders(reset_observation_pool, monkeypatch):
    manager = reset_observation_pool
    manager.update_daily_picks([
        make_pick("300001", "测试科技"),
        make_pick("300002", "测试材料", rank=2),
    ])

    monkeypatch.setattr(
        "src.market.observation_pool.shanghai_today",
        lambda: date(2026, 6, 29),
    )
    manager.update_daily_picks([make_pick("300001", "测试科技")])

    leaders = manager.get_continuous_leaders(min_days=2)
    assert [item.symbol for item in leaders] == ["300001"]


def test_get_new_entries(reset_observation_pool, monkeypatch):
    manager = reset_observation_pool
    manager.update_daily_picks([make_pick("300001", "测试科技")])

    monkeypatch.setattr(
        "src.market.observation_pool.shanghai_today",
        lambda: date(2026, 6, 29),
    )
    manager.update_daily_picks([
        make_pick("300001", "测试科技"),
        make_pick("300003", "测试机器人", rank=2),
    ])

    new_entries = manager.get_new_entries()
    assert [item.symbol for item in new_entries] == ["300003"]


def test_get_dropped_stocks(reset_observation_pool, monkeypatch):
    manager = reset_observation_pool
    manager.update_daily_picks([make_pick("300001", "测试科技")])

    monkeypatch.setattr(
        "src.market.observation_pool.shanghai_today",
        lambda: date(2026, 6, 29),
    )
    manager.update_daily_picks([])

    dropped = manager.get_dropped_stocks()
    assert [item.symbol for item in dropped] == ["300001"]


def test_morning_report_service_displays_observation_pool_tracking(monkeypatch):
    agent = MagicMock()
    agent.generate_morning_report.return_value = (
        "## 【早报】2026-06-28\n\n"
        "### 市场概览\n市场震荡。\n\n"
        "### 热点板块\n半导体活跃。\n\n"
        "### 风险提示\n注意缩量。"
    )
    feishu = MagicMock()
    screener = MagicMock()
    analyzer = MagicMock()
    pool = MagicMock()
    picks = [make_pick("300001", "测试科技")]
    screener.screen_top_stocks.return_value = []
    analyzer.analyze_candidates.return_value = picks
    pool.update_daily_picks.return_value = []
    pool.get_continuous_leaders.return_value = [
        make_pool_entry("300001", "测试科技", consecutive_days=3)
    ]
    pool.get_new_entries.return_value = [
        make_pool_entry("300002", "测试机器人")
    ]
    pool.get_dropped_stocks.return_value = [
        make_pool_entry("300003", "测试材料", status=ObservationStatus.DROPPED)
    ]

    monkeypatch.setattr(
        "src.reports.morning_report_service.settings.admin_user_open_id",
        "ou_admin",
    )
    service = MorningReportService(
        report_agent=agent,
        feishu_client=feishu,
        stock_screener=screener,
        strong_stock_analyzer=analyzer,
        observation_pool=pool,
    )

    assert service.send() is True

    pool.update_daily_picks.assert_called_once_with(picks)
    content = feishu.send_markdown.call_args.args[1]
    assert "### ⑤ 连续跟踪池" in content
    assert "测试科技：连续3天上榜" in content
    assert "### ⑥ 淘汰池" in content
    assert "测试材料：今日跌出观察池" in content


def make_pool_entry(
    symbol,
    name,
    consecutive_days=1,
    status=ObservationStatus.ACTIVE,
):
    from src.db.models import ObservationPoolEntry

    return ObservationPoolEntry(
        symbol=symbol,
        name=name,
        industry="半导体",
        first_seen="2026-06-28",
        last_seen="2026-06-28",
        consecutive_days=consecutive_days,
        highest_score=90.0,
        latest_score=90.0,
        latest_rank=1,
        latest_reason="量价齐升",
        status=status,
    )
