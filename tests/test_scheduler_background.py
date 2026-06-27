"""Background APScheduler tests."""

import logging
from unittest.mock import MagicMock

import pytest

from src.scheduler import background


class FakeScheduler:
    instances = []

    def __init__(self, timezone):
        self.timezone = timezone
        self.jobs = []
        self.running = False
        self.shutdown_called = False
        FakeScheduler.instances.append(self)

    def add_job(self, func, trigger, **kwargs):
        self.jobs.append({
            "func": func,
            "trigger": trigger,
            **kwargs,
        })

    def start(self):
        self.running = True

    def shutdown(self, wait=False):
        self.shutdown_called = True
        self.shutdown_wait = wait
        self.running = False


@pytest.fixture(autouse=True)
def reset_scheduler(monkeypatch):
    FakeScheduler.instances.clear()
    monkeypatch.setattr(background, "_scheduler", None)
    yield
    monkeypatch.setattr(background, "_scheduler", None)


def test_start_scheduler_registers_daily_report_jobs(monkeypatch):
    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background.settings, "timezone", "Asia/Shanghai")
    monkeypatch.setattr(background, "AsyncIOScheduler", FakeScheduler)

    background.start_scheduler()

    scheduler = background.get_scheduler()
    assert scheduler is FakeScheduler.instances[0]
    assert scheduler.running is True
    assert len(scheduler.jobs) == 3

    jobs_by_id = {job["id"]: job for job in scheduler.jobs}
    assert jobs_by_id["daily_report_morning"]["trigger"] == "cron"
    assert jobs_by_id["daily_report_morning"]["hour"] == 8
    assert jobs_by_id["daily_report_morning"]["minute"] == 30
    assert jobs_by_id["daily_report_morning"]["args"] == ["morning"]

    assert jobs_by_id["daily_report_noon"]["hour"] == 12
    assert jobs_by_id["daily_report_noon"]["minute"] == 0
    assert jobs_by_id["daily_report_noon"]["args"] == ["noon"]

    assert jobs_by_id["daily_report_closing"]["hour"] == 15
    assert jobs_by_id["daily_report_closing"]["minute"] == 30
    assert jobs_by_id["daily_report_closing"]["args"] == ["closing"]


def test_start_scheduler_skips_when_disabled(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.scheduler.background")
    monkeypatch.setattr(background.settings, "daily_report_enabled", False)
    monkeypatch.setattr(background, "AsyncIOScheduler", FakeScheduler)

    background.start_scheduler()

    assert background.get_scheduler() is None
    assert FakeScheduler.instances == []
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "report_type=all" in logs
    assert "send_status=disabled" in logs


def test_start_scheduler_is_idempotent(monkeypatch):
    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background, "AsyncIOScheduler", FakeScheduler)

    background.start_scheduler()
    background.start_scheduler()

    assert len(FakeScheduler.instances) == 1


def test_stop_scheduler_shutdowns_existing_scheduler(monkeypatch):
    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background, "AsyncIOScheduler", FakeScheduler)
    background.start_scheduler()
    scheduler = background.get_scheduler()

    background.stop_scheduler()

    assert scheduler.shutdown_called is True
    assert scheduler.shutdown_wait is False
    assert background.get_scheduler() is None


@pytest.mark.parametrize(
    ("report_type", "method_name", "content"),
    [
        ("morning", "generate_morning_report", "早报内容"),
        ("noon", "generate_noon_report", "午间观察内容"),
        ("closing", "generate_closing_report", "收盘复盘内容"),
    ],
)
def test_send_daily_report_uses_report_agent_and_send_markdown(
    monkeypatch,
    report_type,
    method_name,
    content,
):
    agent = MagicMock()
    getattr(agent, method_name).return_value = content
    feishu = MagicMock()

    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background.settings, "admin_user_open_id", "ou_admin")
    monkeypatch.setattr(background, "get_report_agent", lambda: agent)
    monkeypatch.setattr(background, "get_feishu_client", lambda: feishu)

    result = background.send_daily_report(report_type)

    assert result is True
    getattr(agent, method_name).assert_called_once_with()
    feishu.send_markdown.assert_called_once_with("ou_admin", content)


def test_send_daily_report_skips_without_admin_open_id(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.scheduler.background")
    agent = MagicMock()
    feishu = MagicMock()

    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background.settings, "admin_user_open_id", "")
    monkeypatch.setattr(background, "get_report_agent", lambda: agent)
    monkeypatch.setattr(background, "get_feishu_client", lambda: feishu)

    result = background.send_daily_report("morning")

    assert result is False
    agent.generate_morning_report.assert_not_called()
    feishu.send_markdown.assert_not_called()
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "report_type=morning" in logs
    assert "send_status=skipped" in logs
    assert "admin_user_open_id is empty" in logs


def test_send_daily_report_logs_failure(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.scheduler.background")
    agent = MagicMock()
    agent.generate_morning_report.side_effect = RuntimeError("DeepSeek timeout")

    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background.settings, "admin_user_open_id", "ou_admin")
    monkeypatch.setattr(background, "get_report_agent", lambda: agent)

    result = background.send_daily_report("morning")

    assert result is False
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "report_type=morning" in logs
    assert "send_status=failed" in logs
    assert "error_message=DeepSeek timeout" in logs
