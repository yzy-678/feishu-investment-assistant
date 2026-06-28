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
    assert len(scheduler.jobs) == 1

    jobs_by_id = {job["id"]: job for job in scheduler.jobs}
    assert jobs_by_id["daily_report_morning"]["trigger"] == "cron"
    assert jobs_by_id["daily_report_morning"]["hour"] == 8
    assert jobs_by_id["daily_report_morning"]["minute"] == 30
    assert jobs_by_id["daily_report_morning"]["func"] is background.send_morning_report


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


def test_send_morning_report_uses_morning_report_service(monkeypatch):
    service = MagicMock()
    service.send.return_value = True

    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background, "get_morning_report_service", lambda: service)

    result = background.send_morning_report()

    assert result is True
    service.send.assert_called_once_with()


def test_send_morning_report_logs_service_skip(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.scheduler.background")
    service = MagicMock()
    service.send.return_value = False

    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background, "get_morning_report_service", lambda: service)

    result = background.send_morning_report()

    assert result is False
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "report_type=morning" in logs
    assert "send_status=skipped" in logs
    assert "admin_user_open_id is empty" in logs


def test_send_morning_report_logs_failure(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="src.scheduler.background")
    service = MagicMock()
    service.send.side_effect = RuntimeError("DeepSeek timeout")

    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background, "get_morning_report_service", lambda: service)

    result = background.send_morning_report()

    assert result is False
    logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "report_type=morning" in logs
    assert "send_status=failed" in logs
    assert "error_message=DeepSeek timeout" in logs


def test_send_daily_report_only_keeps_morning_compatible(monkeypatch):
    service = MagicMock()
    service.send.return_value = True

    monkeypatch.setattr(background.settings, "daily_report_enabled", True)
    monkeypatch.setattr(background, "get_morning_report_service", lambda: service)

    assert background.send_daily_report("morning") is True
    assert background.send_daily_report("noon") is False
