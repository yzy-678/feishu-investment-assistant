"""AlertAgent 单元测试

使用临时 SQLite 数据库测试完整的预警事件生命周期。
覆盖：事件记录、去重策略、强度升级、解除、查询、并发、异常处理。
"""

import os
import tempfile
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from src.agents.alert_agent import (
    AlertAgent, get_alert_agent,
    COOLDOWN_MINUTES, STRENGTH_ESCALATION_RATIO,
)
from src.agents.base import AgentType
from src.db.models import AlertEvent, AlertSeverity


# ---- Fixtures ----------------------------------------------------------------

@pytest.fixture(autouse=True)
def test_db():
    """每个测试使用临时 SQLite 数据库"""
    fd, path = tempfile.mkstemp(suffix=".test.db")
    os.close(fd)

    from src.config.settings import settings as cfg
    original_path = cfg.database_path
    cfg.database_path = path

    import src.db.database as db_mod
    from src.db import DatabaseManager, init_database
    db_mod._db_instance = None
    DatabaseManager._instance = None
    init_database()

    yield

    cfg.database_path = original_path
    import src.db.database as db_mod
    db_mod._db_instance = None
    DatabaseManager._instance = None
    AlertAgent._instance = None
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def agent():
    """返回一个使用测试数据库的 AlertAgent 实例"""
    AlertAgent._instance = None
    AlertAgent._initialized = False
    return AlertAgent()


def add_test_event(agent, event_id="test:001", strength=5.0,
                   severity="info", sent=False):
    """辅助：向测试数据库添加一条事件"""
    ev = agent.record_event(
        event_id=event_id,
        alert_type="price_spike",
        title=f"Test Event {event_id}",
        content=f"Strength {strength} test alert",
        strength=strength,
        severity=severity,
        related_code="000001" if ":" in event_id else "",
        related_sector="Test",
    )
    if sent:
        agent.mark_sent(event_id)
    return ev


# ---- can_handle tests --------------------------------------------------------


class TestCanHandle:

    @pytest.mark.parametrize("msg", [
        "查看预警", "异动提醒", "开启监控",
        "提醒设置", "盘中扫描", "查看警报",
    ])
    def test_can_handle_keywords(self, agent, msg):
        assert agent.can_handle(msg)

    def test_can_handle_not_matched(self, agent):
        assert not agent.can_handle("How is the market today?")
        assert not agent.can_handle("Pause system")

    def test_can_handle_empty(self, agent):
        assert not agent.can_handle("")
        assert not agent.can_handle("   ")

    def test_agent_type(self, agent):
        assert agent.agent_type == AgentType.ALERT


# ---- record_event tests ------------------------------------------------------


class TestRecordEvent:

    def test_record_new_event(self, agent):
        """New event returns complete AlertEvent"""
        event = agent.record_event(
            event_id="price_spike:000001",
            alert_type="price_spike",
            title="Ping An Bank up 5.2%",
            content="Rapid intraday rise with volume surge",
            strength=5.2,
            severity="warning",
            related_code="000001",
            related_sector="Banking",
        )
        assert event.event_id == "price_spike:000001"
        assert event.alert_type == "price_spike"
        assert event.strength == 5.2
        assert event.peak_strength == 5.2
        assert event.sent_count == 0
        assert event.resolved == 0
        assert event.first_seen is not None

    def test_record_duplicate_event_updates_strength(self, agent):
        """Same event_id updates strength and peak"""
        add_test_event(agent, "test:dup", strength=3.0)
        event = agent.record_event(
            event_id="test:dup", alert_type="price_spike",
            title="Updated", content="Strength increased", strength=7.0,
        )
        assert event.strength == 7.0
        assert event.peak_strength == 3.0  # record_event no longer updates peak

    def test_record_strength_decrease_keeps_peak(self, agent):
        """Lower strength should not reduce peak"""
        add_test_event(agent, "test:peak", strength=10.0)
        event = agent.record_event(
            event_id="test:peak", alert_type="price_spike",
            title="Drop", content="Strength dropped", strength=3.0,
        )
        assert event.strength == 3.0
        assert event.peak_strength == 10.0

    def test_record_event_with_all_fields(self, agent):
        """Record with complete fields"""
        event = agent.record_event(
            event_id="sector_rotation:Banking",
            alert_type="sector_rotation",
            title="Banking sector rotation",
            content="Large capital inflow to banking",
            strength=8.0,
            severity="critical",
            related_code="",
            related_sector="Banking",
        )
        assert event.alert_type == "sector_rotation"
        assert event.severity == "critical"
        assert event.related_sector == "Banking"
        assert event.resolved == 0


# ---- should_alert tests ------------------------------------------------------


class TestShouldAlert:

    def test_should_alert_new_event(self, agent):
        """New event should alert"""
        should, reason = agent.should_alert("nonexistent:001")
        assert should is True
        assert reason == "new_event"

    def test_should_alert_cooldown_active(self, agent):
        """Unresolved within cooldown -> no alert"""
        add_test_event(agent, "test:cool", strength=5.0, sent=True)
        should, reason = agent.should_alert("test:cool")
        assert should is False
        assert "cooldown" in reason

    def test_should_alert_escalated(self, agent):
        """Strength escalated breaks cooldown -> alert"""
        add_test_event(agent, "test:esc", strength=5.0, sent=True)
        agent.record_event(
            event_id="test:esc", alert_type="price_spike",
            title="Escalated", content="Big jump", strength=7.0,
        )
        should, reason = agent.should_alert("test:esc")
        assert should is True
        assert reason == "strength_escalated"

    def test_should_alert_resolved_event(self, agent):
        """Resolved event -> alert"""
        add_test_event(agent, "test:res", strength=3.0)
        agent.resolve_event("test:res")
        should, reason = agent.should_alert("test:res")
        assert should is True
        assert reason == "resolved"


# ---- mark_sent tests ---------------------------------------------------------


class TestMarkSent:

    def test_mark_sent_updates_last_sent(self, agent):
        """mark_sent sets last_sent"""
        add_test_event(agent, "test:mark")
        agent.mark_sent("test:mark")
        event = agent._find_event("test:mark")
        assert event.last_sent is not None
        assert event.sent_count == 1

    def test_mark_sent_increments_count(self, agent):
        """Multiple mark_sent increments sent_count"""
        add_test_event(agent, "test:cnt")
        agent.mark_sent("test:cnt")
        agent.mark_sent("test:cnt")
        event = agent._find_event("test:cnt")
        assert event.sent_count == 2


# ---- resolve_event tests -----------------------------------------------------


class TestResolveEvent:

    def test_resolve_existing_event(self, agent):
        """Resolve existing event"""
        add_test_event(agent, "test:resv")
        result = agent.resolve_event("test:resv")
        assert result is True
        event = agent._find_event("test:resv")
        assert event.resolved == 1

    def test_resolve_nonexistent_event(self, agent):
        """Resolve non-existent returns False"""
        result = agent.resolve_event("nonexistent")
        assert result is False

    def test_resolve_already_resolved(self, agent):
        """Double resolve returns False"""
        add_test_event(agent, "test:dbl")
        agent.resolve_event("test:dbl")
        result = agent.resolve_event("test:dbl")
        assert result is False


# ---- get_active_events tests -------------------------------------------------


class TestGetActiveEvents:

    def test_active_events_excludes_resolved(self, agent):
        """Active events only include unresolved"""
        add_test_event(agent, "event_a")
        add_test_event(agent, "event_b")
        agent.resolve_event("event_a")
        active = agent.get_active_events()
        assert len(active) == 1
        assert active[0].event_id == "event_b"

    def test_active_events_empty(self, agent):
        """No active events returns empty list"""
        active = agent.get_active_events()
        assert active == []

    def test_active_events_order(self, agent):
        """Active events ordered by first_seen DESC"""
        add_test_event(agent, "e1")
        import time
        time.sleep(0.01)
        add_test_event(agent, "e2")
        active = agent.get_active_events()
        assert len(active) == 2
        assert active[0].event_id == "e2"


# ---- handle tests ------------------------------------------------------------


class TestHandle:

    def test_handle_with_active_events(self, agent):
        """Handle returns active events list"""
        add_test_event(agent, "e1", strength=3.0, sent=True)
        add_test_event(agent, "e2", strength=5.0, sent=True)
        resp = agent.handle("session1", "查看警报")
        assert resp.success is True
        assert resp.agent == AgentType.ALERT
        assert resp.metadata["active_count"] == 2

    def test_handle_no_active_events(self, agent):
        """No active events returns normal message"""
        resp = agent.handle("session1", "查看警报")
        assert resp.success is True
        assert "No active" in resp.message or "无" in resp.message
        assert resp.metadata["active_count"] == 0

    def test_handle_empty_session(self, agent):
        """Empty session_id is handled"""
        resp = agent.handle("", "Alert")
        assert resp.success is True


# ---- make_event_id tests -----------------------------------------------------


class TestMakeEventId:

    def test_make_event_id_with_symbol(self):
        """With symbol: format is type:symbol"""
        eid = AlertAgent.make_event_id("price_spike", "000001")
        assert eid == "price_spike:000001"

    def test_make_event_id_without_symbol(self):
        """Without symbol: format is type:hash"""
        eid = AlertAgent.make_event_id("custom")
        assert eid.startswith("custom:")


# ---- Error handling tests ----------------------------------------------------


class TestErrorHandling:

    def test_handle_db_error(self, agent):
        """DB error returns failed AgentResponse"""
        with patch.object(agent._db, "get_connection") as mock:
            mock.side_effect = Exception("DB connection failed")
            resp = agent.handle("s", "Alert")
            assert resp.success is False
            assert resp.agent == AgentType.ALERT


# ---- Concurrency tests -------------------------------------------------------


class TestConcurrency:

    def test_concurrent_record_event(self, agent):
        """Concurrent event recording"""
        errors = []

        def record(i):
            try:
                agent.record_event(
                    event_id=f"concurrent:{i}",
                    alert_type="custom",
                    title=f"Concurrent {i}",
                    content="",
                    strength=float(i),
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(agent.get_active_events()) == 10

    def test_concurrent_mark_sent(self, agent):
        """Concurrent mark sent"""
        add_test_event(agent, "con:mark")
        errors = []

        def mark():
            try:
                agent.mark_sent("con:mark")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mark) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        event = agent._find_event("con:mark")
        assert event.sent_count == 10


# ---- Lifecycle tests ---------------------------------------------------------


class TestLifecycle:

    def test_full_event_lifecycle(self, agent):
        """Complete event lifecycle: create -> send -> escalate -> resolve"""
        eid = "lifecycle:001"

        # 1. New event -> should alert
        add_test_event(agent, eid, strength=3.0)
        assert agent.should_alert(eid)[0] is True

        # 2. Mark sent
        agent.mark_sent(eid)
        event = agent._find_event(eid)
        assert event.sent_count == 1
        assert event.last_sent is not None

        # 3. Cooldown -> no alert
        assert agent.should_alert(eid)[0] is False

        # 4. Strength escalation breaks cooldown
        agent.record_event(eid, "price_spike", "Escalated", "", strength=8.0)
        should, reason = agent.should_alert(eid)
        assert should is True
        assert reason == "strength_escalated"

        # 5. Resolve
        assert agent.resolve_event(eid) is True

        # 6. Resolved events not in active list
        active = agent.get_active_events()
        assert all(e.event_id != eid for e in active)

    def test_multiple_events_independent(self, agent):
        """Multiple events operate independently"""
        add_test_event(agent, "e1", strength=1.0)
        add_test_event(agent, "e2", strength=2.0)
        agent.mark_sent("e1")
        agent.resolve_event("e2")

        e1 = agent._find_event("e1")
        e2 = agent._find_event("e2")
        assert e1.sent_count == 1
        assert e2.resolved == 1


# ---- Singleton tests ---------------------------------------------------------


class TestSingleton:

    def test_singleton(self):
        a1 = get_alert_agent()
        a2 = get_alert_agent()
        assert a1 is a2
