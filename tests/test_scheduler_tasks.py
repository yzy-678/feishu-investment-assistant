"""Scheduler CLI tests."""

from unittest.mock import MagicMock

from src.scheduler import tasks


class TestResolveAdminOpenId:
    def test_runtime_config_takes_priority(self, monkeypatch):
        config = MagicMock()
        config.get_value.return_value = "ou_runtime"

        monkeypatch.setattr(tasks.settings, "admin_user_open_id", "ou_env")

        assert tasks._resolve_admin_open_id(config) == "ou_runtime"

    def test_settings_fallback_used_when_runtime_missing(self, monkeypatch):
        config = MagicMock()
        config.get_value.return_value = None

        monkeypatch.setattr(tasks.settings, "admin_user_open_id", "ou_env")

        assert tasks._resolve_admin_open_id(config) == "ou_env"


class TestMain:
    def test_main_sends_report_to_settings_fallback(self, monkeypatch):
        config = MagicMock()
        config.get_enabled.return_value = True
        config.get_market.return_value = "CN"
        config.get_value.return_value = None

        agent = MagicMock()
        agent.generate_report.return_value = "report body"

        feishu = MagicMock()

        monkeypatch.setattr(tasks.settings, "admin_user_open_id", "ou_env")
        monkeypatch.setattr("src.db.init_database", lambda: None)
        monkeypatch.setattr("src.config.manager.get_config", lambda: config)
        monkeypatch.setattr("src.agents.report_agent.get_report_agent", lambda: agent)
        monkeypatch.setattr("src.bot.client.get_feishu_client", lambda: feishu)
        monkeypatch.setattr("sys.argv", ["tasks.py", "morning"])

        result = tasks.main()

        assert result == 0
        feishu.send_text.assert_called_once_with("ou_env", "report body")

    def test_main_skips_send_when_no_admin_open_id(self, monkeypatch):
        config = MagicMock()
        config.get_enabled.return_value = True
        config.get_market.return_value = "CN"
        config.get_value.return_value = ""

        agent = MagicMock()
        agent.generate_report.return_value = "report body"

        feishu = MagicMock()

        monkeypatch.setattr(tasks.settings, "admin_user_open_id", "")
        monkeypatch.setattr("src.db.init_database", lambda: None)
        monkeypatch.setattr("src.config.manager.get_config", lambda: config)
        monkeypatch.setattr("src.agents.report_agent.get_report_agent", lambda: agent)
        monkeypatch.setattr("src.bot.client.get_feishu_client", lambda: feishu)
        monkeypatch.setattr("sys.argv", ["tasks.py", "morning"])

        result = tasks.main()

        assert result == 0
        feishu.send_text.assert_not_called()

    def test_main_scan_sends_deliverable_alerts(self, monkeypatch):
        config = MagicMock()
        config.get_enabled.return_value = True
        config.get_market.return_value = "CN"
        config.get_value.return_value = None

        alert_agent = MagicMock()
        event = MagicMock()
        event.related_code = "000001"
        event.title = "平安银行 放量拉升"
        event.strength = 8.1
        event.event_id = "price_spike:000001"
        alert_agent.scan_watchlist.return_value = {
            "scanned": 2,
            "triggered": 1,
            "deliverable": 1,
            "message": "扫描完成，发现 1 条可推送预警。",
            "alerts": [
                {"event": event, "should_send": True, "reason": "new_event"},
            ],
        }

        feishu = MagicMock()

        monkeypatch.setattr(tasks.settings, "admin_user_open_id", "ou_env")
        monkeypatch.setattr("src.db.init_database", lambda: None)
        monkeypatch.setattr("src.config.manager.get_config", lambda: config)
        monkeypatch.setattr("src.agents.alert_agent.get_alert_agent", lambda: alert_agent)
        monkeypatch.setattr("src.bot.client.get_feishu_client", lambda: feishu)
        monkeypatch.setattr("sys.argv", ["tasks.py", "scan"])

        result = tasks.main()

        assert result == 0
        feishu.send_text.assert_called_once()
        alert_agent.mark_delivered.assert_called_once_with(["price_spike:000001"])

    def test_main_scan_skips_send_without_deliverable_alerts(self, monkeypatch):
        config = MagicMock()
        config.get_enabled.return_value = True
        config.get_market.return_value = "CN"
        config.get_value.return_value = None

        alert_agent = MagicMock()
        alert_agent.scan_watchlist.return_value = {
            "scanned": 2,
            "triggered": 0,
            "deliverable": 0,
            "message": "未发现新的盘中预警。",
            "alerts": [],
        }

        feishu = MagicMock()

        monkeypatch.setattr(tasks.settings, "admin_user_open_id", "ou_env")
        monkeypatch.setattr("src.db.init_database", lambda: None)
        monkeypatch.setattr("src.config.manager.get_config", lambda: config)
        monkeypatch.setattr("src.agents.alert_agent.get_alert_agent", lambda: alert_agent)
        monkeypatch.setattr("src.bot.client.get_feishu_client", lambda: feishu)
        monkeypatch.setattr("sys.argv", ["tasks.py", "scan"])

        result = tasks.main()

        assert result == 0
        feishu.send_text.assert_not_called()
