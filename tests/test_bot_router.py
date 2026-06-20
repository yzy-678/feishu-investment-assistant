"""Feishu 事件回调路由测试

覆盖：URL 验证挑战、消息事件转发、健康检查、配置 API。
使用 FastAPI TestClient 模拟 HTTP 请求。
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.agents.base import AgentResponse, AgentType


@pytest.fixture
def client():
    """创建 FastAPI TestClient"""
    return TestClient(app)


class TestURLVerification:
    """URL 验证挑战测试"""

    def test_url_verification_basic(self, client):
        """基本 URL 验证"""
        resp = client.post("/feishu/event", json={
            "type": "url_verification",
            "challenge": "test_challenge_123",
            "token": "",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["challenge"] == "test_challenge_123"

    def test_url_verification_returned(self, client):
        """验证返回格式包含 challenge"""
        challenge = "unique_challenge_value"
        resp = client.post("/feishu/event", json={
            "type": "url_verification",
            "challenge": challenge,
        })
        assert resp.json()["challenge"] == challenge

    def test_url_verification_without_challenge(self, client):
        """缺少 challenge 时返回空字符串"""
        resp = client.post("/feishu/event", json={
            "type": "url_verification",
        })
        assert resp.json()["challenge"] == ""


class TestMessageEvent:
    """消息事件转发测试"""

    @patch("src.bot.router.get_handler")
    def test_message_event_forwarded(self, mock_get_handler, client):
        """消息事件应转发到 MessageHandler"""
        mock_handler = MagicMock()
        mock_get_handler.return_value = mock_handler

        event = {
            "schema": "2.0",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {"message_id": "om_test", "content": '{"text": "你好"}'},
                "sender": {"sender_id": {"open_id": "ou_test"}},
            },
        }
        resp = client.post("/feishu/event", json=event)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0
        mock_handler.handle_event.assert_called_once_with(event)

    @patch("src.bot.router.get_handler")
    def test_non_message_event_ignored(self, mock_get_handler, client):
        """非消息事件不触发 handler"""
        mock_handler = MagicMock()
        mock_get_handler.return_value = mock_handler

        event = {
            "header": {"event_type": "im.message.receive_v999"},
        }
        resp = client.post("/feishu/event", json=event)
        assert resp.status_code == 200
        mock_handler.handle_event.assert_not_called()

    @patch("src.bot.router.get_handler")
    def test_handler_exception_swallowed(self, mock_get_handler, client):
        """handler 异常不应返回 500"""
        mock_handler = MagicMock()
        mock_handler.handle_event.side_effect = Exception("Unexpected error")
        mock_get_handler.return_value = mock_handler

        event = {
            "schema": "2.0",
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {"message_id": "om_test", "content": '{"text": "hi"}'},
            },
        }
        resp = client.post("/feishu/event", json=event)
        assert resp.status_code == 200
        assert resp.json()["code"] == 0

    # Note: 无效 JSON 的测试由 Starlette/FastAPI 处理，不在本项目测试范围内


class TestHealth:
    """健康检查测试"""

    def test_health_endpoint(self, client):
        """健康检查返回 ok"""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAPIConfig:
    """内部配置 API 测试"""

    @patch("src.bot.router.get_config")
    def test_api_config_returns_config(self, mock_get_config, client):
        """配置 API 返回运行时配置"""
        mock_cfg = MagicMock()
        mock_cfg.get_enabled.return_value = True
        mock_cfg.get_market.return_value = "CN"
        mock_cfg.get_scan_interval.return_value = 1800
        mock_get_config.return_value = mock_cfg

        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["market"] == "CN"
        assert data["scan_interval"] == 1800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
