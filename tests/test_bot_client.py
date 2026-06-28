"""FeishuClient 单元测试

测试覆盖：Token 获取/缓存/刷新、消息发送/回复、Markdown 转换、
健康检查、异常处理、重试、超时、并发。
所有 HTTP 调用使用 mock。
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.bot import FeishuClient, FeishuError, get_feishu_client
from src.bot.text_utils import sanitize_text


# ── Mock 辅助函数 ──────────────────────────────────────


def _mock_response(status_code=200, json_data=None):
    """创建模拟 httpx 响应"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def token_response(token="mock_token_abc", expire=7200):
    return _mock_response(json_data={
        "code": 0, "msg": "ok",
        "tenant_access_token": token,
        "expire": expire,
    })


def send_response(message_id="om_mock_msg_id"):
    return _mock_response(json_data={
        "code": 0, "msg": "ok",
        "data": {"message_id": message_id},
    })


def error_response(code=99991663, msg="invalid auth"):
    return _mock_response(json_data={
        "code": code, "msg": msg,
    })


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_singleton():
    FeishuClient._instance = None
    FeishuClient._initialized = False


@pytest.fixture
def client():
    c = FeishuClient()
    c.app_id = "cli_test_app_id"
    c.app_secret = "test_secret_abc"
    return c


# ── Token 管理测试 ─────────────────────────────────────


class TestToken:

    @patch("httpx.Client")
    def test_get_token_success(self, mock_client_class, client):
        """成功获取 token"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.return_value = token_response()
        mock_client_class.return_value = mock_cli

        token = client._ensure_token()
        assert token == "mock_token_abc"

    @patch("httpx.Client")
    def test_token_caching(self, mock_client_class, client):
        """第二次调用不触发网络请求"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.return_value = token_response("token_1")
        mock_client_class.return_value = mock_cli

        token1 = client._ensure_token()
        token2 = client._ensure_token()

        assert token1 == token2 == "token_1"
        # 网络请求只应调用一次
        assert mock_cli.post.call_count == 1

    @patch("httpx.Client")
    def test_token_refresh_when_expired(self, mock_client_class, client):
        """token 过期后自动刷新"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli

        # 第一次返回短过期 token
        mock_cli.post.side_effect = [
            token_response("token_a", expire=10),
            token_response("token_b", expire=7200),
        ]
        mock_client_class.return_value = mock_cli

        t1 = client._ensure_token()
        assert t1 == "token_a"

        # 让 token 过期
        client._expires_at = time.time() - 1

        t2 = client._ensure_token()
        assert t2 == "token_b"
        assert mock_cli.post.call_count == 2

    @patch("httpx.Client")
    def test_token_refresh_failure(self, mock_client_class, client):
        """获取 token 失败抛异常"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.return_value = error_response(10003, "invalid app")
        mock_client_class.return_value = mock_cli

        with pytest.raises(FeishuError, match="获取 token 失败"):
            client._ensure_token()


# ── send_text 测试 ─────────────────────────────────────


class TestSendText:

    @patch("httpx.Client")
    def test_send_text_success(self, mock_client_class, client):
        """发送文本消息成功"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        # Token request
        mock_cli.post.side_effect = [
            token_response(),               # token
            send_response("om_123"),         # send
        ]
        mock_client_class.return_value = mock_cli

        result = client.send_text("ou_user123", "Hello 飞书")
        assert result["data"]["message_id"] == "om_123"

    @patch("httpx.Client")
    def test_send_text_body_format(self, mock_client_class, client):
        """验证 request body 格式"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            send_response(),
        ]
        mock_client_class.return_value = mock_cli

        client.send_text("ou_test", "测试内容")

        # 验证发送请求
        send_call = mock_cli.post.call_args_list[1]
        sent_json = send_call.kwargs["json"]
        assert sent_json["receive_id"] == "ou_test"
        assert sent_json["msg_type"] == "text"
        # content 应为 JSON 字符串
        assert json.loads(sent_json["content"])["text"] == "测试内容"
        assert '"text": "测试内容"' in sent_json["content"]
        assert "\\u6d4b" not in sent_json["content"]

    @patch("httpx.Client")
    def test_send_text_sanitizes_without_breaking_chinese(
        self,
        mock_client_class,
        client,
    ):
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            send_response(),
        ]
        mock_client_class.return_value = mock_cli

        client.send_text("ou_test", "有研新材\u200b")

        send_call = mock_cli.post.call_args_list[1]
        sent_json = send_call.kwargs["json"]
        assert json.loads(sent_json["content"])["text"] == "有研新材"
        bad_words = ["失" + "十", "促" + "B", "丰" + "制", chr(0x5F0A)]
        assert not any(word in sent_json["content"] for word in bad_words)

    @patch("httpx.Client")
    def test_send_text_chat_id_receive_type(self, mock_client_class, client):
        """普通群消息应支持 receive_id_type=chat_id。"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            send_response(),
        ]
        mock_client_class.return_value = mock_cli

        client.send_text("oc_chat", "测试内容", receive_id_type="chat_id")

        send_call = mock_cli.post.call_args_list[1]
        assert "receive_id_type=chat_id" in send_call.args[0]
        assert send_call.kwargs["json"]["receive_id"] == "oc_chat"

    @patch("httpx.Client")
    def test_send_text_api_error(self, mock_client_class, client):
        """API 返回错误码"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            error_response(99991663, "invalid auth"),
        ]
        mock_client_class.return_value = mock_cli

        with pytest.raises(FeishuError, match="99991663"):
            client.send_text("ou_test", "test")


# ── reply_text 测试 ─────────────────────────────────────


class TestReplyText:

    @patch("httpx.Client")
    def test_reply_text_success(self, mock_client_class, client):
        """回复消息成功"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            send_response("om_reply"),
        ]
        mock_client_class.return_value = mock_cli

        result = client.reply_text("om_original", "回复你")
        assert result["data"]["message_id"] == "om_reply"

    @patch("httpx.Client")
    def test_reply_text_format(self, mock_client_class, client):
        """验证 reply body 格式"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            send_response(),
        ]
        mock_client_class.return_value = mock_cli

        client.reply_text("om_original", "回复内容")
        call = mock_cli.post.call_args_list[1]
        assert "messages/om_original/reply" in call.args[0]
        assert call.kwargs["json"]["msg_type"] == "text"


def test_sanitize_text_keeps_normal_chinese_and_failure_message():
    assert sanitize_text("有研新材") == "有研新材"
    assert (
        sanitize_text("实时数据获取失败，暂时无法分析")
        == "实时数据获取失败，暂时无法分析"
    )
    assert sanitize_text("有研新材\u200b") == "有研新材"


# ── send_markdown 测试 ─────────────────────────────────


class TestSendMarkdown:

    @patch("httpx.Client")
    def test_send_markdown_success(self, mock_client_class, client):
        """发送 Markdown 消息成功"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            send_response("om_md"),
        ]
        mock_client_class.return_value = mock_cli

        result = client.send_markdown("ou_test", "**粗体**文本")
        assert result["data"]["message_id"] == "om_md"

    @patch("httpx.Client")
    def test_send_markdown_body_format(self, mock_client_class, client):
        """验证 Markdown 请求为 post 类型"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            send_response(),
        ]
        mock_client_class.return_value = mock_cli

        client.send_markdown("ou_test", "测试消息")
        call = mock_cli.post.call_args_list[1]
        assert call.kwargs["json"]["msg_type"] == "post"

    def test_parse_bold(self, client):
        """**bold** 解析为 bold tag"""
        tags = FeishuClient._parse_md_line("这是**粗体**文本")
        assert len(tags) >= 3
        assert tags[1]["tag"] == "bold"
        assert tags[1]["text"] == "粗体"

    def test_parse_link(self, client):
        """[text](url) 解析为 link tag"""
        tags = FeishuClient._parse_md_line("点击[链接](https://feishu.cn)查看")
        assert any(t["tag"] == "a" and t["href"] == "https://feishu.cn" for t in tags)

    def test_md_to_post_content_structure(self, client):
        """Markdown 转 post content 结构正确"""
        md = "## 标题\n\n第一段\n\n第二段"
        result = client._md_to_post_content(md)
        assert "zh_cn" in result
        assert "title" in result["zh_cn"]
        assert "content" in result["zh_cn"]
        assert len(result["zh_cn"]["content"]) >= 2


# ── health_check 测试 ─────────────────────────────────


class TestHealthCheck:

    @patch("httpx.Client")
    def test_health_check_success(self, mock_client_class, client):
        """健康检查通过"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.return_value = token_response()
        mock_client_class.return_value = mock_cli

        assert client.health_check() is True

    def test_health_check_no_credentials(self):
        """无配置时健康检查失败"""
        c = FeishuClient()
        c.app_id = ""
        c.app_secret = ""
        assert c.health_check() is False

    @patch("httpx.Client")
    def test_health_check_network_error(self, mock_client_class, client):
        """网络错误时健康检查失败"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = httpx.RequestError("DNS fail")
        mock_client_class.return_value = mock_cli

        assert client.health_check() is False


# ── 重试与超时测试 ────────────────────────────────────


class TestRetry:

    @patch("httpx.Client")
    def test_retry_on_network_error(self, mock_client_class, client):
        """网络错误触发重试后成功"""
        client.backoff = 0.001
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            httpx.RequestError("conn refuse"),  # 第一次发送失败
            send_response("om_retry"),           # 第二次成功
        ]
        mock_client_class.return_value = mock_cli

        result = client.send_text("ou_t", "重试测试")
        assert result["data"]["message_id"] == "om_retry"

    @patch("httpx.Client")
    def test_retry_exhausted(self, mock_client_class, client):
        """重试耗尽后抛出异常"""
        client.backoff = 0.001
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            httpx.RequestError("fail"),
            httpx.RequestError("fail"),
            httpx.RequestError("fail"),  # 3次重试
        ]
        mock_client_class.return_value = mock_cli

        with pytest.raises(FeishuError, match="重试 3 次"):
            client.send_text("ou_t", "耗尽测试")

    @patch("httpx.Client")
    def test_timeout_triggers_retry(self, mock_client_class, client):
        """超时触发重试"""
        client.backoff = 0.001
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.side_effect = [
            token_response(),
            httpx.TimeoutException("timeout"),  # 超时
            send_response("om_timeout"),          # 重试成功
        ]
        mock_client_class.return_value = mock_cli

        result = client.send_text("ou_t", "超时测试")
        assert result["data"]["message_id"] == "om_timeout"


# ── 并发测试 ───────────────────────────────────────────


class TestConcurrency:

    @patch("httpx.Client")
    def test_concurrent_send_text(self, mock_client_class, client):
        """并发发送消息"""
        mock_cli = MagicMock()
        mock_cli.__enter__.return_value = mock_cli
        mock_cli.post.return_value = send_response()
        mock_client_class.return_value = mock_cli

        # Pre-set token
        client._token = "pre_set_token"
        client._expires_at = time.time() + 7200

        errors = []

        def send_msg(i: int):
            try:
                client.send_text(f"ou_{i}", f"msg_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=send_msg, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ── 单例测试 ───────────────────────────────────────────


class TestSingleton:

    def test_singleton(self):
        c1 = get_feishu_client()
        c2 = get_feishu_client()
        assert c1 is c2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
