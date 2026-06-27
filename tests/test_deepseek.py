"""
DeepSeekClient 单元测试

全部使用 mock，不调用真实 API。
覆盖：正常调用、异常、超时、重试、记忆集成、健康检查、Token 估算。
"""

from unittest.mock import patch, MagicMock, call
from typing import Any

import httpx
import pytest

from src.ai.deepseek import DeepSeekClient, DeepSeekError, get_deepseek
from src.memory import MemoryError


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前重置单例"""
    DeepSeekClient._instance = None
    DeepSeekClient._initialized = False  # type: ignore[attr-defined]


@pytest.fixture
def client():
    """创建干净的客户端实例"""
    return DeepSeekClient()


# ── 辅助 Mock ───────────────────────────────────────────

def make_mock_response(status_code: int, json_data: dict) -> MagicMock:
    """构造模拟的 httpx 响应"""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.text = str(json_data)
    return mock


def make_mock_client(post_return: MagicMock = None) -> MagicMock:
    """构造模拟的 httpx.Client"""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    if post_return is not None:
        mock_client.post.return_value = post_return
    return mock_client


def make_success_response(content: str = "今日市场震荡上行") -> MagicMock:
    """构造成功的 API 响应 Mock"""
    return make_mock_response(200, {
        "choices": [{"message": {"content": content}}]
    })


# ═══════════════════════════════════════════════════════════
#  基础 chat 测试
# ═══════════════════════════════════════════════════════════


class TestChat:
    """chat() 基础功能测试"""

    @patch("httpx.Client")
    def test_chat_success(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """正常对话返回"""
        mock_response = make_success_response("今日市场震荡上行")
        mock_client = make_mock_client(mock_response)
        mock_client_class.return_value = mock_client

        result = client.chat([{"role": "user", "content": "今天怎么样"}])

        assert result == "今日市场震荡上行"
        # 验证调用参数
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["model"] == client.model
        assert call_kwargs["json"]["messages"] == [{"role": "user", "content": "今天怎么样"}]
        assert call_kwargs["json"]["temperature"] == 0.7

    @patch("httpx.Client")
    def test_chat_with_system_message(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """包含系统消息的对话"""
        mock_client = make_mock_client(make_success_response())
        mock_client_class.return_value = mock_client

        messages = [
            {"role": "system", "content": "你是一个投资助手"},
            {"role": "user", "content": "分析大盘"},
        ]
        result = client.chat(messages)

        assert result == "今日市场震荡上行"
        assert mock_client.post.call_args.kwargs["json"]["messages"] == messages

    @patch("httpx.Client")
    def test_chat_with_temperature_and_max_tokens(
        self, mock_client_class: MagicMock, client: DeepSeekClient
    ):
        """自定义 temperature 和 max_tokens"""
        mock_client = make_mock_client(make_success_response())
        mock_client_class.return_value = mock_client

        client.chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=500,
        )

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 500

    @patch("httpx.Client")
    def test_chat_empty_messages(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """空消息列表应抛异常"""
        with pytest.raises(DeepSeekError, match="不能为空"):
            client.chat([])

    @patch("httpx.Client")
    def test_chat_api_error_400(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """400 客户端错误不重试"""
        error_response = make_mock_response(400, {"error": "invalid_request"})
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=MagicMock(), response=error_response,
        )
        mock_client = make_mock_client(error_response)
        mock_client_class.return_value = mock_client

        with pytest.raises(DeepSeekError, match="400|API"):
            client.chat([{"role": "user", "content": "hi"}])

    @patch("httpx.Client")
    def test_chat_api_error_500(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """500 服务端错误触发重试"""
        client.backoff = 0.001  # 加速测试
        error_response = make_mock_response(500, {"error": "server_error"})
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Error", request=MagicMock(), response=error_response,
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = error_response
        mock_client_class.return_value = mock_client

        with pytest.raises(DeepSeekError, match="重试耗尽|500"):
            client.chat([{"role": "user", "content": "hi"}])

        # 验证重试了 max_retries + 1 次
        assert mock_client.post.call_count == client.max_retries + 1

    @patch("httpx.Client")
    def test_chat_network_error_retry(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """网络错误触发重试"""
        client.backoff = 0.001  # 加速测试

        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client

        # 前 3 次失败，第 4 次成功
        mock_client.post.side_effect = [
            httpx.RequestError("Connection refused"),
            httpx.RequestError("Connection refused"),
            None,  # 先不返回，让第三次调用成功
        ]

        return_value = make_success_response("最终回复")
        mock_client.post.side_effect = [
            httpx.RequestError("Connection refused"),
            make_success_response("最终回复"),
        ]

        mock_client_class.return_value = mock_client

        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "最终回复"
        assert mock_client.post.call_count == 2  # 1 fail + 1 success

    @patch("httpx.Client")
    def test_chat_retry_exhausted(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """所有重试耗尽后抛出异常"""
        client.backoff = 0.001  # 加速测试
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.RequestError("Network down")
        mock_client_class.return_value = mock_client

        with pytest.raises(DeepSeekError, match="重试耗尽"):
            client.chat([{"role": "user", "content": "hi"}])

        assert mock_client.post.call_count == client.max_retries + 1

    @patch("httpx.Client")
    def test_chat_response_format_error(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """API 返回格式异常"""
        bad_json = {"unexpected": "format"}
        mock_response = make_mock_response(200, bad_json)
        mock_client = make_mock_client(mock_response)
        mock_client_class.return_value = mock_client

        with pytest.raises(DeepSeekError, match="格式异常"):
            client.chat([{"role": "user", "content": "hi"}])

    @patch("httpx.Client")
    def test_chat_custom_model(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """验证使用的模型名来自 settings"""
        mock_client = make_mock_client(make_success_response())
        mock_client_class.return_value = mock_client

        client.chat([{"role": "user", "content": "hi"}])
        model = mock_client.post.call_args.kwargs["json"]["model"]
        assert model == client.model


# ═══════════════════════════════════════════════════════════
#  chat_with_memory 测试
# ═══════════════════════════════════════════════════════════


class TestChatWithMemory:
    """chat_with_memory() 记忆集成测试"""

    @patch("httpx.Client")
    def test_chat_with_memory_basic(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """基本记忆集成流程"""
        mock_client = make_mock_client(make_success_response("分析结果"))
        mock_client_class.return_value = mock_client

        with (
            patch.object(client._memory, "get_context") as mock_get_ctx,
            patch.object(client._memory, "add_message") as mock_add,
            patch.object(client._memory, "summarize_if_needed") as mock_summarize,
        ):
            mock_get_ctx.return_value = [{"role": "user", "content": "之前对话"}]

            result = client.chat_with_memory("session_1", "分析大盘")

            assert result == "分析结果"

            # 1. 保存用户消息
            mock_add.assert_any_call("session_1", "user", "分析大盘")
            # 2. 获取上下文
            mock_get_ctx.assert_called_once_with("session_1")
            # 3. 保存回复
            mock_add.assert_any_call("session_1", "assistant", "分析结果")
            # 4. 检查摘要
            mock_summarize.assert_called_once_with("session_1")

    @patch("httpx.Client")
    def test_chat_with_memory_context_passed(self, mock_client_class: MagicMock,
                                              client: DeepSeekClient):
        """确保上下文被传递给 AI"""
        mock_client = make_mock_client(make_success_response("回复"))
        mock_client_class.return_value = mock_client

        with patch.object(client._memory, "get_context") as mock_get_ctx:
            mock_get_ctx.return_value = [
                {"role": "system", "content": "摘要"},
                {"role": "user", "content": "之前的话"},
            ]
            client.chat_with_memory("s", "新问题")

            sent_messages = mock_client.post.call_args.kwargs["json"]["messages"]
            # get_context 返回值直接传递给了 API（已包含新用户消息）
            assert len(sent_messages) == 2  # get_context 返回: system + user
            assert sent_messages[0] == {"role": "system", "content": "摘要"}

    @patch("httpx.Client")
    def test_chat_with_memory_transient_system_messages(
        self,
        mock_client_class: MagicMock,
        client: DeepSeekClient,
    ):
        """本轮 system 消息应置顶，并过滤历史实时行情上下文"""
        mock_client = make_mock_client(make_success_response("回复"))
        mock_client_class.return_value = mock_client

        with patch.object(client._memory, "get_context") as mock_get_ctx:
            mock_get_ctx.return_value = [
                {"role": "system", "content": "当前关注市场: CN。\n【实时行情】旧价格"},
                {"role": "user", "content": "之前的话"},
            ]
            client.chat_with_memory(
                "s",
                "新问题",
                system_messages=["新系统规则", "【实时行情】新价格"],
            )

            sent_messages = mock_client.post.call_args.kwargs["json"]["messages"]
            assert sent_messages == [
                {"role": "system", "content": "新系统规则"},
                {"role": "system", "content": "【实时行情】新价格"},
                {"role": "user", "content": "之前的话"},
            ]

    @patch("httpx.Client")
    def test_chat_with_memory_error(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """记忆模块异常应转为 DeepSeekError"""
        mock_client = make_mock_client(make_success_response(""))
        mock_client_class.return_value = mock_client

        with patch.object(client._memory, "add_message") as mock_add:
            mock_add.side_effect = MemoryError("DB error")

            with pytest.raises(DeepSeekError, match="记忆模块"):
                client.chat_with_memory("s", "hello")


# ═══════════════════════════════════════════════════════════
#  health_check 测试
# ═══════════════════════════════════════════════════════════


class TestHealthCheck:
    """health_check() 测试"""

    @patch("httpx.Client")
    def test_health_check_success(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """API 健康"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        assert client.health_check() is True

    @patch("httpx.Client")
    def test_health_check_failure(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """API 不健康"""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        assert client.health_check() is False

    @patch("httpx.Client")
    def test_health_check_network_error(self, mock_client_class: MagicMock, client: DeepSeekClient):
        """网络不通"""
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.side_effect = httpx.RequestError("DNS failed")
        mock_client_class.return_value = mock_client

        assert client.health_check() is False


# ═══════════════════════════════════════════════════════════
#  estimate_tokens 测试
# ═══════════════════════════════════════════════════════════


class TestEstimateTokens:
    """estimate_tokens() 测试"""

    def test_estimate_empty(self, client: DeepSeekClient):
        """空文本返回 0"""
        assert client.estimate_tokens("") == 0

    def test_estimate_chinese(self, client: DeepSeekClient):
        """中文文本估算"""
        text = "今日市场震荡上行，上证指数收于3200点"
        tokens = client.estimate_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_english(self, client: DeepSeekClient):
        """英文文本估算"""
        text = "The market is up today with strong volume across all sectors."
        tokens = client.estimate_tokens(text)
        assert tokens > 0

    def test_estimate_mixed(self, client: DeepSeekClient):
        """中英文混合"""
        text = "上证指数 Shanghai Composite 收于 3200 点"
        tokens = client.estimate_tokens(text)
        assert tokens > 0

    def test_estimate_returns_at_least_1(self, client: DeepSeekClient):
        """非空文本至少返回 1"""
        assert client.estimate_tokens("a") >= 1
        assert client.estimate_tokens("。") >= 1


# ═══════════════════════════════════════════════════════════
#  单例测试
# ═══════════════════════════════════════════════════════════


class TestSingleton:

    def test_singleton(self):
        """验证单例"""
        c1 = get_deepseek()
        c2 = get_deepseek()
        assert c1 is c2


# ═══════════════════════════════════════════════════════════
#  DeepSeekError 测试
# ═══════════════════════════════════════════════════════════


class TestDeepSeekError:

    def test_error_is_exception(self):
        """DeepSeekError 是 Exception 子类"""
        assert issubclass(DeepSeekError, Exception)

    def test_error_raised(self):
        """验证异常抛出"""
        with pytest.raises(DeepSeekError):
            raise DeepSeekError("测试错误")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
