"""NewsAgent 单元测试。"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.agents.base import AgentResponse, AgentType
from src.agents.news_agent import (
    EastMoneyNewsItem,
    NewsAgent,
    clean_text,
    get_news_agent,
    is_news_intent,
)
from src.ai.deepseek import DeepSeekError


@pytest.fixture(autouse=True)
def reset_singleton():
    NewsAgent._instance = None
    NewsAgent._initialized = False  # type: ignore[attr-defined]
    import src.agents.news_agent as news_module

    news_module._news_agent_instance = None


@pytest.fixture
def mock_deps():
    with (
        patch("src.agents.news_agent.get_deepseek") as mock_ds,
        patch("src.agents.news_agent.get_market_data_service") as mock_mds,
    ):
        deepseek = MagicMock()
        deepseek.chat.return_value = (
            "【核心事件】\n有新闻。\n"
            "【事件解读】\n事件解读。\n"
            "【对板块影响】\n板块影响。\n"
            "【对个股影响】\n个股影响。\n"
            "【持续性判断】\n持续性判断。"
        )
        mock_ds.return_value = deepseek

        market_data = MagicMock()
        market_data.extract_symbol.return_value = None
        mock_mds.return_value = market_data

        agent = NewsAgent()
        yield {
            "agent": agent,
            "deepseek": deepseek,
            "market_data": market_data,
        }


class TestNewsIntent:
    @pytest.mark.parametrize(
        "message",
        [
            "信维通信今天有什么消息？",
            "商业航天为什么涨？",
            "PCB板块有什么催化？",
            "长川科技最近有什么新闻？",
            "机器人事件驱动是什么？",
        ],
    )
    def test_can_handle_news_queries(self, mock_deps, message):
        assert mock_deps["agent"].can_handle(message) is True
        assert is_news_intent(message) is True

    def test_can_handle_rejects_non_news_query(self, mock_deps):
        assert mock_deps["agent"].can_handle("分析平安银行") is False

    def test_extract_symbol_as_keyword(self, mock_deps):
        mock_deps["market_data"].extract_symbol.return_value = "300136"

        assert mock_deps["agent"]._extract_keyword("300136 今天有什么消息") == "300136"

    def test_extract_text_keyword(self, mock_deps):
        assert mock_deps["agent"]._extract_keyword("PCB板块有什么催化？") == "PCB"


class TestNewsAgentHandle:
    def test_handle_no_news_returns_clear_message(self, mock_deps):
        mock_deps["agent"].search_eastmoney = MagicMock(return_value=[])

        resp = mock_deps["agent"].handle("ou_x", "长川科技最近有什么新闻？")

        assert isinstance(resp, AgentResponse)
        assert resp.success is True
        assert resp.agent == AgentType.NEWS
        assert "未获取到相关新闻" in resp.message
        assert resp.metadata["news_count"] == 0
        mock_deps["deepseek"].chat.assert_not_called()

    def test_handle_with_news_calls_deepseek_with_source_items(self, mock_deps):
        items = [
            EastMoneyNewsItem(
                title="PCB产业链加速价值重估",
                content="AI算力浪潮带动PCB需求。",
                date="2026-06-26 16:12:31",
                source="东方财富",
                url="http://finance.eastmoney.com/a/test.html",
                item_type="新闻",
            )
        ]
        mock_deps["agent"].search_eastmoney = MagicMock(return_value=items)

        resp = mock_deps["agent"].handle("ou_x", "PCB板块有什么催化？")

        assert resp.success is True
        assert resp.agent == AgentType.NEWS
        assert "【核心事件】" in resp.message
        assert "【事件解读】" in resp.message
        assert resp.metadata["news_count"] == 1

        messages = mock_deps["deepseek"].chat.call_args.args[0]
        prompt = messages[1]["content"]
        assert "PCB产业链加速价值重估" in prompt
        assert "AI算力浪潮带动PCB需求" in prompt
        assert "不允许编造新闻" in prompt

    def test_handle_wraps_malformed_llm_reply(self, mock_deps):
        mock_deps["agent"].search_eastmoney = MagicMock(return_value=[
            EastMoneyNewsItem(
                title="商业航天消息",
                content="产业链消息。",
                date="2026-06-26 10:00:00",
                source="东方财富",
                url="",
                item_type="新闻",
            )
        ])
        mock_deps["deepseek"].chat.return_value = "只有一句分析"

        resp = mock_deps["agent"].handle("ou_x", "商业航天为什么涨？")

        assert "【核心事件】" in resp.message
        assert "只有一句分析" in resp.message
        assert "【持续性判断】" in resp.message

    def test_handle_deepseek_error(self, mock_deps):
        mock_deps["agent"].search_eastmoney = MagicMock(return_value=[
            EastMoneyNewsItem(
                title="信维通信消息",
                content="公司消息。",
                date="2026-06-26 10:00:00",
                source="东方财富",
                url="",
                item_type="新闻",
            )
        ])
        mock_deps["deepseek"].chat.side_effect = DeepSeekError("API timeout")

        resp = mock_deps["agent"].handle("ou_x", "信维通信今天有什么消息？")

        assert resp.success is False
        assert resp.agent == AgentType.NEWS
        assert resp.metadata["error_type"] == "DeepSeekError"


class TestEastMoneySearch:
    def test_search_eastmoney_parses_news_and_notice(self, mock_deps, monkeypatch):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url, params):
                param = json.loads(params["param"])
                type_name = param["type"][0]
                if type_name == "cmsArticleWebOld":
                    payload = {
                        "result": {
                            "cmsArticleWebOld": [
                                {
                                    "title": "信维通信获机构关注",
                                    "content": "<em>信维通信</em> 相关消息",
                                    "date": "2026-06-26 17:47:00",
                                    "mediaName": "东方财富",
                                    "url": "http://finance.eastmoney.com/a/1.html",
                                }
                            ]
                        }
                    }
                else:
                    payload = {
                        "result": {
                            "noticeWeb": [
                                {
                                    "title": "信维通信: 关于项目进展公告",
                                    "content": "公告摘要",
                                    "date": "2026-06-27 00:00:00",
                                    "securityFullName": "信维通信",
                                    "url": "http://data.eastmoney.com/notices/1.html",
                                }
                            ]
                        }
                    }
                request = httpx.Request("GET", url)
                return httpx.Response(
                    200,
                    request=request,
                    text=f"jQuery({json.dumps(payload, ensure_ascii=False)})",
                )

        monkeypatch.setattr("src.agents.news_agent.httpx.Client", FakeClient)

        items = mock_deps["agent"].search_eastmoney("信维通信")

        assert len(items) == 2
        assert items[0].item_type == "新闻"
        assert items[0].content == "信维通信 相关消息"
        assert items[1].item_type == "公告"
        assert items[1].source == "信维通信"

    def test_search_eastmoney_handles_http_error(self, mock_deps, monkeypatch):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def get(self, url, params):
                request = httpx.Request("GET", url)
                return httpx.Response(500, request=request)

        monkeypatch.setattr("src.agents.news_agent.httpx.Client", FakeClient)

        assert mock_deps["agent"].search_eastmoney("信维通信") == []

    def test_clean_text_removes_html(self):
        assert clean_text("<em>信维通信</em>&nbsp;消息") == "信维通信 消息"


def test_get_news_agent_singleton(mock_deps):
    assert get_news_agent() is get_news_agent()
