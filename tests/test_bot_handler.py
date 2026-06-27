"""MessageHandler 单元测试

覆盖：系统命令、自选股操作、Coordinator 路由、事件解析、错误处理。
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.bot.handler import MessageHandler, get_handler
from src.bot.client import FeishuError
from src.ai.deepseek import DeepSeekError
from src.watchlist.manager import WatchlistError
from src.agents.base import AgentResponse, AgentType


# ── Fixtures ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_singletons():
    """重置所有单例"""
    from src.bot.handler import MessageHandler as MH
    MH._instance = None
    MH._initialized = False


@pytest.fixture
def mock_deps():
    """创建所有 mock 依赖"""
    with (
        patch("src.bot.handler.get_feishu_client") as mock_f,
        patch("src.bot.handler.get_coordinator") as mock_c,
        patch("src.bot.handler.get_config") as mock_cfg,
        patch("src.bot.handler.get_watchlist") as mock_wl,
        patch("src.bot.handler.get_market_agent") as mock_market_agent,
        patch("src.bot.handler.get_report_agent") as mock_report_agent,
        patch("src.bot.handler.get_alert_agent") as mock_alert_agent,
        patch("src.bot.handler.get_news_agent") as mock_news_agent,
        patch("src.bot.handler.get_general_agent") as mock_general_agent,
    ):
        market_agent = MagicMock()
        report_agent = MagicMock()
        alert_agent = MagicMock()
        news_agent = MagicMock()
        general_agent = MagicMock()
        mock_market_agent.return_value = market_agent
        mock_report_agent.return_value = report_agent
        mock_alert_agent.return_value = alert_agent
        mock_news_agent.return_value = news_agent
        mock_general_agent.return_value = general_agent

        handler = MessageHandler()

        yield {
            "handler": handler,
            "feishu": handler.feishu,
            "coordinator": handler.coordinator,
            "config": handler.config,
            "watchlist": handler.watchlist,
            "alert_agent": handler.alert_agent,
            "market_agent": market_agent,
            "report_agent": report_agent,
            "news_agent": news_agent,
            "general_agent": general_agent,
            "mock_feishu_get": mock_f,
            "mock_coordinator_get": mock_c,
            "mock_config_get": mock_cfg,
            "mock_watchlist_get": mock_wl,
        }


def test_register_agents_order(mock_deps):
    """Agent 注册顺序：Market → Report → Alert → News → General。"""
    registered = [
        call.args[0]
        for call in mock_deps["coordinator"].register.call_args_list
    ]
    assert registered == [
        mock_deps["market_agent"],
        mock_deps["report_agent"],
        mock_deps["alert_agent"],
        mock_deps["news_agent"],
        mock_deps["general_agent"],
    ]


def make_text_event(text: str, open_id: str = "ou_test",
                    message_id: str = "om_test") -> dict:
    """构造飞书文本消息事件"""
    return {
        "schema": "2.0",
        "header": {
            "event_id": "evt_test",
            "event_type": "im.message.receive_v1",
        },
        "event": {
            "message": {
                "message_id": message_id,
                "content": json.dumps({"text": text}),
                "message_type": "text",
            },
            "sender": {
                "sender_id": {"open_id": open_id},
            },
        },
    }


def make_text_event_with_content(content: str, open_id: str = "ou_test",
                                 message_id: str = "om_test") -> dict:
    event = make_text_event("", open_id=open_id, message_id=message_id)
    event["event"]["message"]["content"] = content
    return event


# ═══════════════════════════════════════════════════════════
#  系统命令测试
# ═══════════════════════════════════════════════════════════


class TestSystemCommands:

    def test_handle_start(self, mock_deps):
        """启动命令"""
        assert mock_deps["handler"].process_message("ou_x", "om_x", "启动")
        mock_deps["config"].set_enabled.assert_called_with(True)

    def test_handle_pause(self, mock_deps):
        """暂停命令"""
        mock_deps["handler"].process_message("ou_x", "om_x", "暂停")
        mock_deps["config"].set_enabled.assert_called_with(False)

    def test_handle_status(self, mock_deps):
        """状态查询"""
        mock_deps["config"].get_enabled.return_value = True
        mock_deps["config"].get_market.return_value = "CN"
        mock_deps["config"].get_scan_interval.return_value = 1800

        reply = mock_deps["handler"].process_message("ou_x", "om_x", "状态")
        assert "运行中" in reply
        assert "CN" in reply
        assert "1800" in reply

    def test_handle_status_paused(self, mock_deps):
        """暂停状态查询"""
        mock_deps["config"].get_enabled.return_value = False
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "状态")
        assert "已暂停" in reply

    def test_switch_to_hk(self, mock_deps):
        """切换港股"""
        mock_deps["handler"].process_message("ou_x", "om_x", "切换港股")
        mock_deps["config"].set_market.assert_called_with("HK")

    def test_switch_to_us(self, mock_deps):
        """切换美股"""
        mock_deps["handler"].process_message("ou_x", "om_x", "切换美股")
        mock_deps["config"].set_market.assert_called_with("US")

    def test_set_scan_interval(self, mock_deps):
        """设置扫描间隔"""
        mock_deps["handler"].process_message("ou_x", "om_x", "扫描频率 60")
        mock_deps["config"].set_scan_interval.assert_called_with(60)

    def test_set_scan_interval_invalid(self, mock_deps):
        """无效的扫描间隔"""
        mock_deps["config"].set_scan_interval.side_effect = ValueError("至少60秒")
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "扫描频率 30")
        assert "失败" in reply

    def test_manual_scan_command(self, mock_deps):
        """手动扫描命令"""
        mock_event = MagicMock()
        mock_event.related_code = "000001"
        mock_event.title = "平安银行 放量拉升"
        mock_event.strength = 8.1
        mock_event.event_id = "price_spike:000001"
        mock_deps["alert_agent"].scan_watchlist.return_value = {
            "data_source": "mock",
            "scanned": 3,
            "triggered": 1,
            "deliverable": 1,
            "message": "扫描完成，发现 1 条可推送预警。",
            "alerts": [
                {"event": mock_event, "should_send": True, "reason": "new_event"},
            ],
        }

        reply = mock_deps["handler"].process_message("ou_x", "om_x", "立即扫描")

        assert "盘中扫描完成" in reply
        assert "000001" in reply
        mock_deps["alert_agent"].mark_delivered.assert_called_once_with(
            ["price_spike:000001"]
        )


# ═══════════════════════════════════════════════════════════
#  自选股测试
# ═══════════════════════════════════════════════════════════


class TestWatchlistCommands:

    def test_add_watchlist(self, mock_deps):
        """添加自选股"""
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "添加自选 000001")
        assert "已添加" in reply
        mock_deps["watchlist"].add_stock.assert_called_with(
            symbol="000001", name="000001", market="CN"
        )

    def test_add_watchlist_no_code(self, mock_deps):
        """添加自选缺少代码"""
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "添加自选")
        assert "使用方法" in reply
        mock_deps["watchlist"].add_stock.assert_not_called()

    def test_add_watchlist_duplicate(self, mock_deps):
        """重复添加"""
        mock_deps["watchlist"].add_stock.side_effect = WatchlistError("已存在")
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "添加自选 000001")
        assert "失败" in reply

    def test_remove_watchlist(self, mock_deps):
        """删除自选股"""
        mock_deps["watchlist"].remove_stock.return_value = True
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "删除自选 000001")
        assert "删除" in reply
        mock_deps["watchlist"].remove_stock.assert_called_with("000001")

    def test_remove_nonexistent(self, mock_deps):
        """删除不存在的自选股"""
        mock_deps["watchlist"].remove_stock.return_value = False
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "删除自选 999999")
        assert "未找到" in reply

    def test_list_watchlist(self, mock_deps):
        """查看自选股"""
        from src.db.models import WatchlistItem
        from datetime import datetime
        mock_deps["watchlist"].list_stocks.return_value = [
            WatchlistItem(id=1, symbol="000001", name="平安银行", market="a",
                          tags="银行", notes="", added_at=datetime.now()),
        ]
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "我的自选")
        assert "平安银行" in reply
        assert "000001" in reply
        assert "银行" in reply

    def test_list_watchlist_empty(self, mock_deps):
        """空自选股列表"""
        mock_deps["watchlist"].list_stocks.return_value = []
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "自选股")
        assert "空" in reply or "列表为空" in reply

    def test_clear_watchlist(self, mock_deps):
        """清空自选股"""
        mock_deps["watchlist"].clear.return_value = 3
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "清空自选")
        assert "清空" in reply

    def test_watchlist_error(self, mock_deps):
        """Watchlist 异常"""
        mock_deps["watchlist"].clear.side_effect = WatchlistError("DB error")
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "清空自选")
        assert "失败" in reply


# ═══════════════════════════════════════════════════════════
#  Coordinator 路由测试
# ═══════════════════════════════════════════════════════════


class TestCoordinatorRouting:

    def test_route_market_question(self, mock_deps):
        """市场问题路由到 Coordinator"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.MARKET, message="市场分析结果",
        )
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "今天市场怎么样")
        assert reply == "市场分析结果"
        mock_deps["coordinator"].route.assert_called_with("ou_x", "今天市场怎么样")

    def test_route_report_request(self, mock_deps):
        """日报请求路由到 Coordinator"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.REPORT, message="日报内容",
        )
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "生成早报")
        assert reply == "日报内容"

    def test_route_deepseek_error(self, mock_deps):
        """DeepSeek 错误返回兜底"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=False, agent=AgentType.REPORT, message="请稍后重试",
        )
        reply = mock_deps["handler"].process_message("ou_x", "om_x", "分析")
        assert "重试" in reply or "错误" in reply or "稍后" in reply


# ═══════════════════════════════════════════════════════════
#  事件解析测试
# ═══════════════════════════════════════════════════════════


class TestEventParsing:

    def test_handle_event_success(self, mock_deps):
        """完整事件处理流程"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.MARKET, message="分析回复",
        )
        event = make_text_event("今天市场怎么样")
        mock_deps["handler"].handle_event(event)
        mock_deps["feishu"].reply_text.assert_called()

    def test_handle_non_message_event(self, mock_deps):
        """非消息事件忽略"""
        event = {"header": {"event_type": "url_verification"}}
        result = mock_deps["handler"].handle_event(event)
        assert result is None
        mock_deps["feishu"].reply_text.assert_not_called()

    def test_handle_invalid_content_json(self, mock_deps):
        """无效 content JSON"""
        event = make_text_event("")
        event["event"]["message"]["content"] = "invalid json"
        result = mock_deps["handler"].handle_event(event)
        assert result is None

    def test_handle_empty_text(self, mock_deps):
        """空文本"""
        event = make_text_event("")
        result = mock_deps["handler"].handle_event(event)
        assert result == "消息不能为空，请重新输入。"
        mock_deps["feishu"].reply_text.assert_called_once_with(
            "om_test",
            "消息不能为空，请重新输入。",
        )

    def test_parse_plain_chinese_stock_name(self, mock_deps):
        """content='{"text":"有研新材"}' 必须完整传入 Coordinator。"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.MARKET, message="市场回复",
        )
        event = make_text_event_with_content(
            '{"text":"有研新材"}',
            message_id="om_cn_1",
        )

        result = mock_deps["handler"].handle_event(event)

        assert result == "市场回复"
        mock_deps["coordinator"].route.assert_called_once_with(
            "ou_test",
            "有研新材",
        )

    def test_parse_chinese_query_text(self, mock_deps):
        """content='{"text":"查一下中瓷电子"}' 必须完整传入 Coordinator。"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.MARKET, message="市场回复",
        )
        event = make_text_event_with_content(
            '{"text":"查一下中瓷电子"}',
            message_id="om_cn_2",
        )

        result = mock_deps["handler"].handle_event(event)

        assert result == "市场回复"
        mock_deps["coordinator"].route.assert_called_once_with(
            "ou_test",
            "查一下中瓷电子",
        )

    def test_parse_text_with_at_bot_keeps_chinese(self, mock_deps):
        """包含 @机器人 的飞书消息只移除 @ 标签，不破坏中文。"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.MARKET, message="市场回复",
        )
        event = make_text_event_with_content(
            '{"text":"<at user_id=\\"ou_bot\\">投资助手</at> 查一下中瓷电子"}',
            message_id="om_at_1",
        )
        event["event"]["message"]["mentions"] = [
            {
                "key": "@_user_1",
                "name": "投资助手",
                "id": {"open_id": "ou_bot"},
            }
        ]

        result = mock_deps["handler"].handle_event(event)

        assert result == "市场回复"
        mock_deps["coordinator"].route.assert_called_once_with(
            "ou_test",
            "查一下中瓷电子",
        )

    def test_parse_text_with_at_placeholder_keeps_chinese(self, mock_deps):
        """飞书 mentions[*].key 占位符只移除 @，不破坏后续中文。"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.MARKET, message="市场回复",
        )
        event = make_text_event_with_content(
            '{"text":"@_user_1 有研新材"}',
            message_id="om_at_2",
        )
        event["event"]["message"]["mentions"] = [
            {"key": "@_user_1", "name": "投资助手"}
        ]

        result = mock_deps["handler"].handle_event(event)

        assert result == "市场回复"
        mock_deps["coordinator"].route.assert_called_once_with(
            "ou_test",
            "有研新材",
        )

    def test_reply_uses_original_message_id_for_feishu_quote(self, mock_deps):
        """回复引用应使用原始 message_id，而不是清洗后的文本。"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.MARKET, message="市场回复",
        )
        event = make_text_event("有研新材", message_id="om_original_cn")

        mock_deps["handler"].handle_event(event)

        mock_deps["feishu"].reply_text.assert_called_once_with(
            "om_original_cn",
            "市场回复",
        )

    def test_handle_fei_shu_error(self, mock_deps):
        """飞书 API 错误不应导致崩溃"""
        mock_deps["feishu"].reply_text.side_effect = FeishuError("API error")
        event = make_text_event("状态")
        result = mock_deps["handler"].handle_event(event)
        assert result is None  # 异常被吞掉

    def test_duplicate_message_id_only_replied_once(self, mock_deps):
        """同一条飞书消息重复投递时只回复一次"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.GENERAL, message="只回复一次",
        )
        event = make_text_event("你好", message_id="om_dup_1")

        first = mock_deps["handler"].handle_event(event)
        second = mock_deps["handler"].handle_event(event)

        assert first == "只回复一次"
        assert second is None
        mock_deps["coordinator"].route.assert_called_once_with("ou_test", "你好")
        mock_deps["feishu"].reply_text.assert_called_once()

    def test_failed_message_can_retry(self, mock_deps):
        """首次处理失败后，同一 message_id 后续仍可重试"""
        mock_deps["coordinator"].route.return_value = AgentResponse(
            success=True, agent=AgentType.GENERAL, message="重试成功",
        )
        mock_deps["feishu"].reply_text.side_effect = [
            FeishuError("temporary error"),
            None,
        ]
        event = make_text_event("你好", message_id="om_retry_1")

        first = mock_deps["handler"].handle_event(event)
        second = mock_deps["handler"].handle_event(event)

        assert first is None
        assert second == "重试成功"
        assert mock_deps["coordinator"].route.call_count == 2
        assert mock_deps["feishu"].reply_text.call_count == 2


# ═══════════════════════════════════════════════════════════
#  单例测试
# ═══════════════════════════════════════════════════════════


class TestSingleton:

    def test_singleton(self):
        h1 = get_handler()
        h2 = get_handler()
        assert h1 is h2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
