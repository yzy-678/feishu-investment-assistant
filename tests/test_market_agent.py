"""
MarketAgent 单元测试

测试覆盖：can_handle 关键词匹配、handle 流程、额外接口
（analyze_stock/analyze_watchlist/market_overview）、
Prompt 注入、异常处理、边界条件、并发。
所有外部依赖使用 mock。
"""

import threading
from unittest.mock import MagicMock, patch, call

import pytest

from src.agents.base import AgentType, AgentResponse
from src.agents.market_agent import MarketAgent, get_market_agent
from src.ai.deepseek import DeepSeekError
from src.watchlist.manager import WatchlistError
from src.db.models import WatchlistItem


# ── 辅助: 创建测试用自选股 ─────────────────────────────

def make_watchlist_item(symbol: str, name: str, market: str = "a",
                        tags: str = "", notes: str = "") -> WatchlistItem:
    from datetime import datetime
    return WatchlistItem(
        id=hash(symbol) % 100000,
        symbol=symbol,
        name=name,
        market=market,
        tags=tags,
        notes=notes,
        added_at=datetime.now(),
    )


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """每个测试前重置 MarketAgent 单例"""
    MarketAgent._instance = None
    MarketAgent._initialized = False  # type: ignore[attr-defined]


@pytest.fixture
def mock_deps():
    """创建所有 mock 依赖并注入"""
    with (
        patch("src.agents.market_agent.get_deepseek") as mock_ds,
        patch("src.agents.market_agent.get_memory") as mock_mem,
        patch("src.agents.market_agent.get_watchlist") as mock_wl,
        patch("src.agents.market_agent.get_config") as mock_cfg,
    ):
        # DeepSeek mock
        mock_ds_instance = MagicMock()
        mock_ds_instance.chat_with_memory.return_value = "这是一个AI测试回复"
        mock_ds_instance.chat.return_value = "这是一个AI测试回复"
        mock_ds.return_value = mock_ds_instance

        # Memory mock
        mock_mem_instance = MagicMock()
        mock_mem.return_value = mock_mem_instance

        # Watchlist mock
        mock_wl_instance = MagicMock()
        mock_wl_instance.list_stocks.return_value = []
        mock_wl.return_value = mock_wl_instance

        # Config mock
        mock_cfg_instance = MagicMock()
        mock_cfg_instance.get_market.return_value = "CN"
        mock_cfg.return_value = mock_cfg_instance

        agent = MarketAgent()
        yield {
            "agent": agent,
            "deepseek": mock_ds_instance,
            "memory": mock_mem_instance,
            "watchlist": mock_wl_instance,
            "config": mock_cfg_instance,
        }


# ═══════════════════════════════════════════════════════════
#  can_handle 测试
# ═══════════════════════════════════════════════════════════


class TestCanHandle:
    """can_handle 关键词匹配测试"""

    @pytest.mark.parametrize("msg", [
        "分析大盘", "市场怎么样", "怎么看平安银行",
        "银行板块分析", "推荐股票", "今天热点是什么",
        "有什么投资机会", "主要风险在哪里",
        "今天主线是什么", "自选股分析",
        "个股推荐", "行情怎么样",
        "关注哪些行业", "持有建议",
        "今日走势", "如何看这个板块",
    ])
    def test_can_handle_keywords(self, mock_deps, msg: str):
        assert mock_deps["agent"].can_handle(msg)

    def test_can_handle_not_matched(self, mock_deps):
        assert not mock_deps["agent"].can_handle("你好呀")
        assert not mock_deps["agent"].can_handle("帮我启动系统")
        assert not mock_deps["agent"].can_handle("暂停")

    def test_can_handle_empty(self, mock_deps):
        assert not mock_deps["agent"].can_handle("")
        assert not mock_deps["agent"].can_handle("   ")

    def test_agent_type(self, mock_deps):
        assert mock_deps["agent"].agent_type == AgentType.MARKET


# ═══════════════════════════════════════════════════════════
#  handle 测试
# ═══════════════════════════════════════════════════════════


class TestHandle:
    """handle() 流程测试"""

    def test_handle_basic(self, mock_deps):
        """基本问答流程"""
        resp = mock_deps["agent"].handle("session1", "分析大盘")
        assert isinstance(resp, AgentResponse)
        assert resp.success is True
        assert resp.agent == AgentType.MARKET
        assert "测试回复" in resp.message

    def test_handle_injects_system_context(self, mock_deps):
        """handle 应注入 system 级别上下文"""
        mock_deps["agent"].handle("session1", "分析大盘")
        calls = mock_deps["memory"].add_message.call_args_list
        system_calls = [c for c in calls if c[0][1] == "system"]
        assert len(system_calls) >= 1
        ctx_text = system_calls[0][0][2]
        assert "当前市场: CN" in ctx_text or "当前关注市场: CN" in ctx_text

    def test_handle_injects_watchlist_context(self, mock_deps):
        """有自选股时应注入自选股信息"""
        items = [
            make_watchlist_item("000001", "平安银行", tags="银行"),
            make_watchlist_item("600519", "贵州茅台", tags="白酒"),
        ]
        mock_deps["watchlist"].list_stocks.return_value = items

        mock_deps["agent"].handle("session1", "分析大盘")
        calls = mock_deps["memory"].add_message.call_args_list
        system_calls = [c for c in calls if c[0][1] == "system"]
        ctx_text = system_calls[0][0][2]
        assert "平安银行" in ctx_text
        assert "贵州茅台" in ctx_text

    def test_handle_calls_chat_with_memory(self, mock_deps):
        """确认调用了 chat_with_memory"""
        mock_deps["agent"].handle("session1", "分析平安银行")
        mock_deps["deepseek"].chat_with_memory.assert_called_once_with(
            "session1", "分析平安银行"
        )

    def test_handle_session_id_propagation(self, mock_deps):
        """session_id 应传递到 metadata"""
        resp = mock_deps["agent"].handle("user_abc", "市场怎么样")
        assert resp.metadata.get("session_id") == "user_abc"

    def test_handle_empty_message(self, mock_deps):
        """空消息依然传递到 AI（由 can_handle 过滤）"""
        resp = mock_deps["agent"].handle("session1", "")
        assert resp.success is True

    def test_handle_long_message(self, mock_deps):
        """超长消息不影响处理"""
        long_msg = "分析" + "市场" * 500
        resp = mock_deps["agent"].handle("session1", long_msg)
        assert resp.success is True


# ═══════════════════════════════════════════════════════════
#  analyze_stock 测试
# ═══════════════════════════════════════════════════════════


class TestAnalyzeStock:
    """analyze_stock() 测试"""

    def test_analyze_stock_basic(self, mock_deps):
        """个股分析"""
        mock_deps["deepseek"].chat.return_value = "平安银行分析结果"
        result = mock_deps["agent"].analyze_stock("000001")
        assert result == "平安银行分析结果"
        mock_deps["deepseek"].chat.assert_called_once()
        # 验证 prompt 中包含股票代码
        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "000001" in prompt

    def test_analyze_stock_deepseek_error(self, mock_deps):
        """DeepSeek 异常应向上传递"""
        mock_deps["deepseek"].chat.side_effect = DeepSeekError("API错误")
        with pytest.raises(DeepSeekError):
            mock_deps["agent"].analyze_stock("000001")

    def test_analyze_stock_hk_market(self, mock_deps):
        """港股市场下的个股分析"""
        mock_deps["config"].get_market.return_value = "HK"
        mock_deps["deepseek"].chat.return_value = "港股分析结果"
        result = mock_deps["agent"].analyze_stock("00700")
        assert result == "港股分析结果"
        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "HK" in prompt or "00700" in prompt


# ═══════════════════════════════════════════════════════════
#  analyze_watchlist 测试
# ═══════════════════════════════════════════════════════════


class TestAnalyzeWatchlist:
    """analyze_watchlist() 测试"""

    def test_analyze_watchlist_with_items(self, mock_deps):
        """有自选股时的组合分析"""
        items = [
            make_watchlist_item("000001", "平安银行", tags="银行,蓝筹"),
            make_watchlist_item("600519", "贵州茅台", tags="白酒,消费"),
        ]
        mock_deps["watchlist"].list_stocks.return_value = items
        mock_deps["deepseek"].chat.return_value = "组合分析结果"

        result = mock_deps["agent"].analyze_watchlist()
        assert result == "组合分析结果"

        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "平安银行" in prompt
        assert "贵州茅台" in prompt
        assert "银行" in prompt  # 标签信息应包含

    def test_analyze_watchlist_empty(self, mock_deps):
        """空自选股时直接返回提示"""
        mock_deps["watchlist"].list_stocks.return_value = []
        result = mock_deps["agent"].analyze_watchlist()
        assert "自选股列表为空" in result
        mock_deps["deepseek"].chat.assert_not_called()

    def test_analyze_watchlist_error(self, mock_deps):
        """Watchlist 异常应向上传递"""
        mock_deps["watchlist"].list_stocks.side_effect = WatchlistError("DB错误")
        with pytest.raises(WatchlistError):
            mock_deps["agent"].analyze_watchlist()


# ═══════════════════════════════════════════════════════════
#  market_overview 测试
# ═══════════════════════════════════════════════════════════


class TestMarketOverview:
    """market_overview() 测试"""

    def test_market_overview_cn(self, mock_deps):
        """A股市场概况"""
        mock_deps["config"].get_market.return_value = "CN"
        mock_deps["deepseek"].chat.return_value = "A股概况"
        result = mock_deps["agent"].market_overview()
        assert result == "A股概况"
        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "CN" in prompt or "A" in prompt

    def test_market_overview_hk(self, mock_deps):
        """港股市场概况"""
        mock_deps["config"].get_market.return_value = "HK"
        mock_deps["deepseek"].chat.return_value = "港股概况"
        result = mock_deps["agent"].market_overview()
        assert result == "港股概况"

    def test_market_overview_deepseek_error(self, mock_deps):
        """DeepSeek异常应向上传递"""
        mock_deps["deepseek"].chat.side_effect = DeepSeekError("超时")
        with pytest.raises(DeepSeekError):
            mock_deps["agent"].market_overview()


# ═══════════════════════════════════════════════════════════
#  错误处理测试
# ═══════════════════════════════════════════════════════════


class TestErrorHandling:
    """handle() 异常处理测试"""

    def test_handle_deepseek_error(self, mock_deps):
        """DeepSeekError → AgentResponse(success=False)"""
        mock_deps["deepseek"].chat_with_memory.side_effect = DeepSeekError("API超时")
        resp = mock_deps["agent"].handle("session1", "分析大盘")
        assert resp.success is False
        assert resp.agent == AgentType.MARKET

    def test_handle_watchlist_error_in_context(self, mock_deps):
        """构建上下文时 WatchlistError 不使整体失败"""
        mock_deps["watchlist"].list_stocks.side_effect = WatchlistError("DB错误")
        resp = mock_deps["agent"].handle("session1", "分析大盘")
        # 上下文构建失败不影响 AI 调用
        assert resp.success is True

    def test_error_response_metadata(self, mock_deps):
        """失败响应包含错误信息"""
        mock_deps["deepseek"].chat_with_memory.side_effect = DeepSeekError("超时")
        resp = mock_deps["agent"].handle("session1", "分析")
        assert "error" in resp.metadata
        assert resp.metadata["error_type"] == "DeepSeekError"


# ═══════════════════════════════════════════════════════════
#  上下文构建测试
# ═══════════════════════════════════════════════════════════


class TestContextBuilding:
    """_build_market_context() 内部方法测试"""

    def test_context_includes_market(self, mock_deps):
        """上下文包含市场信息"""
        ctx = mock_deps["agent"]._build_market_context()
        assert "CN" in ctx

    def test_context_includes_watchlist(self, mock_deps):
        """上下文包含自选股"""
        items = [make_watchlist_item("000001", "平安银行")]
        mock_deps["watchlist"].list_stocks.return_value = items
        ctx = mock_deps["agent"]._build_market_context()
        assert "平安银行" in ctx

    def test_context_handles_watchlist_error(self, mock_deps):
        """自选股异常时包含降级提示"""
        mock_deps["watchlist"].list_stocks.side_effect = WatchlistError("DB错误")
        ctx = mock_deps["agent"]._build_market_context()
        assert "暂不可用" in ctx

    def test_context_empty_watchlist(self, mock_deps):
        """无自选股时不包含股票信息"""
        mock_deps["watchlist"].list_stocks.return_value = []
        ctx = mock_deps["agent"]._build_market_context()
        assert "自选股" not in ctx or "用户自选股" not in ctx


# ═══════════════════════════════════════════════════════════
#  单例与并发测试
# ═══════════════════════════════════════════════════════════


class TestSingletonAndConcurrency:
    """单例和并发测试"""

    def test_singleton(self):
        """验证单例"""
        a1 = get_market_agent()
        a2 = get_market_agent()
        assert a1 is a2

    def test_concurrent_handle(self, mock_deps):
        """并发调用 handle 线程安全"""
        errors: list[Exception] = []

        def call_handle(i: int):
            try:
                resp = mock_deps["agent"].handle(f"session_{i}", "分析大盘")
                assert resp.success is True
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_handle, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_market_switching_reflected(self, mock_deps):
        """切换市场后上下文反映新市场"""
        mock_deps["config"].get_market.return_value = "US"
        ctx = mock_deps["agent"]._build_market_context()
        assert "US" in ctx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
