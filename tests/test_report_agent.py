"""ReportAgent 单元测试

覆盖：ReportType 枚举、can_handle 关键词匹配、Prompt 构建、
三种报告生成、handle() 路由、市场切换、自选股上下文、异常处理、并发。
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import AgentType
from src.agents.report_agent import (
    ReportAgent, ReportType, get_report_agent,
)
from src.ai.deepseek import DeepSeekError
from src.db.models import WatchlistItem


# ── 辅助: 创建测试用自选股 ─────────────────────────────

def make_item(symbol: str, name: str, market: str = "a",
              tags: str = "") -> WatchlistItem:
    from datetime import datetime
    return WatchlistItem(
        id=hash(symbol) % 100000, symbol=symbol, name=name,
        market=market, tags=tags, notes="",
        added_at=datetime.now(),
    )


# ── Fixtures ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    ReportAgent._instance = None
    ReportAgent._initialized = False


@pytest.fixture
def mock_deps():
    with (
        patch("src.agents.report_agent.get_deepseek") as mock_ds,
        patch("src.agents.report_agent.get_watchlist") as mock_wl,
        patch("src.agents.report_agent.get_config") as mock_cfg,
    ):
        # DeepSeek mock
        ds = MagicMock()
        ds.chat.return_value = "这是生成的报告内容"
        mock_ds.return_value = ds

        # Watchlist mock
        wl = MagicMock()
        wl.list_stocks.return_value = []
        mock_wl.return_value = wl

        # Config mock
        cfg = MagicMock()
        cfg.get_market.return_value = "CN"
        mock_cfg.return_value = cfg

        agent = ReportAgent()
        yield {"agent": agent, "deepseek": ds, "watchlist": wl, "config": cfg}


# ═══════════════════════════════════════════════════════════
#  ReportType 枚举测试
# ═══════════════════════════════════════════════════════════


class TestReportType:

    def test_enum_values(self):
        assert ReportType.MORNING.value == "morning"
        assert ReportType.NOON.value == "noon"
        assert ReportType.CLOSING.value == "closing"

    def test_display_names(self):
        assert ReportType.MORNING.display_name == "早报"
        assert ReportType.NOON.display_name == "午间观察"
        assert ReportType.CLOSING.display_name == "收盘复盘"

    def test_timeframe(self):
        assert ReportType.MORNING.timeframe == "盘前"
        assert ReportType.NOON.timeframe == "午间"
        assert ReportType.CLOSING.timeframe == "收盘"


# ═══════════════════════════════════════════════════════════
#  can_handle 测试
# ═══════════════════════════════════════════════════════════


class TestCanHandle:

    @pytest.mark.parametrize("msg", [
        "生成早报", "生成午报", "收盘复盘",
        "今天复盘", "查看日报", "生成报告",
        "早盘分析", "午间观察", "收评",
    ])
    def test_can_handle_keywords(self, mock_deps, msg: str):
        assert mock_deps["agent"].can_handle(msg)

    def test_can_handle_not_matched(self, mock_deps):
        assert not mock_deps["agent"].can_handle("市场怎么样")
        assert not mock_deps["agent"].can_handle("分析平安银行")

    def test_can_handle_empty(self, mock_deps):
        assert not mock_deps["agent"].can_handle("")
        assert not mock_deps["agent"].can_handle("   ")

    def test_agent_type(self, mock_deps):
        assert mock_deps["agent"].agent_type == AgentType.REPORT


# ═══════════════════════════════════════════════════════════
#  Prompt 构建测试
# ═══════════════════════════════════════════════════════════


class TestPromptBuilding:

    def test_prompt_includes_market(self, mock_deps):
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        assert "CN" in prompt

    def test_prompt_includes_report_type(self, mock_deps):
        morning = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        assert "早报" in morning
        noon = mock_deps["agent"]._build_report_prompt(ReportType.NOON)
        assert "午间观察" in noon
        closing = mock_deps["agent"]._build_report_prompt(ReportType.CLOSING)
        assert "收盘复盘" in closing

    def test_prompt_includes_sections(self, mock_deps):
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        sections = ["市场概览", "热点板块", "风险提示", "自选股观察", "操作关注点"]
        for section in sections:
            assert section in prompt

    def test_prompt_includes_watchlist(self, mock_deps):
        items = [make_item("000001", "平安银行", tags="银行")]
        mock_deps["watchlist"].list_stocks.return_value = items
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        assert "平安银行" in prompt

    def test_prompt_watchlist_empty(self, mock_deps):
        mock_deps["watchlist"].list_stocks.return_value = []
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        assert "暂无自选股" in prompt

    def test_prompt_watchlist_error_handling(self, mock_deps):
        mock_deps["watchlist"].list_stocks.side_effect = Exception("DB error")
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        assert "暂不可用" in prompt

    def test_prompt_different_markets(self, mock_deps):
        mock_deps["config"].get_market.return_value = "HK"
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.NOON)
        assert "HK" in prompt

    def test_prompt_has_markdown_format(self, mock_deps):
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        assert "##" in prompt
        assert "###" in prompt


# ═══════════════════════════════════════════════════════════
#  generate_report 测试
# ═══════════════════════════════════════════════════════════


class TestGenerateReport:

    def test_generate_morning(self, mock_deps):
        result = mock_deps["agent"].generate_morning_report()
        assert result == "这是生成的报告内容"
        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "早报" in prompt

    def test_generate_noon(self, mock_deps):
        result = mock_deps["agent"].generate_noon_report()
        assert result == "这是生成的报告内容"
        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "午间观察" in prompt

    def test_generate_closing(self, mock_deps):
        result = mock_deps["agent"].generate_closing_report()
        assert result == "这是生成的报告内容"
        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "收盘复盘" in prompt

    def test_generate_via_enum(self, mock_deps):
        for rt in [ReportType.MORNING, ReportType.NOON, ReportType.CLOSING]:
            result = mock_deps["agent"].generate_report(rt)
            assert result == "这是生成的报告内容"

    def test_generate_report_deepseek_error(self, mock_deps):
        mock_deps["deepseek"].chat.side_effect = DeepSeekError("API error")
        with pytest.raises(DeepSeekError):
            mock_deps["agent"].generate_morning_report()

    def test_generate_passes_correct_prompt(self, mock_deps):
        mock_deps["agent"].generate_report(ReportType.MORNING)
        messages = mock_deps["deepseek"].chat.call_args[0][0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_generate_noon_timeframe_in_prompt(self, mock_deps):
        mock_deps["agent"].generate_noon_report()
        prompt = mock_deps["deepseek"].chat.call_args[0][0][0]["content"]
        assert "午间" in prompt


# ═══════════════════════════════════════════════════════════
#  handle 测试
# ═══════════════════════════════════════════════════════════


class TestHandle:

    def test_handle_morning(self, mock_deps):
        resp = mock_deps["agent"].handle("s1", "生成早报")
        assert resp.success is True
        assert resp.agent == AgentType.REPORT
        assert resp.metadata["report_type"] == "morning"

    def test_handle_noon(self, mock_deps):
        resp = mock_deps["agent"].handle("s1", "生成午报")
        assert resp.success is True
        assert resp.metadata["report_type"] == "noon"

    def test_handle_closing(self, mock_deps):
        resp = mock_deps["agent"].handle("s1", "收盘复盘")
        assert resp.success is True
        assert resp.metadata["report_type"] == "closing"

    def test_handle_report_default(self, mock_deps):
        resp = mock_deps["agent"].handle("s1", "日报")
        assert resp.success is True
        assert resp.metadata["report_type"] == "morning"

    def test_handle_deepseek_error(self, mock_deps):
        mock_deps["deepseek"].chat.side_effect = DeepSeekError("API error")
        resp = mock_deps["agent"].handle("s1", "生成早报")
        assert resp.success is False
        assert resp.agent == AgentType.REPORT

    def test_handle_sends_reply_message(self, mock_deps):
        resp = mock_deps["agent"].handle("s1", "早报")
        assert resp.message == "这是生成的报告内容"


# ═══════════════════════════════════════════════════════════
#  自选股上下文测试
# ═══════════════════════════════════════════════════════════


class TestWatchlistContext:

    def test_watchlist_context_with_items(self, mock_deps):
        items = [
            make_item("000001", "平安银行", market="a", tags="银行"),
            make_item("00700", "腾讯控股", market="hk", tags="科技"),
        ]
        mock_deps["watchlist"].list_stocks.return_value = items
        ctx = mock_deps["agent"]._build_watchlist_context()
        assert "平安银行" in ctx
        assert "腾讯控股" in ctx
        assert "A股" in ctx
        assert "港股" in ctx

    def test_watchlist_context_empty(self, mock_deps):
        mock_deps["watchlist"].list_stocks.return_value = []
        ctx = mock_deps["agent"]._build_watchlist_context()
        assert "暂无自选股" in ctx

    def test_watchlist_context_error(self, mock_deps):
        mock_deps["watchlist"].list_stocks.side_effect = Exception("DB error")
        ctx = mock_deps["agent"]._build_watchlist_context()
        assert "暂不可用" in ctx


# ═══════════════════════════════════════════════════════════
#  市场切换测试
# ═══════════════════════════════════════════════════════════


class TestMarketSwitching:

    def test_cn_market(self, mock_deps):
        mock_deps["config"].get_market.return_value = "CN"
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.MORNING)
        assert "CN" in prompt

    def test_hk_market(self, mock_deps):
        mock_deps["config"].get_market.return_value = "HK"
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.NOON)
        assert "HK" in prompt

    def test_us_market(self, mock_deps):
        mock_deps["config"].get_market.return_value = "US"
        prompt = mock_deps["agent"]._build_report_prompt(ReportType.CLOSING)
        assert "US" in prompt


# ═══════════════════════════════════════════════════════════
#  _detect_report_type 测试
# ═══════════════════════════════════════════════════════════


class TestDetectReportType:

    def test_detect_morning(self):
        assert ReportAgent._detect_report_type("早报") == ReportType.MORNING
        assert ReportAgent._detect_report_type("早盘行情") == ReportType.MORNING

    def test_detect_noon(self):
        assert ReportAgent._detect_report_type("午报") == ReportType.NOON
        assert ReportAgent._detect_report_type("午间观察") == ReportType.NOON

    def test_detect_closing(self):
        assert ReportAgent._detect_report_type("收盘复盘") == ReportType.CLOSING
        assert ReportAgent._detect_report_type("今天复盘") == ReportType.CLOSING

    def test_detect_default(self):
        assert ReportAgent._detect_report_type("日报") == ReportType.MORNING
        assert ReportAgent._detect_report_type("生成报告") == ReportType.MORNING
        assert ReportAgent._detect_report_type("未知内容") == ReportType.MORNING


# ═══════════════════════════════════════════════════════════
#  单例与并发测试
# ═══════════════════════════════════════════════════════════


class TestSingletonAndConcurrency:

    def test_singleton(self):
        a1 = get_report_agent()
        a2 = get_report_agent()
        assert a1 is a2

    def test_concurrent_generate(self, mock_deps):
        errors = []

        def generate(i: int):
            try:
                rt = [ReportType.MORNING, ReportType.NOON, ReportType.CLOSING][i % 3]
                result = mock_deps["agent"].generate_report(rt)
                assert result == "这是生成的报告内容"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=generate, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mock_deps["deepseek"].chat.call_count == 15

    def test_concurrent_handle(self, mock_deps):
        errors = []

        def handle_msg(i: int):
            try:
                msgs = ["生成早报", "午报", "收盘复盘"]
                resp = mock_deps["agent"].handle(f"s{i}", msgs[i % 3])
                assert resp.success is True
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=handle_msg, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
