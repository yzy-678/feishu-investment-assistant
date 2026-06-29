"""
MarketAgent 单元测试

测试覆盖：can_handle 关键词匹配、handle 流程、额外接口
（analyze_stock/analyze_watchlist/market_overview）、
Prompt 注入、异常处理、边界条件、并发。
所有外部依赖使用 mock。
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from src.agents.base import AgentType, AgentResponse
from src.agents.market_agent import (
    MarketAgent,
    STOCK_DATA_FAILURE_MESSAGE,
    get_market_agent,
)
from src.ai.deepseek import DeepSeekError
from src.ai.prompts import INVESTMENT_ASSISTANT_SYSTEM_PROMPT
from src.watchlist.manager import WatchlistError
from src.db.models import WatchlistItem
from src.market import (
    HistoryBar,
    MACDSnapshot,
    MASnapshot,
    MarketDataError,
    QuoteSnapshot,
    StockInfo,
)
from src.providers.base import ProviderResult
from src.rating import DataQualityItem, DataQualityReport, InvestmentRating, RatingLevel
from src.rating import SectorContext


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
        patch("src.agents.market_agent.get_market_data_service") as mock_mds,
        patch("src.agents.market_agent.get_rating_engine") as mock_rating_engine,
        patch("src.agents.market_agent.settings.data_source", "eastmoney"),
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

        # Market data mock
        mock_mds_instance = MagicMock()
        mock_mds_instance.build_market_snapshot_text.return_value = (
            "【实时 A 股快照】\n"
            "数据时间（Asia/Shanghai）：2026-06-22 09:30:00\n"
            "主要指数：\n"
            "  - 上证指数 3400.00 (+0.80%, +27.00)"
        )
        mock_mds_instance.extract_symbol.return_value = None
        mock_mds_instance.format_quote_detail.return_value = (
            "000001 平安银行 10.52 (-2.41%)"
        )
        mock_mds_instance.get_quote.return_value = QuoteSnapshot(
            symbol="000001",
            name="平安银行",
            price=10.52,
            change=-0.05,
            change_pct=-0.48,
            open_price=10.55,
            high_price=10.60,
            low_price=10.50,
            prev_close=10.57,
            volume=100000,
            amount=98000000,
            amplitude_pct=0.95,
            turnover_rate=0.42,
            fetched_at="2026-06-22 10:00:00",
            source="EastMoney",
            timestamp="2026-06-22 10:00:00",
            data_age_seconds=0,
        )
        mock_mds_instance.get_recent_bars.return_value = [
            MagicMock(trade_date="2026-06-18", close_price=10.52, change_pct=-2.41, amplitude_pct=2.32)
        ]
        mock_mds_instance.get_history.return_value = [
            HistoryBar(
                date=f"2026-04-{day:02d}",
                open=10 + index * 0.1,
                high=10.5 + index * 0.1,
                low=9.8 + index * 0.1,
                close=10 + index * 0.1,
                volume=100000 + index,
                amount=10000000 + index,
            )
            for index, day in enumerate(range(1, 31), start=0)
        ]
        mock_mds_instance.get_ma.return_value = MASnapshot(
            symbol="000001",
            MA5=10.80,
            MA10=10.70,
            MA20=10.55,
            MA60=10.20,
        )
        mock_mds_instance.get_macd.return_value = MACDSnapshot(
            symbol="000001",
            DIF=0.12,
            DEA=0.08,
            MACD=0.08,
        )
        mock_mds_instance.get_stock_info.return_value = StockInfo(
            symbol="000001",
            name="平安银行",
            industry="银行",
            concepts=["互联金融", "破净股"],
        )
        mock_mds.return_value = mock_mds_instance

        rating = InvestmentRating(
            symbol="000001",
            name="平安银行",
            total_score=88,
            rating_level=RatingLevel.A,
            trend_score=18,
            volume_score=17,
            sector_score=18,
            breakout_score=17,
            strength_score=18,
            previous_score=84,
            score_change=4,
            change_direction="⬆",
            change_reasons=[
                "放量突破/结构改善，K线结构评分提升 +2.0。",
                "板块热度或主线联动提升，板块评分提升 +1.0。",
                "成交量/成交额或价升量增改善，量价评分提升 +1.0。",
            ],
            summary="趋势较强",
            warning=(
                "当前评级仅基于已接入的行情、技术和量价数据，"
                "不包含未接入的新闻、公告、财报和资金流数据。"
            ),
            timestamp="2026-06-22 10:00:00",
            data_source="EastMoney, AkShare",
        )
        mock_rating_engine_instance = MagicMock()
        mock_rating_engine_instance.evaluate.return_value = rating
        mock_rating_engine.return_value = mock_rating_engine_instance

        agent = MarketAgent()
        agent.provider_manager = SimpleNamespace(
            get_sector=lambda symbol: ProviderResult.success(
                SectorContext(
                    name="平安银行",
                    industry="银行",
                    concepts=["互联金融", "破净股"],
                    data_source="AkShare",
                ),
                "AkShare",
            ),
            providers=[
                SimpleNamespace(
                    sector_source=SimpleNamespace(
                        debug_snapshot=lambda symbol: {
                            "provider": "EastMoneyRawSectorSource",
                            "symbol": symbol,
                            "stock_get": {
                                "status_code": 200,
                                "f127_industry": "银行",
                                "f128_region_sector": "广东板块",
                            },
                            "hot_keyword": {
                                "status_code": 200,
                                "row_count": 2,
                                "concepts": ["互联金融", "破净股"],
                            },
                        }
                    )
                )
            ],
        )
        yield {
            "agent": agent,
            "deepseek": mock_ds_instance,
            "memory": mock_mem_instance,
            "watchlist": mock_wl_instance,
            "config": mock_cfg_instance,
            "market_data": mock_mds_instance,
            "rating_engine": mock_rating_engine_instance,
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

    def test_can_handle_debug_quote(self, mock_deps):
        assert mock_deps["agent"].can_handle("debug quote 300136")
        assert mock_deps["agent"].can_handle("debug sector 003031")
        assert mock_deps["agent"].can_handle("debug quote 有研新材")
        assert mock_deps["agent"].can_handle("/debug 有研新材")

    @pytest.mark.parametrize("msg", [
        "你现在的信息来自哪里",
        "你不是接了通用型AI吗",
        "你是谁",
        "你能做什么",
    ])
    def test_can_handle_general_ai_questions_fall_through(self, mock_deps, msg: str):
        assert not mock_deps["agent"].can_handle(msg)

    def test_can_handle_bare_stock_name_after_symbol_resolution(self, mock_deps):
        assert mock_deps["agent"].can_handle("有研新材")

        mock_deps["market_data"].extract_symbol.assert_not_called()

    @pytest.mark.parametrize("msg", [
        "信维通信今天有什么消息？",
        "商业航天为什么涨？",
        "PCB板块有什么催化？",
        "长川科技最近有什么新闻？",
    ])
    def test_market_agent_lets_news_agent_handle_news_intent(self, mock_deps, msg):
        assert not mock_deps["agent"].can_handle(msg)

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
        """handle 应以本轮临时 system 消息注入上下文"""
        mock_deps["agent"].handle("session1", "分析大盘")
        kwargs = mock_deps["deepseek"].chat_with_memory.call_args.kwargs
        system_text = "\n".join(kwargs["system_messages"])
        assert INVESTMENT_ASSISTANT_SYSTEM_PROMPT in system_text
        assert "当前关注市场: CN" in system_text
        mock_deps["memory"].add_message.assert_not_called()

    def test_handle_injects_watchlist_context(self, mock_deps):
        """有自选股时应注入自选股信息"""
        items = [
            make_watchlist_item("000001", "平安银行", tags="银行"),
            make_watchlist_item("600519", "贵州茅台", tags="白酒"),
        ]
        mock_deps["watchlist"].list_stocks.return_value = items

        mock_deps["agent"].handle("session1", "分析大盘")
        kwargs = mock_deps["deepseek"].chat_with_memory.call_args.kwargs
        ctx_text = "\n".join(kwargs["system_messages"])
        assert "平安银行" in ctx_text
        assert "贵州茅台" in ctx_text

    def test_handle_calls_chat_with_memory(self, mock_deps):
        """确认调用了 chat_with_memory"""
        mock_deps["market_data"].extract_symbol.return_value = "000001"
        mock_deps["agent"].handle("session1", "分析平安银行")
        args = mock_deps["deepseek"].chat_with_memory.call_args.args
        kwargs = mock_deps["deepseek"].chat_with_memory.call_args.kwargs
        assert args == ("session1", "分析平安银行")
        assert "system_messages" in kwargs
        assert kwargs["temperature"] == 0.2

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

    def test_handle_prefixes_code_generated_quote_block(self, mock_deps):
        """个股分析回复应前置代码生成的实时行情块"""
        mock_deps["deepseek"].chat_with_memory.return_value = "AI 只负责解读"

        resp = mock_deps["agent"].handle("session1", "分析 000001")

        assert resp.message.startswith("【数据卡片】")
        assert "股票代码：000001" in resp.message
        assert "数据来源：EastMoney" in resp.message
        assert "数据时间：2026-06-22 10:00:00" in resp.message
        assert "当前价：10.52" in resp.message
        assert "涨跌幅：-0.48%" in resp.message
        assert "成交额：0.98 亿" in resp.message
        assert "MA5/MA20：MA5=10.8000，MA20=10.5500" in resp.message
        assert "MACD：0.0800" in resp.message
        assert "行业：银行" in resp.message
        assert "概念：互联金融、破净股" in resp.message
        assert "📊 AI Investment Rating" in resp.message
        assert "综合评级：A" in resp.message
        assert "当前评分：88 /100" in resp.message
        assert "昨日评分：84" in resp.message
        assert "变化：⬆ +4" in resp.message
        assert "✓ 放量突破/结构改善" in resp.message
        assert "当前评级仅基于已接入的行情、技术和量价数据" in resp.message
        assert "【核心结论】" in resp.message
        assert "【3条逻辑】" in resp.message
        assert "【3条风险】" in resp.message
        assert "AI 只负责解读" in resp.message
        mock_deps["rating_engine"].evaluate.assert_called_with("000001")

    def test_stock_prompt_uses_default_feishu_length_structure(self, mock_deps):
        mock_deps["agent"].handle("session1", "分析 000001")

        kwargs = mock_deps["deepseek"].chat_with_memory.call_args.kwargs
        system_text = "\n".join(kwargs["system_messages"])
        assert "默认个股分析控制在 600~900 字" in system_text
        assert "数据卡片、Investment Rating、核心结论、3条逻辑、3条风险" in system_text

    def test_stock_prompt_uses_brief_length_when_requested(self, mock_deps):
        mock_deps["agent"].handle("session1", "简短分析 000001")

        kwargs = mock_deps["deepseek"].chat_with_memory.call_args.kwargs
        system_text = "\n".join(kwargs["system_messages"])
        assert "个股分析控制在 200 字以内" in system_text

    def test_rating_block_displays_missing_sector_as_not_included(self):
        rating = InvestmentRating(
            symbol="000001",
            name="平安银行",
            total_score=88,
            rating_level=RatingLevel.A,
            trend_score=18,
            volume_score=17,
            sector_score=None,
            breakout_score=17,
            strength_score=18,
            previous_score=84,
            score_change=4,
            change_direction="⬆",
            change_reasons=["放量突破平台。"],
            summary="趋势较强",
            warning="板块评分暂未纳入。当前评级仅基于已接入数据。",
            timestamp="2026-06-22 10:00:00",
            data_source="EastMoney, AkShare",
        )

        block = MarketAgent._format_rating_block(rating)

        assert "板块：暂未纳入" in block
        assert "提示：板块评分暂未纳入。" in block

    def test_rating_block_displays_partial_sector_as_partially_included(self):
        rating = InvestmentRating(
            symbol="000001",
            name="平安银行",
            total_score=88,
            rating_level=RatingLevel.A,
            trend_score=18,
            volume_score=17,
            sector_score=10,
            breakout_score=17,
            strength_score=18,
            previous_score=84,
            score_change=4,
            change_direction="⬆",
            change_reasons=["行业数据可用。"],
            summary="趋势较强",
            warning="概念数据暂不可用，板块评分部分纳入。",
            timestamp="2026-06-22 10:00:00",
            data_source="EastMoneyRaw",
            reserved={"sector_status": "部分纳入"},
        )

        block = MarketAgent._format_rating_block(rating)

        assert "板块：部分纳入" in block
        assert "提示：板块评分暂未纳入。" not in block

    def test_rating_block_displays_data_quality_summary(self):
        rating = InvestmentRating(
            symbol="000001",
            name="平安银行",
            total_score=88,
            rating_level=RatingLevel.A,
            trend_score=18,
            volume_score=17,
            sector_score=10,
            breakout_score=17,
            strength_score=18,
            previous_score=84,
            score_change=4,
            change_direction="⬆",
            change_reasons=["行业数据可用。"],
            summary="趋势较强",
            warning="当前评级仅基于已接入数据。",
            timestamp="2026-06-22 10:00:00",
            data_source="EastMoney",
            data_quality=DataQualityReport(
                items=[
                    DataQualityItem(
                        name="历史K线",
                        source="EastMoney",
                        status="cache",
                        cache_hit=True,
                    )
                ],
                missing_dimensions=["板块评分"],
            ),
        )

        block = MarketAgent._format_rating_block(rating)

        assert "数据质量：使用缓存；未纳入：板块评分" in block

    def test_handle_removes_llm_generated_quote_lines(self, mock_deps):
        """最终回复不得保留 LLM 生成/篡改的行情字段"""
        mock_deps["deepseek"].chat_with_memory.return_value = (
            "【实时行情】\n"
            "数据来源：LLM\n"
            "数据时间：2099-01-01 00:00:00\n"
            "当前价：999.99\n"
            "涨跌幅：+99.99%\n\n"
            "【技术分析】\n"
            "MA5=999\n\n"
            "🏭 行业属性\n"
            "所属行业：LLM行业\n\n"
            "🧠 AI综合判断\n"
            "资金偏弱，见上方实时行情区。"
        )

        resp = mock_deps["agent"].handle("session1", "分析 000001")

        assert "当前价：10.52" in resp.message
        assert "涨跌幅：-0.48%" in resp.message
        assert "LLM" not in resp.message
        assert "2099-01-01" not in resp.message
        assert "999.99" not in resp.message
        assert "+99.99%" not in resp.message
        assert "LLM行业" not in resp.message
        assert "MA5=999" not in resp.message
        assert "资金偏弱" in resp.message

    def test_handle_keeps_analysis_sentences_with_market_terms(self, mock_deps):
        """分析正文提到成交额/均线/MACD 时，不应被误删。"""
        mock_deps["deepseek"].chat_with_memory.return_value = (
            "🧠 AI综合判断\n"
            "结论：有研新材近60日涨幅接近190%，当前单日成交额高达83.76亿，这是典型的主升浪加速段。\n\n"
            "逻辑：股价已大幅脱离MA20和MA60等中长期均线，MACD处于高位正值，说明多头动能仍然强劲。\n\n"
            "⚠ 风险提示\n"
            "单日83亿成交额是典型筹码交换信号，股价远离均线，技术上有强烈的回踩MA5甚至MA20的需求。"
        )

        resp = mock_deps["agent"].handle("session1", "分析 000001")

        assert "近60日涨幅接近190%" in resp.message
        assert "当前单日成交额高达83.76亿" in resp.message
        assert "MA20和MA60" in resp.message
        assert "MACD处于高位正值" in resp.message
        assert "单日83亿成交额是典型筹码交换信号" in resp.message
        assert "回踩MA5甚至MA20" in resp.message

    def test_handle_removes_only_llm_quote_blocks_but_keeps_following_analysis(
        self,
        mock_deps,
    ):
        """即使 LLM 先输出重复行情块，后面的分析正文也要完整保留。"""
        mock_deps["deepseek"].chat_with_memory.return_value = (
            "【实时行情】\n"
            "数据来源：LLM\n"
            "数据时间：2099-01-01 00:00:00\n"
            "当前价：999.99\n"
            "涨跌幅：+99.99%\n\n"
            "【技术分析】\n"
            "MA5=999\n"
            "MACD=999\n\n"
            "【行业属性】\n"
            "所属行业：LLM行业\n"
            "所属概念：LLM概念\n\n"
            "🧠 AI综合判断\n"
            "逻辑：股价已大幅脱离MA20和MA60，MACD处于高位正值，但这属于分析解释，不应被删除。\n\n"
            "⚠ 风险提示\n"
            "风险：单日83亿成交额若无法延续，可能形成高位震荡。"
        )

        resp = mock_deps["agent"].handle("session1", "分析 000001")

        assert "LLM行业" not in resp.message
        assert "999.99" not in resp.message
        assert "MA5=999" not in resp.message
        assert "MACD=999" not in resp.message
        assert "股价已大幅脱离MA20和MA60" in resp.message
        assert "MACD处于高位正值" in resp.message
        assert "单日83亿成交额若无法延续" in resp.message

    def test_handle_blocks_suspect_llm_garbled_text(self, mock_deps):
        """LLM 分析段出现疑似乱码/错字时，不能直接发给用户。"""
        garbled_a = "冲" + "啥"
        garbled_b = chr(0x66E6) + "及"
        garbled_c = "簿" + "仟"
        mock_deps["deepseek"].chat_with_memory.return_value = (
            "🧠 AI综合判断\n"
            f"结论：短期情绪征动的加速{garbled_a}医制，{garbled_b}锡材、{garbled_c}等关键材料。\n\n"
            "⚠ 风险提示\n"
            "注意波动。"
        )

        resp = mock_deps["agent"].handle("session1", "分析 000001")

        assert garbled_a not in resp.message
        assert garbled_b not in resp.message
        assert garbled_c not in resp.message
        assert "AI 分析文本质量校验未通过" in resp.message
        assert resp.message.startswith("【数据卡片】")

    def test_handle_stock_uses_market_service_and_provider_manager_data(self, mock_deps):
        """个股分析应通过兼容服务取行情技术，并通过 ProviderManager 取板块。"""
        mock_deps["agent"].handle("session1", "分析 000001")

        mock_deps["market_data"].get_quote.assert_called_with(
            "000001",
            market="CN",
        )
        mock_deps["market_data"].get_history.assert_called_with(
            "000001",
            period=60,
        )
        mock_deps["market_data"].get_ma.assert_called_once_with("000001")
        mock_deps["market_data"].get_macd.assert_called_once_with("000001")

        kwargs = mock_deps["deepseek"].chat_with_memory.call_args.kwargs
        prompt_context = "\n".join(kwargs["system_messages"])
        assert "【数据卡片】" in prompt_context
        assert "【实时行情】" in prompt_context
        assert "【技术分析】" in prompt_context
        assert "【行业属性】" in prompt_context
        assert "MA5=10.8000" in prompt_context
        assert "DIF=0.1200" in prompt_context
        assert "所属概念：互联金融、破净股" in prompt_context

    def test_admin_can_debug_quote_without_calling_deepseek(self, mock_deps):
        with patch(
            "src.agents.market_agent.settings.admin_user_open_id",
            "ou_admin",
        ):
            resp = mock_deps["agent"].handle(
                "ou_admin",
                "debug quote 300136",
            )

        assert resp.success is True
        assert "【Debug 股票解析】" in resp.message
        assert "识别股票代码：300136" in resp.message
        assert "是否成功：是" in resp.message
        assert "价格：10.52" in resp.message
        assert "涨跌幅：-0.48" in resp.message
        assert "数据时间：2026-06-22 10:00:00" in resp.message
        assert "quote_valid：true" in resp.message
        assert "missing_fields：[]" in resp.message
        mock_deps["market_data"].get_quote.assert_called_once_with(
            "300136",
            market="CN",
        )
        mock_deps["deepseek"].chat_with_memory.assert_not_called()

    def test_admin_can_debug_sector_without_calling_deepseek(self, mock_deps):
        mock_deps["agent"].provider_manager = SimpleNamespace(
            get_sector=lambda symbol: ProviderResult.success(
                SectorContext(
                    name="中瓷电子",
                    industry="通信设备",
                    concepts=["先进封装", "商业航天"],
                    data_source="EastMoneyRaw, EastMoneyHotKeyword",
                ),
                "EastMoneyRaw, EastMoneyHotKeyword",
            ),
            providers=[
                SimpleNamespace(
                    sector_source=SimpleNamespace(
                        debug_snapshot=lambda symbol: {
                            "provider": "EastMoneyRawSectorSource",
                            "symbol": symbol,
                            "stock_get": {
                                "status_code": 200,
                                "f58_name": "中瓷电子",
                                "f127_industry": "通信设备",
                                "f128_region_sector": "河北板块",
                            },
                            "hot_keyword": {
                                "status_code": 200,
                                "row_count": 2,
                                "concepts": ["先进封装", "商业航天"],
                            },
                        }
                    )
                )
            ],
        )

        with patch(
            "src.agents.market_agent.settings.admin_user_open_id",
            "ou_admin",
        ):
            resp = mock_deps["agent"].handle(
                "ou_admin",
                "debug sector 003031",
            )

        assert resp.success is True
        assert "【Sector Debug】" in resp.message
        assert "股票代码：003031" in resp.message
        assert "industry：通信设备" in resp.message
        assert "concepts：先进封装、商业航天" in resp.message
        assert "provider：EastMoneyRaw, EastMoneyHotKeyword" in resp.message
        assert "available：true" in resp.message
        assert "raw_response 摘要" in resp.message
        assert "f127_industry" in resp.message
        mock_deps["deepseek"].chat_with_memory.assert_not_called()

    def test_non_admin_cannot_debug_quote(self, mock_deps):
        with patch(
            "src.agents.market_agent.settings.admin_user_open_id",
            "ou_admin",
        ):
            resp = mock_deps["agent"].handle(
                "ou_other",
                "debug quote 300136",
            )

        assert resp.success is False
        assert "无权" in resp.message
        mock_deps["market_data"].get_quote.assert_not_called()
        mock_deps["deepseek"].chat_with_memory.assert_not_called()

    def test_debug_quote_reports_missing_fields(self, mock_deps):
        mock_deps["market_data"].get_quote.return_value = SimpleNamespace(
            symbol="300136",
            source="EastMoney",
            price=52.1,
            change_pct=1.25,
        )

        with patch(
            "src.agents.market_agent.settings.admin_user_open_id",
            "ou_admin",
        ):
            resp = mock_deps["agent"].handle(
                "ou_admin",
                "debug quote 300136",
            )

        assert "quote_valid：false" in resp.message
        assert "missing_fields：['timestamp']" in resp.message

    def test_debug_quote_reports_stale_quote(self, mock_deps):
        mock_deps["market_data"].get_quote.return_value = SimpleNamespace(
            symbol="300136",
            source="EastMoney",
            timestamp="2026-06-24 09:50:00",
            fetched_at="2026-06-24 10:00:00",
            data_age_seconds=600,
            is_trading_session=True,
            price=52.1,
            change_pct=1.25,
        )

        with patch(
            "src.agents.market_agent.settings.admin_user_open_id",
            "ou_admin",
        ):
            resp = mock_deps["agent"].handle(
                "ou_admin",
                "debug quote 300136",
            )

        assert "quote_valid：false" in resp.message
        assert "错误信息：stale_quote" in resp.message
        assert "missing_fields：[]" in resp.message

    def test_admin_can_debug_stock_name_without_calling_deepseek(self, mock_deps):
        mock_deps["market_data"].get_quote.return_value = QuoteSnapshot(
            symbol="600206",
            name="有研新材",
            price=60.76,
            change=4.61,
            change_pct=8.19,
            open_price=55.0,
            high_price=61.63,
            low_price=55.0,
            prev_close=56.15,
            volume=100000,
            amount=8376000000,
            amplitude_pct=11.8,
            turnover_rate=9.2,
            fetched_at="2026-06-26 16:12:00",
            source="EastMoney",
            timestamp="2026-06-26 16:11:50",
            data_age_seconds=0,
        )
        mock_deps["market_data"].get_stock_info.return_value = StockInfo(
            symbol="600206",
            name="有研新材",
            industry="小金属",
            concepts=["靶材", "稀土永磁"],
        )
        with patch(
            "src.agents.market_agent.settings.admin_user_open_id",
            "ou_admin",
        ):
            resp = mock_deps["agent"].handle("ou_admin", "/debug 有研新材")

        assert resp.success is True
        assert "原始输入：有研新材" in resp.message
        assert "清洗后输入：有研新材" in resp.message
        assert "识别股票名称：有研新材" in resp.message
        assert "识别股票代码：600206" in resp.message
        assert "行业：小金属" in resp.message
        assert "概念：靶材、稀土永磁" in resp.message
        mock_deps["deepseek"].chat_with_memory.assert_not_called()

    def test_unrecognized_stock_returns_clear_message(self, mock_deps):
        resp = mock_deps["agent"].handle("session1", "查一下不存在股票")

        assert resp.success is True
        assert resp.message == "未能识别股票，请输入股票代码，例如 600206。"
        mock_deps["deepseek"].chat_with_memory.assert_not_called()

    def test_bare_stock_name_resolves_to_symbol(self, mock_deps):
        resp = mock_deps["agent"].handle("session1", "有研新材")

        assert resp.success is True
        mock_deps["market_data"].get_quote.assert_called_with("600206", market="CN")
        assert "股票名称：平安银行" in resp.message or "股票代码：600206" in resp.message


# ═══════════════════════════════════════════════════════════
#  analyze_stock 测试
# ═══════════════════════════════════════════════════════════


class TestAnalyzeStock:
    """analyze_stock() 测试"""

    def test_analyze_stock_basic(self, mock_deps):
        """个股分析"""
        mock_deps["deepseek"].chat.return_value = "平安银行分析结果"
        result = mock_deps["agent"].analyze_stock("000001")
        assert result.startswith("【数据卡片】")
        assert "数据来源：EastMoney" in result
        assert "当前价：10.52" in result
        assert "MA5/MA20：MA5=10.8000，MA20=10.5500" in result
        assert "行业：银行" in result
        assert "【核心结论】" in result
        assert "【3条逻辑】" in result
        assert "【3条风险】" in result
        assert "平安银行分析结果" in result
        mock_deps["deepseek"].chat.assert_called_once()
        assert mock_deps["deepseek"].chat.call_args.kwargs["temperature"] == 0.2
        # 验证 prompt 中包含股票代码
        messages = mock_deps["deepseek"].chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert INVESTMENT_ASSISTANT_SYSTEM_PROMPT in messages[0]["content"]
        prompt = messages[1]["content"]
        assert "000001" in prompt
        assert "【数据卡片】" in prompt
        assert "【实时行情】" in prompt
        assert "数据来源：EastMoney" in prompt
        assert "【技术分析】" in prompt
        assert "近60日趋势" in prompt
        assert "MA5=10.8000" in prompt
        assert "MA10=10.7000" in prompt
        assert "MA20=10.5500" in prompt
        assert "MA60=10.2000" in prompt
        assert "DIF=0.1200" in prompt
        assert "DEA=0.0800" in prompt
        assert "MACD=0.0800" in prompt
        assert "【行业属性】" in prompt
        assert "所属行业：银行" in prompt
        assert "所属概念：互联金融、破净股" in prompt
        assert "AI 不计算指标" in prompt
        assert "AI 不编造指标" in prompt

    @pytest.mark.parametrize(
        "quote",
        [
            SimpleNamespace(
                symbol="000001",
                name="平安银行",
                change_pct=-0.48,
                amount=98000000,
                timestamp="2026-06-22 10:00:00",
                source="EastMoney",
            ),
            SimpleNamespace(
                symbol="000001",
                name="平安银行",
                price=10.52,
                change_pct=-0.48,
                amount=98000000,
                source="EastMoney",
            ),
            SimpleNamespace(
                symbol="000001",
                name="平安银行",
                price=10.52,
                change_pct=-0.48,
                amount=98000000,
                timestamp="2026-06-22 10:00:00",
            ),
        ],
    )
    def test_invalid_quote_stops_stock_analysis(self, mock_deps, quote):
        """缺失关键行情字段时，若 AkShare 可用仍可降级分析。"""
        mock_deps["market_data"].get_quote.return_value = quote
        mock_deps["deepseek"].chat.return_value = "基本面分析"

        result = mock_deps["agent"].analyze_stock("000001")

        assert result.startswith("【数据卡片】")
        assert "实时行情暂缺" in result
        assert "基本面分析" in result
        mock_deps["deepseek"].chat.assert_called_once()

    def test_quote_fetch_error_stops_stock_analysis(self, mock_deps):
        """EastMoney 返回异常时，若 AkShare 可用仍可降级分析。"""
        mock_deps["market_data"].get_quote.side_effect = MarketDataError(
            "timeout",
            reason="timeout",
        )

        result = mock_deps["agent"].analyze_stock("000001")

        assert result.startswith("【数据卡片】")
        assert "实时行情暂缺" in result
        mock_deps["deepseek"].chat.assert_called_once()

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
        assert "港股分析结果" in result
        messages = mock_deps["deepseek"].chat.call_args[0][0]
        prompt = messages[1]["content"]
        assert "HK" in prompt or "00700" in prompt

    def test_quote_state_logged(self, mock_deps, caplog):
        """日志应记录行情校验关键字段"""
        caplog.set_level("INFO", logger="src.agents.market_agent")

        mock_deps["agent"].analyze_stock("000001")

        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert "symbol=000001" in logs
        assert "source=EastMoney" in logs
        assert "timestamp=2026-06-22 10:00:00" in logs
        assert "price=10.52" in logs
        assert "change_pct=-0.48" in logs
        assert "quote_valid=True" in logs
        assert "missing_fields=[]" in logs

    def test_quote_prompt_and_final_data_logged(self, mock_deps, caplog):
        """日志应记录 quote、prompt 行情块、最终用户可见数据"""
        caplog.set_level("INFO", logger="src.agents.market_agent")
        mock_deps["market_data"].extract_symbol.return_value = "000001"

        mock_deps["agent"].handle("session1", "分析 000001")

        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert "MarketAgent quote data:" in logs
        assert '"price": 10.52' in logs
        assert '"change_pct": -0.48' in logs
        assert "MarketAgent prompt quote data:" in logs
        assert '"quote_block": "【数据卡片】' in logs
        assert "MarketAgent final user data:" in logs
        assert '"final_response": "【数据卡片】' in logs

    def test_missing_quote_fields_logged(self, mock_deps, caplog):
        caplog.set_level("INFO", logger="src.agents.market_agent")
        mock_deps["market_data"].get_quote.return_value = SimpleNamespace(
            symbol="000001",
            price=10.52,
            source="EastMoney",
        )

        mock_deps["agent"].analyze_stock("000001")

        logs = "\n".join(record.getMessage() for record in caplog.records)
        assert "quote_valid=False" in logs
        assert "missing_fields=['change_pct', 'timestamp']" in logs

    def test_fetched_at_cannot_replace_missing_timestamp(self, mock_deps):
        quote = SimpleNamespace(
            symbol="000001",
            price=10.52,
            change_pct=-0.48,
            source="EastMoney",
            fetched_at="2026-06-24 10:00:00",
            data_age_seconds=0,
        )

        assert mock_deps["agent"]._is_quote_valid(quote) is False
        assert mock_deps["agent"]._missing_quote_fields(quote) == ["timestamp"]

    def test_stale_quote_stops_stock_analysis(self, mock_deps):
        quote = SimpleNamespace(
            symbol="000001",
            price=10.52,
            change_pct=-0.48,
            source="EastMoney",
            timestamp="2026-06-24 09:50:00",
            fetched_at="2026-06-24 10:00:00",
            data_age_seconds=600,
            is_trading_session=True,
        )
        mock_deps["market_data"].get_quote.return_value = quote

        result = mock_deps["agent"].analyze_stock("000001")

        assert result.startswith("【数据卡片】")
        assert "实时行情暂缺" in result
        mock_deps["deepseek"].chat.assert_called_once()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("history", []),
            ("ma", MASnapshot(symbol="000001", MA5=10.8, MA10=None, MA20=10.5, MA60=10.2)),
            ("macd", MACDSnapshot(symbol="000001", DIF=0.12, DEA=None, MACD=0.08)),
            ("stock_info", StockInfo(symbol="000001", name="平安银行", industry="银行", concepts=[])),
        ],
    )
    def test_akshare_empty_or_missing_data_stops_stock_analysis(
        self,
        mock_deps,
        field,
        value,
    ):
        """AkShare 任一数据缺失时，EastMoney 可用则不阻断分析。"""
        if field == "history":
            mock_deps["market_data"].get_history.return_value = value
        elif field == "ma":
            mock_deps["market_data"].get_ma.return_value = value
        elif field == "macd":
            mock_deps["market_data"].get_macd.return_value = value
        elif field == "stock_info":
            mock_deps["market_data"].get_stock_info.return_value = value
            mock_deps["agent"].provider_manager = SimpleNamespace(
                get_sector=lambda symbol: ProviderResult.success(
                    SectorContext(
                        name="平安银行",
                        industry="银行",
                        concepts=[],
                        data_source="AkShare",
                    ),
                    "AkShare",
                )
            )

        result = mock_deps["agent"].analyze_stock("000001")

        assert result.startswith("【数据卡片】")
        assert "技术指标暂缺" in result or "数据暂不可用" in result
        mock_deps["deepseek"].chat.assert_called_once()

    def test_stock_analysis_displays_industry_and_missing_concepts_separately(
        self,
        mock_deps,
    ):
        mock_deps["market_data"].get_stock_info.return_value = StockInfo(
            symbol="000001",
            name="平安银行",
            industry="银行",
            concepts=[],
        )
        mock_deps["agent"].provider_manager = SimpleNamespace(
            get_sector=lambda symbol: ProviderResult.success(
                SectorContext(
                    name="平安银行",
                    industry="银行",
                    concepts=[],
                    data_source="AkShare",
                ),
                "AkShare",
            )
        )

        result = mock_deps["agent"].analyze_stock("000001")

        assert "行业：银行" in result
        assert "概念：数据暂不可用" in result
        assert "行业概念暂缺" not in result

    def test_stock_analysis_uses_sector_provider_for_visible_industry(
        self,
        mock_deps,
    ):
        mock_deps["agent"].provider_manager = SimpleNamespace(
            get_sector=lambda symbol: ProviderResult.success(
                SectorContext(
                    name="中瓷电子",
                    industry="通信设备",
                    concepts=[],
                    data_source="EastMoneyRaw",
                ),
                "EastMoneyRaw",
            )
        )

        result = mock_deps["agent"].analyze_stock("003031")

        assert "行业：通信设备" in result
        assert "概念：数据暂不可用" in result
        assert "行业概念暂缺" not in result

    def test_akshare_exception_stops_stock_analysis(self, mock_deps):
        """AkShare 超时/异常时，EastMoney 可用则不阻断分析。"""
        mock_deps["market_data"].get_macd.side_effect = TimeoutError("timeout")

        result = mock_deps["agent"].analyze_stock("000001")

        assert result.startswith("【数据卡片】")
        assert "技术指标暂缺" in result
        mock_deps["deepseek"].chat.assert_called_once()

    def test_handle_stock_data_failure_degrades_with_provider_sector(
        self,
        mock_deps,
    ):
        """普通聊天入口遇到行情/技术失败时，板块 provider 可用则降级返回。"""
        mock_deps["market_data"].get_quote.side_effect = MarketDataError(
            "timeout",
            reason="timeout",
        )
        mock_deps["market_data"].get_history.side_effect = TimeoutError("timeout")
        mock_deps["market_data"].get_ma.side_effect = TimeoutError("timeout")
        mock_deps["market_data"].get_macd.side_effect = TimeoutError("timeout")
        mock_deps["market_data"].get_stock_info.side_effect = TimeoutError("timeout")

        resp = mock_deps["agent"].handle("session1", "分析 000001")

        assert resp.success is True
        assert resp.message.startswith("【数据卡片】")
        assert "实时行情暂缺" in resp.message
        assert "行业：银行" in resp.message
        mock_deps["deepseek"].chat_with_memory.assert_called_once()

    def test_handle_bare_stock_name_degrades_with_provider_sector_on_failure(
        self,
        mock_deps,
    ):
        """裸股票名进入 MarketAgent 后，行情失败但 provider 可用时仍返回结果。"""
        mock_deps["market_data"].get_quote.side_effect = MarketDataError(
            "timeout",
            reason="timeout",
        )
        mock_deps["market_data"].get_history.side_effect = TimeoutError("timeout")
        mock_deps["market_data"].get_ma.side_effect = TimeoutError("timeout")
        mock_deps["market_data"].get_macd.side_effect = TimeoutError("timeout")
        mock_deps["market_data"].get_stock_info.side_effect = TimeoutError("timeout")

        resp = mock_deps["agent"].handle("session1", "有研新材")

        assert resp.message.startswith("【数据卡片】")
        assert "实时行情暂缺" in resp.message
        mock_deps["market_data"].get_quote.assert_called_with(
            "600206",
            market="CN",
        )
        mock_deps["deepseek"].chat_with_memory.assert_called_once()


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
        assert result.startswith("【实时 A 股快照】")
        assert "【分析】" in result
        assert "组合分析结果" in result

        messages = mock_deps["deepseek"].chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert INVESTMENT_ASSISTANT_SYSTEM_PROMPT in messages[0]["content"]
        prompt = messages[1]["content"]
        assert "平安银行" in prompt
        assert "贵州茅台" in prompt
        assert "银行" in prompt  # 标签信息应包含
        assert "实时 A 股快照" in prompt

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
        assert result.startswith("【实时 A 股快照】")
        assert "【分析】" in result
        assert "A股概况" in result
        messages = mock_deps["deepseek"].chat.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert INVESTMENT_ASSISTANT_SYSTEM_PROMPT in messages[0]["content"]
        prompt = messages[1]["content"]
        assert "CN" in prompt or "A" in prompt
        assert "实时 A 股快照" in prompt

    def test_market_overview_hk(self, mock_deps):
        """港股市场概况"""
        mock_deps["config"].get_market.return_value = "HK"
        mock_deps["deepseek"].chat.return_value = "港股概况"
        result = mock_deps["agent"].market_overview()
        assert "港股概况" in result

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
        assert "实时 A 股快照" in ctx

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
