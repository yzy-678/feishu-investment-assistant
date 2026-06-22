"""
市场问答 Agent

继承 BaseAgent(AgentType.MARKET)，作为系统的核心投资问答模块。
依赖 DeepSeekClient、ConversationMemory、WatchlistManager、ConfigManager。
"""

import logging
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentType, AgentResponse
from src.ai.deepseek import DeepSeekClient, DeepSeekError, get_deepseek
from src.ai.prompts import INVESTMENT_ASSISTANT_SYSTEM_PROMPT
from src.config.manager import ConfigManager, get_config
from src.config.settings import settings
from src.market import MarketDataError, get_market_data_service
from src.memory import ConversationMemory, get_memory
from src.watchlist.manager import WatchlistManager, WatchlistError, get_watchlist

logger = logging.getLogger(__name__)

# ── 匹配关键词 ───────────────────────────────────────────

_HANDLE_KEYWORDS: list[str] = [
    # 市场
    "市场", "大盘", "行情", "指数", "走势", "涨", "跌",
    # 分析
    "分析", "怎么看", "如何看", "评价",
    # 股票
    "股票", "个股", "股价",
    # 板块
    "板块", "行业",
    # 投资
    "机会", "风险", "主线", "热点", "投资", "建议", "推荐", "关注",
    # 自选
    "自选股", "我的自选",
]

UNRELIABLE_QUOTE_MESSAGE = "未获取到可靠实时行情"

REALTIME_QUOTE_RULES = """
行情硬规则：
- 你不得编造任何价格、涨跌幅、成交额、时间。
- 你只能引用【实时行情】中提供的行情数字。
- 如果【实时行情】显示“未获取到可靠实时行情”，不得分析今日走势、当前强弱或盘中表现。
""".strip()


class MarketAgent(BaseAgent):
    """市场问答 Agent

    处理用户的市场分析、个股分析、板块分析等投资相关问题。
    """

    _instance: Optional["MarketAgent"] = None

    def __new__(cls) -> "MarketAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized: bool = False  # type: ignore[assignment]
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.deepseek: DeepSeekClient = get_deepseek()
        self.memory: ConversationMemory = get_memory()
        self.watchlist: WatchlistManager = get_watchlist()
        self.config: ConfigManager = get_config()
        self.market_data = get_market_data_service()
        self._initialized = True
        logger.info("MarketAgent initialized")

    # ── BaseAgent 接口 ────────────────────────────────────

    @property
    def agent_type(self) -> AgentType:
        return AgentType.MARKET

    def can_handle(self, message: str) -> bool:
        """判断是否可处理此消息"""
        if not message or not message.strip():
            return False
        msg = message.strip()
        return any(kw in msg for kw in _HANDLE_KEYWORDS)

    def handle(self, session_id: str, message: str) -> AgentResponse:
        """处理用户消息

        1. 注入市场上下文和自选股信息到记忆
        2. 调用 DeepSeek 获取回复
        3. 返回结果或异常包装
        """
        try:
            # 1. 注入增强上下文
            market_context, reply_quote_block = self._build_market_context_parts(message)
            self.memory.add_message(
                session_id,
                "system",
                INVESTMENT_ASSISTANT_SYSTEM_PROMPT,
            )
            self.memory.add_message(session_id, "system", market_context)

            # 2. 带记忆的 AI 调用
            response = self.deepseek.chat_with_memory(session_id, message)
            if reply_quote_block:
                response = f"{reply_quote_block}\n\n{response}"

            logger.info(
                "MarketAgent handled: session=%s, msg=%.40s, reply=%d chars",
                session_id, message, len(response),
            )
            return AgentResponse(
                success=True,
                agent=AgentType.MARKET,
                message=response,
                metadata={"session_id": session_id},
            )

        except (DeepSeekError, WatchlistError) as exc:
            logger.warning("MarketAgent error: %s: %s", type(exc).__name__, exc)
            return AgentResponse(
                success=False,
                agent=AgentType.MARKET,
                message=f"处理请求时遇到问题，请稍后再试。",
                metadata={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "session_id": session_id,
                },
            )

    # ── 额外接口 ─────────────────────────────────────────

    def analyze_stock(self, symbol: str) -> str:
        """分析指定个股

        构建分析提示词，调用 DeepSeek 获取分析结果。

        Args:
            symbol: 股票代码

        Returns:
            AI 分析文本

        Raises:
            DeepSeekError: API 调用失败
        """
        market = self.config.get_market()
        data_context, quote_valid, _ = self._build_stock_context(symbol, market)
        analysis_points = self._build_stock_analysis_points(quote_valid)
        prompt = (
            f"你是一个专业的股票分析师。请分析股票 {symbol}（市场: {market}）。\\n\\n"
            f"{data_context}\\n\\n"
            f"{analysis_points}\\n\\n"
            f"{self._build_data_note(market)}"
        )
        return self.deepseek.chat([
            {"role": "system", "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

    def analyze_watchlist(self) -> str:
        """分析当前自选股组合

        获取自选股列表，构建组合分析提示词。

        Returns:
            AI 分析文本（自选股为空时直接返回提示）

        Raises:
            WatchlistError: 自选股查询失败
            DeepSeekError: API 调用失败
        """
        items = self.watchlist.list_stocks()
        if not items:
            return "您的自选股列表为空，请先使用\u201c添加自选\u201d指令添加股票。"

        stock_lines = "\n".join(
            f"  - {item.symbol} {item.name}（{item.market}{' | ' + item.tags if item.tags else ''}）"
            for item in items
        )

        market = self.config.get_market()
        market_snapshot = self._build_watchlist_snapshot(items, market)
        prompt = (
            f"你是一个专业的投资组合分析师。请分析以下自选股组合（市场: {market}）：\\n\\n"
            f"{stock_lines}\\n\\n"
            f"{market_snapshot}\\n\\n"
            f"分析要点：\\n"
            f"1. 组合整体特征（行业分布、风格偏好）\\n"
            f"2. 各股近期表现简评\\n"
            f"3. 组合风险集中度分析\\n"
            f"4. 调仓建议（需增配/减配的方向）\\n\\n"
            f"{self._build_data_note(market)}"
        )
        return self.deepseek.chat([
            {"role": "system", "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

    def market_overview(self) -> str:
        """获取市场概况

        基于当前配置的市场类型，请求 DeepSeek 生成大盘概览。

        Returns:
            AI 生成的市场概览文本

        Raises:
            DeepSeekError: API 调用失败
        """
        market = self.config.get_market()
        market_snapshot = self._build_realtime_context(market=market)
        prompt = (
            f"请提供{market}市场今日概况。\\n\\n"
            f"{market_snapshot}\\n\\n"
            f"内容应包括：\\n"
            f"1. 主要指数表现（涨跌幅、成交量）\\n"
            f"2. 涨跌家数统计\\n"
            f"3. 领涨/领跌板块\\n"
            f"4. 市场情绪判断\\n"
            f"5. 明日关注要点\\n\\n"
            f"{self._build_data_note(market)}"
        )
        return self.deepseek.chat([
            {"role": "system", "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

    # ── 内部方法 ─────────────────────────────────────────

    def _build_market_context(self, message: str = "") -> str:
        """构建增强上下文文本。"""
        context, _ = self._build_market_context_parts(message)
        return context

    def _build_market_context_parts(self, message: str = "") -> tuple[str, str]:
        """构建增强上下文文本

        包含当前市场配置和用户自选股信息，
        注入到对话记忆中作为 system 消息。
        """
        market = self.config.get_market()
        parts: list[str] = [
            f"当前关注市场: {market}。",
        ]
        items = []

        try:
            items = self.watchlist.list_stocks()
            if items:
                stock_lines = "\n".join(
                    f"  * {item.symbol} {item.name}"
                    + (f" [{item.tags}]" if item.tags else "")
                    for item in items
                )
                parts.append(f"\\n用户自选股:\\n{stock_lines}")
        except WatchlistError:
            logger.warning("Failed to load watchlist for market context")
            parts.append("\\n(自选股信息暂不可用)")

        focus_symbol = self._extract_focus_symbol(message, items)
        reply_quote_block = ""

        parts.append(self._build_realtime_context(
            market=market,
            watchlist_items=items,
        ))

        if focus_symbol:
            stock_context, _, reply_quote_block = self._build_stock_context(
                focus_symbol, market
            )
            parts.append(stock_context)

        return "\n".join(parts), reply_quote_block

    def _build_realtime_context(
        self,
        market: str,
        watchlist_items=None,
        focus_symbol: str = "",
    ) -> str:
        if settings.data_source.strip().lower() != "eastmoney":
            return "【实时行情】当前仍在使用 mock 数据源。"

        try:
            return self.market_data.build_market_snapshot_text(
                market=market,
                watchlist_items=watchlist_items,
                focus_symbol=focus_symbol,
            )
        except MarketDataError as exc:
            logger.warning("Failed to build realtime market context: %s", exc)
            return f"【实时行情】暂不可用：{exc}"

    def _build_stock_data_context(self, symbol: str, market: str) -> str:
        context, _, _ = self._build_stock_context(symbol, market)
        return context

    def _build_stock_context(self, symbol: str, market: str) -> tuple[str, bool, str]:
        if settings.data_source.strip().lower() != "eastmoney":
            block = self._build_invalid_quote_block()
            return (
                "\n".join([
                    block,
                    "行情状态：当前仍在使用 mock 数据源。",
                    REALTIME_QUOTE_RULES,
                ]),
                False,
                block,
            )

        quote = None
        try:
            quote = self.market_data.get_quote(symbol, market=market)
        except MarketDataError as exc:
            logger.warning("Failed to build stock data context for %s: %s", symbol, exc)
            self._log_quote_state(symbol, None, quote_valid=False)
            block = self._build_invalid_quote_block()
            return (
                "\n".join([
                    block,
                    f"行情状态：{symbol} 行情暂不可用：{exc}",
                    REALTIME_QUOTE_RULES,
                    "分析范围：仅可分析基本面、行业逻辑和风险因素。",
                ]),
                False,
                block,
            )

        quote_valid = self._is_quote_valid(quote)
        self._log_quote_state(symbol, quote, quote_valid=quote_valid)
        block = self._build_quote_block(quote, quote_valid)

        lines = [block, REALTIME_QUOTE_RULES]

        if not quote_valid:
            lines.append("分析范围：仅可分析基本面、行业逻辑和风险因素。")
            return "\n".join(lines), False, block

        bars = self.market_data.get_recent_bars(symbol, market=market, limit=5)
        if bars:
            lines.append("【最近 5 个交易日】")
            for bar in bars:
                lines.append(
                    f"- {bar.trade_date} 收 {bar.close_price:.2f} "
                    f"({bar.change_pct:+.2f}%) 振幅 {bar.amplitude_pct:.2f}%"
                )
        return "\n".join(lines), True, block

    def _build_watchlist_snapshot(self, items, market: str) -> str:
        return self._build_realtime_context(market=market, watchlist_items=items)

    @staticmethod
    def _build_data_note(market: str) -> str:
        if settings.data_source.strip().lower() == "eastmoney" and market == "CN":
            return (
                "注意：请严格基于以上代码提供的实时 A 股数据分析，"
                "不得自行生成行情数字，仅供参考，不构成投资建议。"
            )
        return "注意：当前仍为模拟/降级数据，仅供参考。"

    @staticmethod
    def _build_stock_analysis_points(quote_valid: bool) -> str:
        if quote_valid:
            return (
                "分析要点：\n"
                "1. 先给结论\n"
                "2. 基于【实时行情】解读当前交易状态\n"
                "3. 公司基本概况（主营业务、行业地位）\n"
                "4. 行业竞争格局与公司竞争优势\n"
                "5. 业绩、估值、预期差和催化因素\n"
                "6. 主要风险因素"
            )
        return (
            "分析要点：\n"
            "1. 先说明未获取到可靠实时行情\n"
            "2. 公司基本面（主营业务、行业地位、业绩逻辑）\n"
            "3. 行业逻辑、政策方向和未来预期\n"
            "4. 主要风险因素\n"
            "禁止分析今日走势、当前强弱或盘中表现。"
        )

    def _is_quote_valid(self, quote: Any) -> bool:
        return all(
            self._quote_field_present(quote, field)
            for field in ("price", "change_pct", "timestamp", "source")
        )

    def _build_quote_block(self, quote: Any, quote_valid: bool) -> str:
        if not quote_valid:
            return self._build_invalid_quote_block()

        source = self._quote_value(quote, "source")
        timestamp = self._quote_value(quote, "timestamp")
        price = self._quote_value(quote, "price")
        change_pct = self._quote_value(quote, "change_pct")
        amount = self._quote_value(quote, "amount")

        return "\n".join([
            "【实时行情】",
            f"数据来源：{source}",
            f"数据时间：{timestamp}",
            f"当前价：{self._format_price(price)}",
            f"涨跌幅：{self._format_pct(change_pct)}",
            f"成交额：{self._format_amount(amount)}",
        ])

    @staticmethod
    def _build_invalid_quote_block() -> str:
        return "\n".join([
            "【实时行情】",
            f"数据来源：{UNRELIABLE_QUOTE_MESSAGE}",
            f"数据时间：{UNRELIABLE_QUOTE_MESSAGE}",
            f"当前价：{UNRELIABLE_QUOTE_MESSAGE}",
            f"涨跌幅：{UNRELIABLE_QUOTE_MESSAGE}",
            f"成交额：{UNRELIABLE_QUOTE_MESSAGE}",
        ])

    def _quote_field_present(self, quote: Any, field: str) -> bool:
        value = self._quote_value(quote, field)
        return value not in (None, "")

    @staticmethod
    def _quote_value(quote: Any, field: str) -> Any:
        if quote is None:
            return None

        candidates = [field]
        if field == "timestamp":
            candidates.append("fetched_at")

        for candidate in candidates:
            if isinstance(quote, dict) and candidate in quote:
                value = quote.get(candidate)
                if value not in (None, ""):
                    return value
                continue
            if hasattr(quote, candidate):
                value = getattr(quote, candidate)
                if value not in (None, ""):
                    return value
        return None

    @staticmethod
    def _format_price(value: Any) -> str:
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_pct(value: Any) -> str:
        try:
            return f"{float(value):+.2f}%"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _format_amount(value: Any) -> str:
        if value in (None, ""):
            return "未提供"
        try:
            return f"{float(value) / 100000000:.2f} 亿"
        except (TypeError, ValueError):
            return str(value)

    def _log_quote_state(self, symbol: str, quote: Any, quote_valid: bool) -> None:
        logger.info(
            "MarketAgent quote state: symbol=%s source=%s timestamp=%s price=%s change_pct=%s quote_valid=%s",
            symbol,
            self._quote_value(quote, "source"),
            self._quote_value(quote, "timestamp"),
            self._quote_value(quote, "price"),
            self._quote_value(quote, "change_pct"),
            quote_valid,
        )

    def _extract_focus_symbol(self, message: str, items=None) -> str:
        symbol = self.market_data.extract_symbol(message)
        if symbol:
            return symbol

        watchlist_items = items or []
        try:
            if not watchlist_items:
                watchlist_items = self.watchlist.list_stocks()
        except WatchlistError:
            return ""

        for item in watchlist_items:
            if item.name and item.name in message:
                return item.symbol
        return ""


# ── 全局单例访问函数 ─────────────────────────────────────

_market_agent_instance: Optional[MarketAgent] = None


def get_market_agent() -> MarketAgent:
    """获取 MarketAgent 单例"""
    global _market_agent_instance  # noqa: PLW0603
    if _market_agent_instance is None:
        _market_agent_instance = MarketAgent()
    return _market_agent_instance
