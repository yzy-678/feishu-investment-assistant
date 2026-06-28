"""
市场问答 Agent

继承 BaseAgent(AgentType.MARKET)，作为系统的核心投资问答模块。
依赖 DeepSeekClient、ConversationMemory、WatchlistManager、ConfigManager。
"""

import json
import logging
import re
from typing import Any, Optional

from src.agents.base import BaseAgent, AgentType, AgentResponse
from src.agents.news_agent import is_news_intent
from src.ai.deepseek import DeepSeekClient, DeepSeekError, get_deepseek
from src.ai.prompts import INVESTMENT_ASSISTANT_SYSTEM_PROMPT
from src.config.manager import ConfigManager, get_config
from src.config.settings import settings
from src.market import (
    MAX_QUOTE_AGE_SECONDS,
    MarketDataError,
    get_market_data_service,
)
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
STOCK_DATA_FAILURE_MESSAGE = "实时数据获取失败，暂时无法分析"
STOCK_UNRECOGNIZED_MESSAGE = "未能识别股票，请输入股票代码，例如 600206。"

REALTIME_QUOTE_RULES = """
行情硬规则：
- 你不得编造任何价格、涨跌幅、成交额、时间。
- 实时行情、技术分析、行业属性由程序直接渲染，你只输出“🧠 AI综合判断”和“⚠ 风险提示”。
- 你不得输出“📈 实时行情”“📊 技术分析”“🏭 行业属性”标题，不得复述或改写当前价、涨跌幅、成交额、数据时间、MA、MACD 等指标数字。
- 如需引用程序数据，请写“见上方实时行情/技术分析/行业属性”，不要写具体数字。
- 如果【实时行情】显示“未获取到可靠实时行情”，不得分析今日走势、当前强弱或盘中表现。
""".strip()

TECHNICAL_ANALYSIS_RULES = """
技术指标硬规则：
- AI 不计算指标。
- AI 不编造指标。
- AI 只能解释程序在【技术分析】中提供的近60日趋势、MA5、MA10、MA20、MA60、DIF、DEA、MACD。
- 如果某项显示“未获取到可靠数据”，不得基于该项做判断。
""".strip()

DEBUG_QUOTE_PATTERN = re.compile(r"^debug\s+quote\s+(.+)$", re.IGNORECASE)
DEBUG_SLASH_PATTERN = re.compile(r"^/debug\s+(.+)$", re.IGNORECASE)


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
        if DEBUG_QUOTE_PATTERN.fullmatch(msg) or DEBUG_SLASH_PATTERN.fullmatch(msg):
            return True
        if is_news_intent(msg):
            return False
        if any(kw in msg for kw in _HANDLE_KEYWORDS):
            return True
        if self._looks_like_stock_name_query(msg):
            return True
        return False

    def handle(self, session_id: str, message: str) -> AgentResponse:
        """处理用户消息

        1. 注入市场上下文和自选股信息到记忆
        2. 调用 DeepSeek 获取回复
        3. 返回结果或异常包装
        """
        try:
            stripped_message = message.strip()
            debug_match = (
                DEBUG_QUOTE_PATTERN.fullmatch(stripped_message)
                or DEBUG_SLASH_PATTERN.fullmatch(stripped_message)
            )
            if debug_match:
                return self._handle_debug_quote(
                    session_id,
                    debug_match.group(1),
                )

            if self._requires_stock_resolution(stripped_message):
                symbol = self._extract_focus_symbol(stripped_message)
                if not symbol:
                    return AgentResponse(
                        success=True,
                        agent=AgentType.MARKET,
                        message=STOCK_UNRECOGNIZED_MESSAGE,
                        metadata={
                            "session_id": session_id,
                            "data_available": False,
                            "reason": "symbol_unrecognized",
                        },
                    )

            # 1. 构建本轮增强上下文（不写入长期记忆，避免历史行情复活）
            market_context, reply_quote_block, stock_data_valid = (
                self._build_market_context_parts(message)
            )
            if not stock_data_valid:
                failure_message = self._build_stock_failure_message(market_context)
                self._log_final_user_data(
                    session_id,
                    reply_quote_block,
                    failure_message,
                )
                return AgentResponse(
                    success=True,
                    agent=AgentType.MARKET,
                    message=failure_message,
                    metadata={
                        "session_id": session_id,
                        "data_available": False,
                    },
                )

            # 2. 带记忆的 AI 调用
            llm_response = self.deepseek.chat_with_memory(
                session_id,
                message,
                system_messages=[
                    INVESTMENT_ASSISTANT_SYSTEM_PROMPT,
                    market_context,
                ],
            )
            response = self._compose_final_response(
                quote_block=reply_quote_block,
                llm_response=llm_response,
            )
            self._log_final_user_data(session_id, reply_quote_block, response)

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
        data_context, data_valid, quote_block = self._build_stock_context(
            symbol,
            market,
        )
        if not data_valid:
            failure_message = self._build_stock_failure_message(data_context)
            self._log_final_user_data(
                "analyze_stock",
                quote_block,
                failure_message,
            )
            return failure_message

        analysis_points = self._build_stock_analysis_points(data_valid)
        prompt = (
            f"你是一个专业的股票分析师。请分析股票 {symbol}（市场: {market}）。\\n\\n"
            f"{data_context}\\n\\n"
            f"{analysis_points}\\n\\n"
            f"{self._build_data_note(market)}"
        )
        llm_response = self.deepseek.chat([
            {"role": "system", "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        return self._compose_final_response(quote_block, llm_response)

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
        llm_response = self.deepseek.chat([
            {"role": "system", "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        return self._compose_final_response(market_snapshot, llm_response)

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
        llm_response = self.deepseek.chat([
            {"role": "system", "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        return self._compose_final_response(market_snapshot, llm_response)

    # ── 内部方法 ─────────────────────────────────────────

    def _build_market_context(self, message: str = "") -> str:
        """构建增强上下文文本。"""
        context, _, _ = self._build_market_context_parts(message)
        return context

    def _build_market_context_parts(self, message: str = "") -> tuple[str, str, bool]:
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
        realtime_context = self._build_realtime_context(
            market=market,
            watchlist_items=items,
        )
        self._log_prompt_quote_data(
            symbol=focus_symbol or "market_snapshot",
            quote_block=realtime_context,
            context_type="market_snapshot",
        )
        parts.append(realtime_context)
        reply_quote_block = realtime_context
        stock_data_valid = True

        if focus_symbol:
            stock_context, stock_data_valid, reply_quote_block = self._build_stock_context(
                focus_symbol, market
            )
            parts.append(stock_context)

        return "\n".join(parts), reply_quote_block, stock_data_valid

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
            display_block = self._build_stock_display_block(
                quote_block=block,
                technical_block=self._build_unavailable_technical_block(),
                industry_block=self._build_unavailable_industry_block(symbol),
            )
            self._log_quote_data(
                symbol,
                None,
                quote_valid=False,
                failure_reason="mock_data_source",
            )
            self._log_prompt_quote_data(
                symbol=symbol,
                quote_block=display_block,
                quote_valid=False,
                failure_reason="mock_data_source",
            )
            return (
                "\n".join([
                    block,
                    self._build_unavailable_technical_block(),
                    self._build_unavailable_industry_block(symbol),
                    "行情状态：当前仍在使用 mock 数据源。",
                    REALTIME_QUOTE_RULES,
                    TECHNICAL_ANALYSIS_RULES,
                ]),
                False,
                display_block,
            )

        quote = None
        try:
            quote = self.market_data.get_quote(symbol, market=market)
        except MarketDataError as exc:
            logger.warning("Failed to build stock data context for %s: %s", symbol, exc)
            self._log_quote_state(
                symbol,
                None,
                quote_valid=False,
                failure_reason=getattr(exc, "reason", "unknown"),
            )
            block = self._build_invalid_quote_block()
            failure_reason = getattr(exc, "reason", "unknown")
            technical_block = self._build_unavailable_technical_block()
            industry_block = self._build_unavailable_industry_block(symbol)
            display_block = self._build_stock_display_block(
                quote_block=block,
                technical_block=technical_block,
                industry_block=industry_block,
            )
            self._log_quote_data(
                symbol,
                None,
                quote_valid=False,
                failure_reason=failure_reason,
            )
            self._log_prompt_quote_data(
                symbol=symbol,
                quote_block=display_block,
                quote_valid=False,
                failure_reason=failure_reason,
            )
            return (
                "\n".join([
                    block,
                    technical_block,
                    industry_block,
                    f"行情状态：{symbol} 行情暂不可用：{exc}",
                    REALTIME_QUOTE_RULES,
                    TECHNICAL_ANALYSIS_RULES,
                    "分析范围：仅可分析基本面、行业逻辑和风险因素。",
                ]),
                False,
                display_block,
            )

        quote_valid = self._is_quote_valid(quote)
        failure_reason = self._quote_failure_reason(quote)
        self._log_quote_state(
            symbol,
            quote,
            quote_valid=quote_valid,
            failure_reason=failure_reason,
        )
        block = self._build_quote_block(quote, quote_valid)
        self._log_quote_data(
            symbol,
            quote,
            quote_valid=quote_valid,
            failure_reason=failure_reason,
        )
        if not quote_valid:
            technical_block = self._build_unavailable_technical_block()
            industry_block = self._build_unavailable_industry_block(symbol)
            display_block = self._build_stock_display_block(
                quote_block=block,
                technical_block=technical_block,
                industry_block=industry_block,
            )
            self._log_prompt_quote_data(
                symbol=symbol,
                quote_block=display_block,
                quote_valid=False,
                failure_reason=failure_reason or "invalid_quote",
            )
            return (
                "\n".join([
                    block,
                    technical_block,
                    industry_block,
                    REALTIME_QUOTE_RULES,
                    TECHNICAL_ANALYSIS_RULES,
                    "行情状态：实时行情无效，禁止继续分析。",
                ]),
                False,
                display_block,
            )

        technical_block, technical_valid, technical_failure = (
            self._build_technical_block_with_status(symbol)
        )
        industry_block, industry_valid, industry_failure = (
            self._build_industry_block_with_status(symbol)
        )
        data_valid = technical_valid and industry_valid
        data_failure_reason = ";".join(
            reason for reason in (technical_failure, industry_failure) if reason
        )
        display_block = self._build_stock_display_block(
            quote_block=block,
            technical_block=technical_block,
            industry_block=industry_block,
        )
        self._log_prompt_quote_data(
            symbol=symbol,
            quote_block=display_block,
            quote_valid=data_valid,
            failure_reason=data_failure_reason,
        )

        lines = [
            block,
            technical_block,
            industry_block,
            REALTIME_QUOTE_RULES,
            TECHNICAL_ANALYSIS_RULES,
            self._build_stock_reply_format_rules(),
        ]

        if not data_valid:
            lines.append("行情状态：AkShare 数据不完整，禁止继续分析。")
            return "\n".join(lines), False, display_block

        return "\n".join(lines), True, display_block

    def _build_technical_block(self, symbol: str) -> str:
        block, _, _ = self._build_technical_block_with_status(symbol)
        return block

    def _build_technical_block_with_status(self, symbol: str) -> tuple[str, bool, str]:
        history = self._safe_market_data_call(
            "history",
            symbol,
            lambda: self.market_data.get_history(symbol, period=60),
            default=[],
        )
        ma = self._safe_market_data_call(
            "ma",
            symbol,
            lambda: self.market_data.get_ma(symbol),
            default=None,
        )
        macd = self._safe_market_data_call(
            "macd",
            symbol,
            lambda: self.market_data.get_macd(symbol),
            default=None,
        )
        failure_reasons = self._technical_failure_reasons(history, ma, macd)

        block = "\n".join([
            "【技术分析】",
            f"近60日趋势：{self._format_history_trend(history)}",
            (
                "均线："
                f"MA5={self._format_optional_number(self._value(ma, 'MA5'))}，"
                f"MA10={self._format_optional_number(self._value(ma, 'MA10'))}，"
                f"MA20={self._format_optional_number(self._value(ma, 'MA20'))}，"
                f"MA60={self._format_optional_number(self._value(ma, 'MA60'))}"
            ),
            (
                "MACD："
                f"DIF={self._format_optional_number(self._value(macd, 'DIF'))}，"
                f"DEA={self._format_optional_number(self._value(macd, 'DEA'))}，"
                f"MACD={self._format_optional_number(self._value(macd, 'MACD'))}"
            ),
        ])
        return block, not failure_reasons, ";".join(failure_reasons)

    def _build_industry_block(self, symbol: str) -> str:
        block, _, _ = self._build_industry_block_with_status(symbol)
        return block

    def _build_industry_block_with_status(self, symbol: str) -> tuple[str, bool, str]:
        stock_info = self._safe_market_data_call(
            "stock_info",
            symbol,
            lambda: self.market_data.get_stock_info(symbol),
            default=None,
        )
        failure_reasons = self._stock_info_failure_reasons(stock_info)
        name = self._value(stock_info, "name") or "未获取到可靠数据"
        industry = self._value(stock_info, "industry") or "未获取到可靠数据"
        concepts = self._value(stock_info, "concepts") or []
        if isinstance(concepts, (list, tuple)) and concepts:
            concept_text = "、".join(str(item) for item in concepts if item)
        else:
            concept_text = "未获取到可靠数据"

        block = "\n".join([
            "【行业属性】",
            f"股票名称：{name}",
            f"所属行业：{industry}",
            f"所属概念：{concept_text}",
        ])
        return block, not failure_reasons, ";".join(failure_reasons)

    def _technical_failure_reasons(self, history: Any, ma: Any, macd: Any) -> list[str]:
        reasons: list[str] = []
        bars = list(history or [])
        history_fields = ("date", "open", "high", "low", "close", "volume", "amount")
        if len(bars) < 2:
            reasons.append("history_empty")
        else:
            missing_history = [
                field
                for bar in bars
                for field in history_fields
                if self._value(bar, field) in (None, "")
            ]
            if missing_history:
                reasons.append("history_missing_fields")

        ma_fields = ("MA5", "MA10", "MA20", "MA60")
        if ma is None:
            reasons.append("ma_empty")
        elif any(self._value(ma, field) in (None, "") for field in ma_fields):
            reasons.append("ma_missing_fields")

        macd_fields = ("DIF", "DEA", "MACD")
        if macd is None:
            reasons.append("macd_empty")
        elif any(self._value(macd, field) in (None, "") for field in macd_fields):
            reasons.append("macd_missing_fields")

        return reasons

    def _stock_info_failure_reasons(self, stock_info: Any) -> list[str]:
        if stock_info is None:
            return ["stock_info_empty"]

        reasons: list[str] = []
        if self._value(stock_info, "name") in (None, ""):
            reasons.append("stock_info_missing_name")
        if self._value(stock_info, "industry") in (None, ""):
            reasons.append("stock_info_missing_industry")
        concepts = self._value(stock_info, "concepts")
        if not isinstance(concepts, (list, tuple)) or not [
            item for item in concepts if item
        ]:
            reasons.append("stock_info_missing_concepts")
        return reasons

    @staticmethod
    def _build_unavailable_technical_block() -> str:
        return "\n".join([
            "【技术分析】",
            "近60日趋势：未获取到可靠数据",
            "均线：MA5=未获取到可靠数据，MA10=未获取到可靠数据，MA20=未获取到可靠数据，MA60=未获取到可靠数据",
            "MACD：DIF=未获取到可靠数据，DEA=未获取到可靠数据，MACD=未获取到可靠数据",
        ])

    @staticmethod
    def _build_unavailable_industry_block(symbol: str) -> str:
        return "\n".join([
            "【行业属性】",
            f"股票代码：{symbol}",
            "股票名称：未获取到可靠数据",
            "所属行业：未获取到可靠数据",
            "所属概念：未获取到可靠数据",
        ])

    @staticmethod
    def _build_stock_display_block(
        quote_block: str,
        technical_block: str,
        industry_block: str,
    ) -> str:
        return "\n\n".join([
            MarketAgent._replace_block_heading(quote_block, "📈 实时行情"),
            MarketAgent._replace_block_heading(technical_block, "📊 技术分析"),
            MarketAgent._replace_block_heading(industry_block, "🏭 行业属性"),
        ])

    @staticmethod
    def _replace_block_heading(block: str, heading: str) -> str:
        lines = str(block or "").splitlines()
        if not lines:
            return heading
        lines[0] = heading
        return "\n".join(lines)

    @staticmethod
    def _build_stock_reply_format_rules() -> str:
        return "\n".join([
            "回复格式要求：",
            "- 程序已直接渲染 📈 实时行情、📊 技术分析、🏭 行业属性。",
            "- 你只输出以下两部分：",
            "🧠 AI综合判断",
            "⚠ 风险提示",
        ])

    def _safe_market_data_call(
        self,
        data_type: str,
        symbol: str,
        fn,
        default: Any,
    ) -> Any:
        try:
            return fn()
        except Exception as exc:
            logger.warning(
                "MarketAgent AkShare data unavailable: symbol=%s data_type=%s error=%s",
                symbol,
                data_type,
                exc,
            )
            return default

    @staticmethod
    def _format_history_trend(history: Any) -> str:
        bars = list(history or [])
        if len(bars) < 2:
            return "未获取到可靠数据"

        first = bars[0]
        last = bars[-1]
        first_close = MarketAgent._value(first, "close")
        last_close = MarketAgent._value(last, "close")
        if first_close in (None, "", 0) or last_close in (None, ""):
            return "未获取到可靠数据"

        try:
            first_close_float = float(first_close)
            last_close_float = float(last_close)
            change_pct = (
                (last_close_float - first_close_float)
                / first_close_float
                * 100
            )
            highs = [
                float(MarketAgent._value(bar, "high"))
                for bar in bars
                if MarketAgent._value(bar, "high") not in (None, "")
            ]
            lows = [
                float(MarketAgent._value(bar, "low"))
                for bar in bars
                if MarketAgent._value(bar, "low") not in (None, "")
            ]
        except (TypeError, ValueError, ZeroDivisionError):
            return "未获取到可靠数据"

        return (
            f"{MarketAgent._value(first, 'date')} 至 {MarketAgent._value(last, 'date')}，"
            f"收盘 {first_close_float:.2f} → {last_close_float:.2f}，"
            f"区间涨跌幅 {change_pct:+.2f}%"
            + (
                f"，区间高/低 {max(highs):.2f}/{min(lows):.2f}"
                if highs and lows
                else ""
            )
            + f"，样本 {len(bars)} 日"
        )

    @staticmethod
    def _format_optional_number(value: Any) -> str:
        if value in (None, ""):
            return "未获取到可靠数据"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _value(obj: Any, field: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    def _build_watchlist_snapshot(self, items, market: str) -> str:
        return self._build_realtime_context(market=market, watchlist_items=items)

    @staticmethod
    def _build_data_note(market: str) -> str:
        if settings.data_source.strip().lower() == "eastmoney" and market == "CN":
            return (
                "注意：请严格基于以上代码提供的实时 A 股数据分析，"
                "不要在分析部分复述行情数字，仅供参考，不构成投资建议。"
            )
        return "注意：当前仍为模拟/降级数据，仅供参考。"

    @staticmethod
    def _build_stock_analysis_points(quote_valid: bool) -> str:
        if quote_valid:
            return (
                "分析要点：\n"
                "1. 只输出“🧠 AI综合判断”和“⚠ 风险提示”两部分\n"
                "2. 解释程序提供的【实时行情】【技术分析】【行业属性】\n"
                "3. 不计算 MA、MACD、趋势涨跌幅等指标\n"
                "4. 不编造任何指标、行业、概念或实时行情数字\n"
                "5. 如某项数据缺失，只说明该项数据不足"
            )
        return (
            "分析要点：\n"
            "1. 只输出“🧠 AI综合判断”和“⚠ 风险提示”两部分\n"
            "2. 先说明未获取到可靠实时行情\n"
            "3. 只解释程序提供的【技术分析】【行业属性】可用数据\n"
            "4. 不计算或编造任何指标\n"
            "禁止分析今日走势、当前强弱或盘中表现。"
        )

    @staticmethod
    def _build_stock_failure_message(context: str) -> str:
        if "AkShare 数据不完整" in context:
            return f"{STOCK_DATA_FAILURE_MESSAGE}\n失败层级：AkShare\n错误信息：历史/技术/行业数据为空或字段缺失"
        if "行情暂不可用" in context or "实时行情无效" in context:
            return f"{STOCK_DATA_FAILURE_MESSAGE}\n失败层级：EastMoney\n错误信息：实时行情失败、为空、超时或字段缺失"
        if "mock 数据源" in context:
            return f"{STOCK_DATA_FAILURE_MESSAGE}\n失败层级：MarketDataService\n错误信息：未启用 EastMoney 实时行情源"
        return f"{STOCK_DATA_FAILURE_MESSAGE}\n失败层级：MarketDataService\n错误信息：实时数据校验失败"

    def _is_quote_valid(self, quote: Any) -> bool:
        return (
            not self._missing_quote_fields(quote)
            and not self._quote_failure_reason(quote)
        )

    def _missing_quote_fields(self, quote: Any) -> list[str]:
        return [
            field
            for field in ("price", "change_pct", "timestamp", "source")
            if not self._quote_field_present(quote, field)
        ]

    def _quote_failure_reason(self, quote: Any) -> str:
        if quote is None:
            return ""

        explicit_reason = self._quote_value(quote, "failure_reason")
        if explicit_reason:
            return str(explicit_reason)

        data_age_seconds = self._quote_value(quote, "data_age_seconds")
        is_trading_session = bool(
            self._quote_value(quote, "is_trading_session")
        )
        try:
            is_stale = (
                is_trading_session
                and data_age_seconds is not None
                and float(data_age_seconds) > MAX_QUOTE_AGE_SECONDS
            )
        except (TypeError, ValueError):
            is_stale = False
        return "stale_quote" if is_stale else ""

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

        if isinstance(quote, dict) and field in quote:
            value = quote.get(field)
            if value not in (None, ""):
                return value
            return None
        if hasattr(quote, field):
            value = getattr(quote, field)
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

    def _compose_final_response(self, quote_block: str, llm_response: str) -> str:
        """最终回复只允许程序行情块 + 清洗后的分析文本。"""
        analysis = self._sanitize_llm_analysis(llm_response)
        if not quote_block:
            return f"【分析】\n{analysis}"
        if quote_block.strip().startswith("📈 实时行情"):
            return (
                f"{quote_block.strip()}\n\n"
                f"{self._ensure_stock_ai_sections(analysis)}"
            )
        return f"{quote_block.strip()}\n\n【分析】\n{analysis}"

    @staticmethod
    def _ensure_stock_ai_sections(analysis: str) -> str:
        text = str(analysis or "").strip()
        if "🧠 AI综合判断" in text and "⚠ 风险提示" in text:
            return text
        return "\n\n".join([
            "🧠 AI综合判断",
            text or "暂未生成有效分析内容，请以上方程序数据为准。",
            "⚠ 风险提示",
            "以上仅基于程序提供的实时行情、技术指标和行业属性解读，仅供参考，不构成投资建议。",
        ])

    def _sanitize_llm_analysis(self, response: str) -> str:
        """移除 LLM 生成的行情区和行情字段，保留分析内容。"""
        lines = str(response or "").splitlines()
        cleaned: list[str] = []
        skipping_quote_block = False

        for line in lines:
            stripped = line.strip()

            if self._is_llm_quote_heading(stripped):
                skipping_quote_block = True
                continue

            if skipping_quote_block:
                if not stripped:
                    skipping_quote_block = False
                elif stripped.startswith("【") and "实时" not in stripped:
                    skipping_quote_block = False
                else:
                    continue

            if self._looks_like_llm_quote_line(stripped):
                continue

            cleaned.append(line)

        analysis = "\n".join(cleaned).strip()
        analysis = self._strip_analysis_heading(analysis)
        return analysis or "暂未生成有效分析内容，请以上方实时行情区为准。"

    @staticmethod
    def _strip_analysis_heading(text: str) -> str:
        stripped = text.strip()
        return re.sub(r"^【分析】\s*", "", stripped).strip()

    @staticmethod
    def _is_llm_quote_heading(text: str) -> bool:
        if not text:
            return False
        normalized = text.replace(" ", "")
        return (
            normalized.startswith("【实时行情】")
            or normalized.startswith("【实时A股快照】")
            or normalized.startswith("【技术分析】")
            or normalized.startswith("【行业属性】")
            or normalized.startswith("📈实时行情")
            or normalized.startswith("📊技术分析")
            or normalized.startswith("🏭行业属性")
            or normalized.startswith("实时行情")
            or normalized.startswith("实时A股快照")
        )

    @staticmethod
    def _looks_like_llm_quote_line(text: str) -> bool:
        if not text:
            return False

        quote_labels = (
            "数据来源",
            "数据时间",
            "当前价",
            "最新价",
            "涨跌幅",
            "成交额",
            "成交量",
            "EastMoney",
            "近60日趋势",
            "均线",
            "MA5",
            "MA10",
            "MA20",
            "MA60",
            "MACD",
            "DIF",
            "DEA",
            "股票名称",
            "所属行业",
            "所属概念",
        )
        if any(label in text for label in quote_labels):
            return True

        return bool(
            re.search(
                r"(当前|最新|股价|价格|涨跌|成交|截至).{0,20}(\d|[+-])",
                text,
            )
        )

    def _quote_log_payload(
        self,
        symbol: str,
        quote: Any,
        quote_valid: bool,
        failure_reason: str = "",
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "name": self._quote_value(quote, "name"),
            "source": self._quote_value(quote, "source"),
            "timestamp": self._quote_value(quote, "timestamp"),
            "fetched_at": self._quote_value(quote, "fetched_at"),
            "data_age_seconds": self._quote_value(quote, "data_age_seconds"),
            "is_trading_session": self._quote_value(quote, "is_trading_session"),
            "price": self._quote_value(quote, "price"),
            "change": self._quote_value(quote, "change"),
            "change_pct": self._quote_value(quote, "change_pct"),
            "amount": self._quote_value(quote, "amount"),
            "quote_valid": quote_valid,
            "failure_reason": failure_reason or self._quote_failure_reason(quote),
            "missing_fields": self._missing_quote_fields(quote),
        }

    def _log_quote_data(
        self,
        symbol: str,
        quote: Any,
        quote_valid: bool,
        failure_reason: str = "",
    ) -> None:
        logger.info(
            "MarketAgent quote data: %s",
            json.dumps(
                self._quote_log_payload(
                    symbol,
                    quote,
                    quote_valid=quote_valid,
                    failure_reason=failure_reason,
                ),
                ensure_ascii=False,
                default=str,
            ),
        )

    @staticmethod
    def _log_prompt_quote_data(
        symbol: str,
        quote_block: str,
        quote_valid: Optional[bool] = None,
        failure_reason: str = "",
        context_type: str = "quote",
    ) -> None:
        logger.info(
            "MarketAgent prompt quote data: %s",
            json.dumps(
                {
                    "symbol": symbol,
                    "context_type": context_type,
                    "quote_valid": quote_valid,
                    "failure_reason": failure_reason,
                    "quote_block": quote_block,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    @staticmethod
    def _log_final_user_data(
        session_id: str,
        quote_block: str,
        final_response: str,
    ) -> None:
        logger.info(
            "MarketAgent final user data: %s",
            json.dumps(
                {
                    "session": session_id[:8],
                    "quote_block": quote_block,
                    "final_response": final_response,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

    def _log_quote_state(
        self,
        symbol: str,
        quote: Any,
        quote_valid: bool,
        failure_reason: str = "",
    ) -> None:
        missing_fields = self._missing_quote_fields(quote)
        logger.info(
            "MarketAgent quote state: symbol=%s source=%s timestamp=%s "
            "fetched_at=%s data_age_seconds=%s price=%s change_pct=%s "
            "quote_valid=%s failure_reason=%s missing_fields=%s",
            symbol,
            self._quote_value(quote, "source"),
            self._quote_value(quote, "timestamp"),
            self._quote_value(quote, "fetched_at"),
            self._quote_value(quote, "data_age_seconds"),
            self._quote_value(quote, "price"),
            self._quote_value(quote, "change_pct"),
            quote_valid,
            failure_reason or self._quote_failure_reason(quote),
            missing_fields,
        )

    def _handle_debug_quote(self, session_id: str, raw_query: str) -> AgentResponse:
        admin_open_id = settings.admin_user_open_id.strip()
        if not admin_open_id or session_id.strip() != admin_open_id:
            logger.warning(
                "Quote debug denied: session=%s raw_query=%s",
                session_id[:8],
                raw_query,
            )
            return AgentResponse(
                success=False,
                agent=AgentType.MARKET,
                message="无权使用行情调试命令。",
                metadata={"type": "quote_debug", "authorized": False},
            )

        raw_message = str(raw_query or "").strip()
        cleaned_message = self._clean_debug_stock_query(raw_message)
        symbol = self.market_data.extract_symbol(cleaned_message) or (
            cleaned_message if re.fullmatch(r"\d{6}", cleaned_message) else ""
        )
        resolved_name = ""

        quote = None
        eastmoney_error = ""
        try:
            if symbol:
                quote = self.market_data.get_quote(symbol, market="CN")
                resolved_name = str(self._quote_value(quote, "name") or "")
            else:
                eastmoney_error = "symbol_unrecognized"
        except MarketDataError as exc:
            eastmoney_error = f"{getattr(exc, 'reason', 'unknown')}: {exc}"
            logger.warning(
                "Quote debug failed: symbol=%s reason=%s error=%s",
                symbol,
                getattr(exc, "reason", "unknown"),
                exc,
            )

        missing_fields = self._missing_quote_fields(quote)
        failure_reason = eastmoney_error or self._quote_failure_reason(quote)
        quote_valid = not missing_fields and not failure_reason
        self._log_quote_state(
            symbol,
            quote,
            quote_valid,
            failure_reason=failure_reason,
        )
        self._log_quote_data(
            symbol,
            quote,
            quote_valid=quote_valid,
            failure_reason=failure_reason,
        )

        stock_info = None
        ma = None
        macd = None
        akshare_error = ""
        if symbol:
            try:
                stock_info = self.market_data.get_stock_info(symbol)
                ma = self.market_data.get_ma(symbol)
                macd = self.market_data.get_macd(symbol)
                if not resolved_name:
                    resolved_name = str(self._value(stock_info, "name") or "")
                akshare_failures = (
                    self._stock_info_failure_reasons(stock_info)
                    + self._technical_failure_reasons([], ma, macd)
                )
                akshare_failures = [
                    item for item in akshare_failures if item != "history_empty"
                ]
                if akshare_failures:
                    akshare_error = ",".join(akshare_failures)
            except Exception as exc:
                akshare_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "AkShare debug failed: symbol=%s error=%s",
                    symbol,
                    exc,
                )
        else:
            akshare_error = "symbol_unrecognized"

        concepts = self._value(stock_info, "concepts") or []
        concept_text = (
            "、".join(str(item) for item in concepts if item)
            if isinstance(concepts, (list, tuple))
            else ""
        )
        final_card = (
            self._build_quote_block(quote, quote_valid)
            if quote is not None
            else self._build_invalid_quote_block()
        )
        logger.info(
            "MarketAgent debug stock data: %s",
            json.dumps(
                {
                    "raw_message": raw_message,
                    "cleaned_message": cleaned_message,
                    "resolved_symbol": symbol,
                    "resolved_name": resolved_name,
                    "eastmoney_status": not bool(eastmoney_error),
                    "akshare_status": not bool(akshare_error),
                    "quote_valid": quote_valid,
                    "missing_fields": missing_fields,
                },
                ensure_ascii=False,
                default=str,
            ),
        )

        reply = "\n".join([
            "【Debug 股票解析】",
            f"原始输入：{raw_message}",
            f"清洗后输入：{cleaned_message}",
            f"识别股票名称：{resolved_name}",
            f"识别股票代码：{symbol}",
            "",
            "【EastMoney】",
            f"是否成功：{'是' if not eastmoney_error and quote is not None else '否'}",
            f"价格：{self._quote_value(quote, 'price') or ''}",
            f"涨跌幅：{self._quote_value(quote, 'change_pct') or ''}",
            f"数据时间：{self._quote_value(quote, 'timestamp') or ''}",
            f"错误信息：{failure_reason}",
            "",
            "【AkShare】",
            f"是否成功：{'是' if not akshare_error else '否'}",
            f"行业：{self._value(stock_info, 'industry') or ''}",
            f"概念：{concept_text}",
            f"MA5：{self._value(ma, 'MA5') or ''}",
            f"MA20：{self._value(ma, 'MA20') or ''}",
            f"MACD：{self._value(macd, 'MACD') or ''}",
            f"错误信息：{akshare_error}",
            "",
            "【MarketDataService】",
            f"quote_valid：{str(quote_valid).lower()}",
            f"missing_fields：{missing_fields}",
            "最终数据卡片：",
            final_card,
        ])
        return AgentResponse(
            success=True,
            agent=AgentType.MARKET,
            message=reply,
            metadata={
                "type": "quote_debug",
                "symbol": symbol,
                "quote_valid": quote_valid,
                "failure_reason": failure_reason,
                "missing_fields": missing_fields,
            },
        )

    @staticmethod
    def _clean_debug_stock_query(raw_query: str) -> str:
        cleaned = str(raw_query or "").strip()
        cleaned = re.sub(r"^(?:/debug|debug\s+quote)\s+", "", cleaned, flags=re.I)
        return cleaned.strip()

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

    def _requires_stock_resolution(self, message: str) -> bool:
        if not message or is_news_intent(message):
            return False
        if self.market_data.extract_symbol(message):
            return False
        if any(word in message for word in ("大盘", "市场", "板块", "行业", "主线", "热点", "自选")):
            return False
        if any(word in message for word in ("查一下", "查询", "看一下", "看看", "看下")):
            return True
        if self._looks_like_stock_name_query(message):
            return True
        return False

    @staticmethod
    def _looks_like_stock_name_query(message: str) -> bool:
        cleaned = re.sub(r"\s+", "", str(message or "").strip())
        if not cleaned or len(cleaned) > 12:
            return False
        if cleaned in {
            "你好",
            "你好呀",
            "您好",
            "谢谢",
            "早上好",
            "晚上好",
            "分析",
            "启动",
            "暂停",
            "状态",
        }:
            return False
        if any(word in cleaned for word in ("启动", "暂停", "状态", "系统")):
            return False
        return bool(re.search(r"[\u4e00-\u9fff]{2,}", cleaned))


# ── 全局单例访问函数 ─────────────────────────────────────

_market_agent_instance: Optional[MarketAgent] = None


def get_market_agent() -> MarketAgent:
    """获取 MarketAgent 单例"""
    global _market_agent_instance  # noqa: PLW0603
    if _market_agent_instance is None:
        _market_agent_instance = MarketAgent()
    return _market_agent_instance
