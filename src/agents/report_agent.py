"""
报告生成 Agent

负责生成早报、午间观察、收盘复盘三大定时报告。
注入市场配置、自选股信息，使用 DeepSeek 生成结构化报告。
"""

import logging
from enum import Enum
from typing import Optional

from src.agents.base import BaseAgent, AgentType, AgentResponse
from src.ai.deepseek import DeepSeekError, get_deepseek
from src.ai.prompts import INVESTMENT_ASSISTANT_SYSTEM_PROMPT
from src.config.manager import get_config
from src.config.settings import settings
from src.market import MarketDataError, get_market_data_service
from src.time_utils import shanghai_now
from src.watchlist.manager import WatchlistError, get_watchlist

logger = logging.getLogger(__name__)

_HANDLE_KEYWORDS: list[str] = [
    "日报", "复盘", "早报", "午报", "收盘", "生成报告",
    "早盘", "午盘", "午间", "收评",
]


class ReportType(str, Enum):
    """报告类型"""
    MORNING = "morning"
    """早报（盘前）"""
    NOON = "noon"
    """午间观察（盘中）"""
    CLOSING = "closing"
    """收盘复盘（盘后）"""

    @property
    def display_name(self) -> str:
        names = {
            "morning": "早报",
            "noon": "午间观察",
            "closing": "收盘复盘",
        }
        return names[self.value]

    @property
    def timeframe(self) -> str:
        timeframes = {
            "morning": "盘前",
            "noon": "午间",
            "closing": "收盘",
        }
        return timeframes[self.value]


class ReportAgent(BaseAgent):
    """报告生成 Agent

    生成结构化市场日报，支持三种报告类型。
    """

    _instance: Optional["ReportAgent"] = None

    def __new__(cls) -> "ReportAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized: bool = False  # type: ignore[assignment]
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.deepseek = get_deepseek()
        self.config = get_config()
        self.watchlist = get_watchlist()
        self.market_data = get_market_data_service()
        self._initialized = True
        logger.info("ReportAgent initialized")

    # ── BaseAgent 接口 ────────────────────────────────────

    @property
    def agent_type(self) -> AgentType:
        return AgentType.REPORT

    def can_handle(self, message: str) -> bool:
        if not message or not message.strip():
            return False
        return any(kw in message for kw in _HANDLE_KEYWORDS)

    def handle(self, session_id: str, message: str) -> AgentResponse:
        """根据用户消息类型生成相应报告"""
        try:
            report_type = self._detect_report_type(message)
            content = self.generate_report(report_type)
            logger.info(
                "ReportAgent handled: session=%s, type=%s, %d chars",
                session_id, report_type.value, len(content),
            )
            return AgentResponse(
                success=True,
                agent=AgentType.REPORT,
                message=content,
                metadata={
                    "session_id": session_id,
                    "report_type": report_type.value,
                },
            )
        except DeepSeekError as exc:
            logger.warning("ReportAgent error: %s", exc)
            return AgentResponse(
                success=False,
                agent=AgentType.REPORT,
                message="生成报告时遇到问题，请稍后再试。",
                metadata={"error": str(exc), "error_type": "DeepSeekError"},
            )

    @staticmethod
    def _detect_report_type(message: str) -> "ReportType":
        """从用户消息中检测请求的报告类型"""
        if any(kw in message for kw in ("早报", "早盘", "早间")):
            return ReportType.MORNING
        if any(kw in message for kw in ("午报", "午间", "午盘")):
            return ReportType.NOON
        if any(kw in message for kw in ("收盘", "复盘", "收评", "复盘", "收盘复盘")):
            return ReportType.CLOSING
        return ReportType.MORNING  # 默认早报

    # ── 报告生成 ─────────────────────────────────────────

    def generate_morning_report(self) -> str:
        """生成早报"""
        return self.generate_report(ReportType.MORNING)

    def generate_noon_report(self) -> str:
        """生成午间观察"""
        return self.generate_report(ReportType.NOON)

    def generate_closing_report(self) -> str:
        """生成收盘复盘"""
        return self.generate_report(ReportType.CLOSING)

    def generate_report(self, report_type: ReportType) -> str:
        """生成指定类型的报告

        Args:
            report_type: 报告类型

        Returns:
            报告文本

        Raises:
            DeepSeekError: API 调用失败
        """
        prompt = self._build_report_prompt(report_type)
        return self.deepseek.chat([
            {"role": "system", "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

    # ── Prompt 构建 ──────────────────────────────────────

    def _build_report_prompt(self, report_type: ReportType) -> str:
        """构建带上下文注入的报告生成提示词"""
        market = self.config.get_market()
        now = shanghai_now()
        today = now.date().isoformat()
        now_text = now.strftime("%Y-%m-%d %H:%M:%S")
        watchlist_ctx = self._build_watchlist_context()
        market_snapshot = self._build_market_snapshot_context(market)
        data_note = self._build_data_note(market)

        prompt = (
            f"请生成{report_type.display_name}（{report_type.timeframe}）。\n\n"
            f"【基本信息】\n"
            f"日期：{today}\n"
            f"当前时间（Asia/Shanghai）：{now_text}\n"
            f"当前市场：{market}\n\n"
            f"{market_snapshot}\n\n"
            f"{watchlist_ctx}\n\n"
            f"请按以下 Markdown 格式输出报告：\n\n"
            f"## 【{report_type.display_name}】{today}\n\n"
            f"### 市场概览\n"
            f"（主要指数表现、成交量、涨跌比等）\n\n"
            f"### 热点板块\n"
            f"（领涨/领跌板块、资金流向）\n\n"
            f"### 风险提示\n"
            f"（潜在风险因素、需要关注的事件）\n\n"
            f"### 自选股观察\n"
            f"（自选股整体表现、重点关注个股）\n\n"
            f"### 操作关注点\n"
            f"（今日或明日操作建议、关键价位）\n\n"
            f"{data_note}"
        )

        return prompt

    # ── 上下文构建 ──────────────────────────────────────

    def _build_watchlist_context(self) -> str:
        """构建自选股上下文"""
        try:
            items = self.watchlist.list_stocks()
            if not items:
                return "（用户暂无自选股）"

            lines = ["【用户自选股】"]
            for item in items:
                market_label = {"a": "A股", "hk": "港股", "us": "美股"}.get(
                    item.market, item.market
                )
                tag_info = f" [{item.tags}]" if item.tags else ""
                lines.append(f"  - {item.symbol} {item.name}（{market_label}）{tag_info}")
            return "\n".join(lines)

        except Exception:
            logger.warning("Failed to load watchlist for report prompt")
            return "（自选股信息暂不可用）"

    def _build_market_snapshot_context(self, market: str) -> str:
        """构建实时市场快照上下文。"""
        if settings.data_source.strip().lower() != "eastmoney":
            return "【市场快照】当前仍在使用 mock 数据源。"

        try:
            items = self.watchlist.list_stocks()
            return self.market_data.build_market_snapshot_text(
                market=market,
                watchlist_items=items,
            )
        except (MarketDataError, WatchlistError, Exception) as exc:
            logger.warning("Failed to build report market snapshot: %s", exc)
            return f"【市场快照】实时行情暂不可用：{exc}"

    @staticmethod
    def _build_data_note(market: str) -> str:
        if settings.data_source.strip().lower() == "eastmoney" and market == "CN":
            return "注意：以上分析基于实时 A 股快照生成，仅供参考，不构成投资建议。"
        return "注意：当前仍为模拟/降级数据，分析仅供参考，不构成投资建议。"


# ── 全局单例访问函数 ─────────────────────────────────────

_report_agent_instance: Optional["ReportAgent"] = None


def get_report_agent() -> "ReportAgent":
    """获取 ReportAgent 单例"""
    global _report_agent_instance  # noqa: PLW0603
    if _report_agent_instance is None:
        _report_agent_instance = ReportAgent()
    return _report_agent_instance
