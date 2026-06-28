"""Morning report delivery service."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Optional, Protocol

from src.agents.report_agent import get_report_agent
from src.bot.client import get_feishu_client
from src.config.settings import settings
from src.db.models import ObservationPoolEntry
from src.market.observation_pool import (
    get_observation_pool_manager,
)
from src.market.layered_observation import (
    LayeredObservationBuilder,
    LayeredObservationReport,
    PotentialRelayPick,
    SectorObservation,
)
from src.market.strong_stock_analyzer import StrongStockPick
from src.market.stock_screener import StockCandidate

logger = logging.getLogger(__name__)


class ReportAgentProtocol(Protocol):
    def generate_morning_report(self) -> str:
        """Generate a raw morning report."""
        ...


class FeishuClientProtocol(Protocol):
    def send_markdown(self, receive_id: str, markdown: str) -> dict:
        """Send markdown content to Feishu."""
        ...


class StockScreenerProtocol(Protocol):
    def screen_top_stocks(self, limit: int = 20) -> list[StockCandidate]:
        ...


class StrongStockAnalyzerProtocol(Protocol):
    def analyze_candidates(
        self,
        candidates: list[StockCandidate],
        limit: int = 3,
    ) -> list[StrongStockPick]:
        ...


class ObservationPoolProtocol(Protocol):
    def update_daily_picks(
        self,
        picks: list[StrongStockPick],
    ) -> list[ObservationPoolEntry]:
        ...

    def get_continuous_leaders(
        self,
        min_days: int = 2,
    ) -> list[ObservationPoolEntry]:
        ...

    def get_new_entries(self) -> list[ObservationPoolEntry]:
        ...

    def get_dropped_stocks(self) -> list[ObservationPoolEntry]:
        ...


class LayeredObservationBuilderProtocol(Protocol):
    def build(self) -> LayeredObservationReport:
        ...


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "市场早报": ("市场早报", "市场概览", "市场概况", "市场综述"),
    "热点板块": ("热点板块", "热门板块"),
    "风险提示": ("风险提示", "风险提醒"),
}

SECTION_ORDER: tuple[str, ...] = ("市场早报", "热点板块", "风险提示")


class MorningReportService:
    """Generate and push the daily 08:30 morning report."""

    def __init__(
        self,
        report_agent: Optional[ReportAgentProtocol] = None,
        feishu_client: Optional[FeishuClientProtocol] = None,
        stock_screener: Optional[StockScreenerProtocol] = None,
        strong_stock_analyzer: Optional[StrongStockAnalyzerProtocol] = None,
        observation_pool: Optional[ObservationPoolProtocol] = None,
        layered_observation_builder: Optional[LayeredObservationBuilderProtocol] = None,
    ) -> None:
        self.report_agent = report_agent or get_report_agent()
        self.feishu_client = feishu_client or get_feishu_client()
        self.stock_screener = stock_screener
        self.strong_stock_analyzer = strong_stock_analyzer
        self.observation_pool = observation_pool
        self.layered_observation_builder = layered_observation_builder

    def send(self, receive_id: Optional[str] = None) -> bool:
        """Generate the Sprint1 morning report and push it to Feishu."""
        target = (receive_id or settings.admin_user_open_id).strip()
        if not target:
            logger.info(
                "Morning report skipped: send_status=skipped error_message=%s",
                "admin_user_open_id is empty",
            )
            return False

        raw_report = self.report_agent.generate_morning_report()
        content = self.build_push_content(raw_report)
        self.feishu_client.send_markdown(target, content)
        logger.info("Morning report sent: send_status=success")
        return True

    def build_push_content(self, raw_report: str) -> str:
        """Keep only Sprint1 sections in the pushed report."""
        sections = _extract_markdown_sections(raw_report)
        layered_report = self.build_layered_observations()
        lines = ["## 每日早报"]

        lines.extend(["", "### ① 市场总览"])
        lines.extend(_format_market_overview(sections, layered_report.sectors))

        lines.extend(["", "### ② 今日主线板块"])
        lines.extend(_format_main_sectors(layered_report.sectors))

        lines.extend(["", "### ③ 龙头池（Top3）"])
        lines.extend(_format_dragon_pool(layered_report.dragon_pool))

        lines.extend(["", "### ④ 潜力接力池"])
        lines.extend(_format_potential_pool(layered_report.potential_pool))

        lines.extend(["", "### ⑤ 连续跟踪池"])
        lines.extend(_format_observation_pool(self._get_observation_pool()))

        lines.extend(["", "### ⑥ 淘汰池"])
        lines.extend(_format_dropped_pool(self._get_observation_pool()))

        lines.extend(["", "### ⑦ 风险提示"])
        lines.extend(_format_final_risk(sections))

        return "\n".join(lines).strip()

    def build_layered_observations(self) -> LayeredObservationReport:
        try:
            builder = self.layered_observation_builder or LayeredObservationBuilder(
                stock_screener=self.stock_screener,
                strong_stock_analyzer=self.strong_stock_analyzer,
            )
            report = builder.build()
            pool_entries = self._get_observation_pool().update_daily_picks(
                report.observation_picks
            )
            return _enrich_layered_report_with_pool_days(report, pool_entries)
        except Exception as exc:
            logger.warning("Layered observation unavailable: %s", exc)
            return LayeredObservationReport(
                sectors=[],
                dragon_pool=[],
                potential_pool=[],
                observation_picks=[],
            )

    def get_strong_stocks(self) -> list[StrongStockPick]:
        """Backward-compatible access to the rule-selected dragon pool."""
        return self.build_layered_observations().dragon_pool

    def _get_observation_pool(self) -> ObservationPoolProtocol:
        return self.observation_pool or get_observation_pool_manager()


_morning_report_service: Optional[MorningReportService] = None


def get_morning_report_service() -> MorningReportService:
    """Return the MorningReportService singleton."""
    global _morning_report_service  # noqa: PLW0603
    if _morning_report_service is None:
        _morning_report_service = MorningReportService()
    return _morning_report_service


def get_strong_stocks() -> list[StrongStockPick]:
    """Return Top3 strong stock observation picks."""
    return get_morning_report_service().get_strong_stocks()


def _extract_markdown_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_title: Optional[str] = None

    for line in markdown.splitlines():
        heading = _parse_heading(line)
        if heading:
            current_title = heading
            sections.setdefault(current_title, [])
            continue

        if current_title is not None:
            sections[current_title].append(line)

    return {
        _normalize_title(title): "\n".join(lines).strip()
        for title, lines in sections.items()
    }


def _parse_heading(line: str) -> Optional[str]:
    match = re.match(r"^\s{0,3}#{2,6}\s+(.+?)\s*$", line)
    if not match:
        return None
    return _normalize_title(match.group(1))


def _normalize_title(title: str) -> str:
    normalized = re.sub(r"^[\d一二三四五六七八九十]+[、.)）]\s*", "", title.strip())
    normalized = normalized.strip("#* `【】[]()（）:：- ")
    return normalized


def _find_section_body(
    sections: dict[str, str],
    aliases: tuple[str, ...],
) -> str:
    for alias in aliases:
        body = sections.get(_normalize_title(alias), "")
        if body:
            return body
    return ""


def _remove_strong_stock_lines(text: str) -> str:
    lines = [line for line in text.splitlines() if "强势股" not in line]
    return "\n".join(lines).strip()


def _format_market_overview(
    sections: dict[str, str],
    sectors: list[SectorObservation],
) -> list[str]:
    market = _find_section_body(sections, SECTION_ALIASES["市场早报"])
    risk = _find_section_body(sections, SECTION_ALIASES["风险提示"])
    focus = "、".join(item.name for item in sectors[:5]) or "数据不足，等待板块扫描"
    return [
        f"- 三大指数预判：{market or '数据不足，暂不编造指数判断。'}",
        "- 隔夜重要消息：公告/研报/新闻数据源暂未接入，暂不编造消息。",
        f"- 今日风险提示：{risk or '数据不足，需等待实时行情确认。'}",
        f"- 今日关注方向：{focus}",
    ]


def _enrich_layered_report_with_pool_days(
    report: LayeredObservationReport,
    pool_entries: list[ObservationPoolEntry],
) -> LayeredObservationReport:
    days_by_symbol = {
        item.symbol: item.consecutive_days
        for item in pool_entries
    }
    dragon_pool = [
        replace(
            pick,
            reserved={
                **pick.reserved,
                "consecutive_days": days_by_symbol.get(pick.symbol, 1),
            },
        )
        for pick in report.dragon_pool
    ]
    return replace(report, dragon_pool=dragon_pool)


def _format_main_sectors(sectors: list[SectorObservation]) -> list[str]:
    if not sectors:
        return ["暂无板块扫描结果。"]

    lines: list[str] = []
    for sector in sectors:
        lines.append(f"- {sector.name}")
        lines.append(f"  - 热度评分：{sector.heat_score:.1f}")
        lines.append(f"  - 持续性评分：{sector.continuity_score:.1f}")
        lines.append(f"  - 资金活跃度：{sector.capital_activity}")
        lines.append(f"  - 当前市场主线：{'是' if sector.is_main_line else '否'}")
        lines.append(f"  - 建议继续关注：{'是' if sector.should_watch else '否'}")
    return lines


def _format_dragon_pool(picks: list[StrongStockPick]) -> list[str]:
    if not picks:
        return ["暂无龙头池候选，等待规则筛选结果。"]

    lines: list[str] = []
    for pick in picks:
        observation_day = pick.reserved.get("consecutive_days", "数据不足")
        lines.append(
            f"★★★★★ {pick.symbol} {pick.name}（{pick.industry or '主线数据不足'}）"
        )
        lines.append(f"  - 综合评分：{pick.score:.1f}")
        lines.append(f"  - 连续观察：第{observation_day}天")
        lines.append(f"  - 所属主线：{pick.industry or '数据不足'}")
        lines.append(f"  - 为什么强：{pick.reason}")
        lines.append(f"  - 主要风险：{pick.risk}")
    return lines


def _format_potential_pool(picks: list[PotentialRelayPick]) -> list[str]:
    if not picks:
        return ["暂无潜力接力候选，等待平台突破/放量启动信号。"]

    lines: list[str] = []
    for pick in picks:
        lines.append(f"- {pick.symbol} {pick.name}")
        lines.append(f"  - 板块：{pick.industry or '数据不足'}")
        lines.append(f"  - 当前阶段：{pick.stage}")
        lines.append(f"  - 入池原因：{pick.reason}")
    return lines


def _format_observation_pool(pool: ObservationPoolProtocol) -> list[str]:
    try:
        continuous = pool.get_continuous_leaders(min_days=3)
    except Exception as exc:
        logger.warning("Observation pool continuous summary unavailable: %s", exc)
        return ["连续跟踪数据暂不可用。"]

    lines: list[str] = []
    if continuous:
        for item in continuous:
            lines.append(
                f"- {item.name}：连续{item.consecutive_days}天上榜，资金持续性越强排序越靠前"
            )
    else:
        lines.append("- 暂无连续上榜股")
    return lines


def _format_dropped_pool(pool: ObservationPoolProtocol) -> list[str]:
    try:
        dropped = pool.get_dropped_stocks()
    except Exception as exc:
        logger.warning("Observation pool dropped summary unavailable: %s", exc)
        return ["淘汰池数据暂不可用。"]

    if not dropped:
        return ["暂无昨日观察股掉出。"]

    return [
        f"- {item.name}：今日跌出观察池，可能由均线走弱、放量下跌、"
        f"板块退潮或评分下降触发，需观察风格变化。"
        for item in dropped
    ]


def _format_final_risk(sections: dict[str, str]) -> list[str]:
    risk = _find_section_body(sections, SECTION_ALIASES["风险提示"])
    lines = []
    if risk:
        lines.append(f"- 今日风险：{risk}")
    lines.append("- 以上为量化规则筛选出的观察池。")
    lines.append("- AI 只负责解释，不负责直接选股。")
    lines.append("- 内容不是买卖建议。")
    lines.append(
        "- 数据来源：EastMoney（实时行情/资金/个股分析）、AkShare"
        "（全市场扫描/历史K线/均线/技术指标/板块统计）。"
    )
    lines.append("- 未来预留：A-Stock-Data Provider（公告、研报、资金流、一致预期）。")
    return lines
