"""Strong stock candidate analyzer.

AI only explains candidates selected by rules. It must not select, replace, or
re-rank stocks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from src.ai.deepseek import get_deepseek
from src.market.stock_screener import StockCandidate
from src.time_utils import shanghai_now

logger = logging.getLogger(__name__)


class DeepSeekProtocol(Protocol):
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        ...


@dataclass(frozen=True)
class StrongStockPick:
    """Top3 strong stock observation pick."""

    symbol: str
    name: str
    industry: str
    score: float
    rank: int
    reason: str
    risk: str
    watch_points: str
    data_source: str
    data_time: str
    reserved: dict[str, Any] = field(default_factory=dict)


class StrongStockAnalyzer:
    """Explain StockScreener candidates selected by rules."""

    def __init__(self, deepseek: Optional[DeepSeekProtocol] = None) -> None:
        self.deepseek = deepseek or get_deepseek()

    def analyze_candidates(
        self,
        candidates: list[StockCandidate],
        limit: int = 3,
    ) -> list[StrongStockPick]:
        """Return explained observation picks from rule-selected candidates."""
        if not candidates or limit <= 0:
            return []

        scoped_candidates = candidates[:limit]
        prompt = self.build_prompt(scoped_candidates, limit=limit)
        response = self.deepseek.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是投资研究助手，只能解释用户给定的规则选股结果，"
                        "不能替换、增加、删除或重新排序股票。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        rows = _parse_json_rows(response)
        return self._rows_to_picks(rows, scoped_candidates, limit=limit)

    def build_prompt(self, candidates: list[StockCandidate], limit: int = 3) -> str:
        """Build a strongly constrained prompt for candidate explanation."""
        payload = [_candidate_payload(candidate) for candidate in candidates[:20]]
        return (
            "下面股票已经由量化规则选定。请只负责解释，不允许重新选股、"
            f"不允许重新排序，按输入顺序输出最多 {limit} 只。\n\n"
            "强约束：\n"
            "1. 不允许编造行情数字。\n"
            "2. 不允许编造板块数据。\n"
            "3. 所有价格、涨跌幅、成交额、评分只能引用 StockCandidate 字段。\n"
            "4. 如果字段缺失，必须说明数据不足。\n"
            "5. 不给买入建议，只给观察逻辑。\n"
            "6. 不允许从全市场重新选股，只能解释输入候选股 symbol。\n"
            "7. 不允许修改 StockCandidate.score，输出中的 score 会被系统忽略。\n"
            "8. 不允许改变输入顺序。\n\n"
            "每只股票解释必须覆盖：\n"
            "1. 为什么强。\n"
            "2. 是板块带动还是独立抱团。\n"
            "3. 是否有突破/反包/量价齐升。\n"
            "4. 风险在哪里。\n"
            "5. 明天重点观察什么。\n\n"
            "请只输出 JSON 数组，不要输出 Markdown。每项字段：\n"
            "symbol, rank, reason, risk, watch_points。\n\n"
            "StockCandidate 输入：\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _rows_to_picks(
        self,
        rows: list[dict[str, Any]],
        candidates: list[StockCandidate],
        limit: int,
    ) -> list[StrongStockPick]:
        rows_by_symbol = {
            str(row.get("symbol") or "").strip(): row
            for row in rows
            if row.get("symbol")
        }
        picks: list[StrongStockPick] = []

        for candidate in candidates[:limit]:
            row = rows_by_symbol.get(candidate.symbol, {})
            picks.append(
                self._build_pick(
                    candidate=candidate,
                    rank=len(picks) + 1,
                    reason=str(row.get("reason") or candidate.reason or "数据不足。"),
                    risk=str(row.get("risk") or "AI 未返回风险，数据不足。"),
                    watch_points=str(
                        row.get("watch_points") or "AI 未返回观察点，数据不足。"
                    ),
                )
            )

        return picks

    def _build_pick(
        self,
        candidate: StockCandidate,
        rank: int,
        reason: str,
        risk: str,
        watch_points: str,
    ) -> StrongStockPick:
        return StrongStockPick(
            symbol=candidate.symbol,
            name=candidate.name,
            industry=candidate.industry,
            score=candidate.score,
            rank=rank,
            reason=reason,
            risk=risk,
            watch_points=watch_points,
            data_source=str(candidate.reserved.get("data_source") or "StockScreener"),
            data_time=str(
                candidate.reserved.get("data_time")
                or shanghai_now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            reserved={"source_score": candidate.score},
        )


def _candidate_payload(candidate: StockCandidate) -> dict[str, Any]:
    return {
        "symbol": candidate.symbol,
        "name": candidate.name,
        "industry": candidate.industry or "数据不足",
        "score": candidate.score,
        "trend_score": candidate.trend_score,
        "volume_score": candidate.volume_score,
        "sector_score": candidate.sector_score,
        "breakout_score": candidate.breakout_score,
        "strength_score": candidate.strength_score,
        "reason": candidate.reason or "数据不足",
        "reserved": candidate.reserved or {},
    }


def _parse_json_rows(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = _load_json_from_fence(text)
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        rows = data.get("picks") or data.get("items") or []
    else:
        rows = data

    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_json_from_fence(text: str) -> Any:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1).strip())

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    return []


_strong_stock_analyzer: Optional[StrongStockAnalyzer] = None


def get_strong_stock_analyzer() -> StrongStockAnalyzer:
    """Return the StrongStockAnalyzer singleton."""
    global _strong_stock_analyzer  # noqa: PLW0603
    if _strong_stock_analyzer is None:
        _strong_stock_analyzer = StrongStockAnalyzer()
    return _strong_stock_analyzer
