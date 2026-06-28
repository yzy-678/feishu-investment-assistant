"""Explainable V1 technical rating rules."""

from __future__ import annotations

from typing import Optional

from src.market.akshare_source import HistoryBar
from src.rating.rating_models import RatingInputData


def calculate_trend_score(data: RatingInputData) -> tuple[float, list[str], list[str]]:
    """趋势 20 分：均线排列、趋势斜率、是否站上均线。"""
    history = _valid_history(data.history)
    evidence: list[str] = []
    warnings: list[str] = []
    if len(history) < 20:
        return 0.0, evidence, ["趋势评分数据不足：少于20根K线"]

    closes = [bar.close for bar in history]
    latest_close = closes[-1]
    ma5 = moving_average(closes, 5)
    ma10 = moving_average(closes, 10)
    ma20 = moving_average(closes, 20)
    score = 0.0

    if ma5 and ma10 and ma20 and ma5 > ma10 > ma20:
        score += 8
        evidence.append("MA5 > MA10 > MA20，均线多头排列 +8")

    previous_ma5 = moving_average(closes[:-1], 5)
    if ma5 and previous_ma5 and ma5 > previous_ma5:
        score += 4
        evidence.append("MA5 上行，短期趋势斜率向上 +4")

    if ma5 and latest_close > ma5:
        score += 3
        evidence.append("收盘价站上 MA5 +3")
    if ma10 and latest_close > ma10:
        score += 2
        evidence.append("收盘价站上 MA10 +2")
    if ma20 and latest_close > ma20:
        score += 3
        evidence.append("收盘价站上 MA20 +3")

    return min(score, 20.0), evidence, warnings


def calculate_volume_score(data: RatingInputData) -> tuple[float, list[str], list[str]]:
    """量价 20 分：成交量、成交额、量比、放量上涨、价升量增。"""
    history = _valid_history(data.history)
    evidence: list[str] = []
    warnings: list[str] = []
    if len(history) < 6:
        return 0.0, evidence, ["量价评分数据不足：少于6根K线"]

    latest = history[-1]
    previous = history[-2]
    baseline = history[-6:-1]
    avg_volume = average([bar.volume for bar in baseline])
    avg_amount = average([bar.amount for bar in baseline])
    volume_ratio = latest.volume / avg_volume if avg_volume else 0.0
    amount_ratio = latest.amount / avg_amount if avg_amount else 0.0
    score = 0.0

    if volume_ratio >= 1.2:
        score += 5
        evidence.append(f"成交量较5日均量放大 {volume_ratio:.2f} 倍 +5")
    if volume_ratio >= 1.5:
        score += 2
        evidence.append("量比达到明显放量阈值 +2")

    if amount_ratio >= 1.2:
        score += 5
        evidence.append(f"成交额较5日均额放大 {amount_ratio:.2f} 倍 +5")
    if amount_ratio >= 1.5:
        score += 2
        evidence.append("成交额显著增加 +2")

    price_up = latest.close > previous.close
    volume_up = latest.volume > previous.volume
    if price_up and volume_up:
        score += 6
        evidence.append("价升量增 +6")

    return min(score, 20.0), evidence, warnings


def calculate_sector_score(
    data: RatingInputData,
) -> tuple[Optional[float], list[str], list[str]]:
    """板块 20 分：热度、持续性、主线、联动。"""
    evidence: list[str] = []
    warnings: list[str] = []
    score = 0.0
    if not data.sector_available:
        return None, evidence, ["板块评分暂未纳入：行业/概念或板块统计数据暂不可用"]

    split_score_available = (
        data.industry_available
        or data.concepts_available
        or data.industry_score is not None
        or data.concept_score is not None
    )
    legacy_score_available = any(
        item is not None
        for item in (
            data.sector_heat_score,
            data.sector_continuity_score,
            data.is_main_sector,
            data.sector_linkage_score,
            data.industry_change_pct,
        )
    )
    if split_score_available and not legacy_score_available:
        if data.industry_available or data.industry_score is not None:
            points = min(10.0, max(0.0, data.industry_score or 10.0))
            score += points
            evidence.append(f"行业数据可用，板块行业维度 +{points:.1f}")
        else:
            warnings.append("行业数据暂不可用")

        if data.concepts_available or data.concept_score is not None:
            points = min(10.0, max(0.0, data.concept_score or 10.0))
            score += points
            evidence.append(f"概念数据可用，板块概念维度 +{points:.1f}")
        else:
            warnings.append("概念数据暂不可用")

        return round(min(score, 20.0), 2), evidence, warnings

    if data.sector_heat_score is None:
        warnings.append("板块热度数据不足")
    else:
        heat_score = max(0.0, min(data.sector_heat_score, 100.0))
        points = min(6.0, heat_score / 100 * 6)
        score += points
        evidence.append(f"所属板块热度 {heat_score:.1f}/100 +{points:.1f}")

    if data.sector_continuity_score is None:
        warnings.append("板块持续性数据不足")
    else:
        continuity_score = max(0.0, min(data.sector_continuity_score, 100.0))
        points = min(5.0, continuity_score / 100 * 5)
        score += points
        evidence.append(f"板块持续性 {continuity_score:.1f}/100 +{points:.1f}")

    if data.is_main_sector is True:
        score += 5
        evidence.append("属于当前市场主线 +5")
    elif data.is_main_sector is None:
        warnings.append("市场主线识别数据不足")

    linkage = data.sector_linkage_score
    if linkage is None and data.quote is not None and data.industry_change_pct is not None:
        linkage = 100.0 if data.quote.change_pct >= data.industry_change_pct else 0.0
    if linkage is None:
        warnings.append("板块联动数据不足")
    else:
        points = min(4.0, max(0.0, linkage) / 100 * 4)
        score += points
        evidence.append(f"板块联动强度 {linkage:.1f}/100 +{points:.1f}")

    return round(min(score, 20.0), 2), evidence, warnings


def calculate_breakout_score(data: RatingInputData) -> tuple[float, list[str], list[str]]:
    """K线结构 20 分：平台突破、20日新高、放量反包、趋势延续。"""
    history = _valid_history(data.history)
    evidence: list[str] = []
    warnings: list[str] = []
    if len(history) < 21:
        return 0.0, evidence, ["K线结构评分数据不足：少于21根K线"]

    latest = history[-1]
    previous = history[-2]
    prior20 = history[-21:-1]
    prior10 = history[-11:-1]
    score = 0.0

    prior20_high = max(bar.high for bar in prior20)
    if latest.close >= prior20_high:
        score += 6
        evidence.append("收盘价创20日新高 +6")

    prior10_high = max(bar.high for bar in prior10)
    prior10_low = min(bar.low for bar in prior10)
    prior10_avg_close = average([bar.close for bar in prior10])
    platform_range = (
        (prior10_high - prior10_low) / prior10_avg_close
        if prior10_avg_close
        else 1.0
    )
    if platform_range <= 0.08 and latest.close > prior10_high:
        score += 5
        evidence.append("平台收敛后向上突破 +5")

    reversal = (
        previous.close < previous.open
        and latest.close > latest.open
        and latest.close > previous.open
        and latest.volume >= previous.volume * 1.2
    )
    if reversal:
        score += 5
        evidence.append("放量反包前一日阴线 +5")

    rising_days = sum(
        1
        for index in range(max(1, len(history) - 5), len(history))
        if history[index].close > history[index - 1].close
    )
    if rising_days >= 3:
        score += 4
        evidence.append("近5日趋势延续，至少3日收涨 +4")

    return min(score, 20.0), evidence, warnings


def calculate_strength_score(data: RatingInputData) -> tuple[float, list[str], list[str]]:
    """相对强度 20 分：强于指数、行业、资金抱团、相对强势。"""
    evidence: list[str] = []
    warnings: list[str] = []
    score = 0.0
    quote = data.quote
    if quote is None:
        return 0.0, evidence, ["相对强度评分数据不足：缺少实时行情"]

    if quote.change_pct > data.index_change_pct:
        score += 6
        evidence.append(
            f"涨跌幅 {quote.change_pct:.2f}% 强于指数 {data.index_change_pct:.2f}% +6"
        )

    if data.industry_change_pct is None:
        warnings.append("行业涨跌幅数据不足")
    elif quote.change_pct > data.industry_change_pct:
        score += 5
        evidence.append(
            f"涨跌幅 {quote.change_pct:.2f}% 强于行业 {data.industry_change_pct:.2f}% +5"
        )

    if quote.amount >= 1_000_000_000:
        score += 4
        evidence.append("成交额超过10亿，具备资金抱团迹象 +4")
    elif quote.amount <= 0:
        warnings.append("成交额数据不足")

    if quote.change_pct > 0 and quote.turnover_rate >= 3:
        score += 5
        evidence.append("正涨幅且换手率达到活跃阈值，相对强势 +5")

    return min(score, 20.0), evidence, warnings


def moving_average(values: list[float], window: int) -> Optional[float]:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def average(values: list[float]) -> float:
    clean = [value for value in values if value > 0]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def _valid_history(history: list[HistoryBar]) -> list[HistoryBar]:
    return [
        bar
        for bar in history
        if bar.close > 0 and bar.high > 0 and bar.low > 0
    ]
