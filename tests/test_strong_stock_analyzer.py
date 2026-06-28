"""StrongStockAnalyzer tests."""

import json
from unittest.mock import MagicMock

from src.market.stock_screener import StockCandidate
from src.market.strong_stock_analyzer import StrongStockAnalyzer
from src.reports.morning_report_service import MorningReportService


class FakeDeepSeek:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, messages, temperature=0.7, max_tokens=None):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self.response


def make_candidate(index, score=None, industry="半导体", reason="趋势多头"):
    return StockCandidate(
        symbol=f"300{index:03d}",
        name=f"股票{index}",
        industry=industry,
        score=float(score if score is not None else 100 - index),
        trend_score=25.0,
        volume_score=25.0,
        sector_score=20.0,
        breakout_score=20.0,
        strength_score=10.0,
        reason=reason,
        reserved={
            "data_source": "StockScreener",
            "data_time": "2026-06-28 08:30:00",
        },
    )


def test_analyze_candidates_accepts_top20_input_and_prompt_is_scoped():
    candidates = [make_candidate(index) for index in range(25)]
    response = json.dumps([
        {
            "symbol": "300000",
            "rank": 1,
            "reason": "评分最高，量价齐升。",
            "risk": "连续上涨后波动放大。",
            "watch_points": "观察板块持续性。",
        },
        {
            "symbol": "300001",
            "rank": 2,
            "reason": "突破平台。",
            "risk": "突破失败回落。",
            "watch_points": "观察成交额。",
        },
        {
            "symbol": "300002",
            "rank": 3,
            "reason": "强于指数。",
            "risk": "行业分化。",
            "watch_points": "观察相对强度。",
        },
    ], ensure_ascii=False)
    deepseek = FakeDeepSeek(response)
    analyzer = StrongStockAnalyzer(deepseek=deepseek)

    result = analyzer.analyze_candidates(candidates)

    prompt = deepseek.calls[0]["messages"][1]["content"]
    assert "不允许重新选股" in prompt
    assert "不允许重新排序" in prompt
    assert "300002" in prompt
    assert "300003" not in prompt
    assert len(result) == 3


def test_analyze_candidates_keeps_rule_order_and_candidate_scores():
    candidates = [make_candidate(index, score=80 - index) for index in range(20)]
    response = json.dumps([
        {
            "symbol": "300005",
            "rank": 1,
            "score": 999,
            "reason": "AI 选择排序，但分数不能改。",
            "risk": "风险说明。",
            "watch_points": "观察点。",
        },
        {
            "symbol": "300003",
            "rank": 2,
            "score": 888,
            "reason": "解释。",
            "risk": "风险。",
            "watch_points": "观察。",
        },
        {
            "symbol": "300001",
            "rank": 3,
            "score": 777,
            "reason": "解释。",
            "risk": "风险。",
            "watch_points": "观察。",
        },
    ], ensure_ascii=False)

    result = StrongStockAnalyzer(deepseek=FakeDeepSeek(response)).analyze_candidates(
        candidates,
        limit=3,
    )

    assert [item.symbol for item in result] == ["300000", "300001", "300002"]
    assert [item.rank for item in result] == [1, 2, 3]
    assert result[0].score == candidates[0].score
    assert result[1].score == candidates[1].score
    assert result[2].score == candidates[2].score


def test_missing_fields_are_marked_as_insufficient_data_in_prompt():
    candidate = make_candidate(1, industry="", reason="")
    prompt = StrongStockAnalyzer(deepseek=FakeDeepSeek("[]")).build_prompt([candidate])

    assert "数据不足" in prompt
    assert '"industry": "数据不足"' in prompt
    assert '"reason": "数据不足"' in prompt


def test_prompt_forbids_fabricated_market_data():
    prompt = StrongStockAnalyzer(deepseek=FakeDeepSeek("[]")).build_prompt(
        [make_candidate(1)]
    )

    assert "不允许编造行情数字" in prompt
    assert "不允许编造板块数据" in prompt
    assert "只能引用 StockCandidate 字段" in prompt
    assert "不允许从全市场重新选股" in prompt
    assert "不给买入建议" in prompt


def test_ai_cannot_add_symbols_outside_rule_selected_candidates():
    candidates = [make_candidate(index) for index in range(3)]
    response = json.dumps([
        {
            "symbol": "999999",
            "rank": 1,
            "reason": "不在候选内。",
            "risk": "无效。",
            "watch_points": "无效。",
        },
        {
            "symbol": "300000",
            "rank": 2,
            "reason": "有效解释。",
            "risk": "有效风险。",
            "watch_points": "有效观察。",
        },
    ], ensure_ascii=False)

    result = StrongStockAnalyzer(deepseek=FakeDeepSeek(response)).analyze_candidates(
        candidates,
        limit=3,
    )

    assert [item.symbol for item in result] == ["300000", "300001", "300002"]
    assert "999999" not in [item.symbol for item in result]


def test_morning_report_service_displays_top3_strong_stock_picks(monkeypatch):
    agent = MagicMock()
    agent.generate_morning_report.return_value = (
        "## 【早报】2026-06-28\n\n"
        "### 市场概览\n市场震荡。\n\n"
        "### 热点板块\n半导体活跃。\n\n"
        "### 风险提示\n注意缩量。"
    )
    feishu = MagicMock()
    screener = MagicMock()
    candidates = [make_candidate(index) for index in range(20)]
    screener.screen_top_stocks.return_value = candidates

    analyzer = MagicMock()
    pool = MagicMock()
    analyzer.analyze_candidates.return_value = StrongStockAnalyzer(
        deepseek=FakeDeepSeek(json.dumps([
            {
                "symbol": "300000",
                "rank": 1,
                "reason": "量价齐升。",
                "risk": "冲高回落。",
                "watch_points": "观察成交额。",
            },
            {
                "symbol": "300001",
                "rank": 2,
                "reason": "平台突破。",
                "risk": "突破失败。",
                "watch_points": "观察承接。",
            },
            {
                "symbol": "300002",
                "rank": 3,
                "reason": "强于指数。",
                "risk": "板块分化。",
                "watch_points": "观察板块。",
            },
        ], ensure_ascii=False))
    ).analyze_candidates(candidates)
    pool.update_daily_picks.return_value = []
    pool.get_continuous_leaders.return_value = []
    pool.get_new_entries.return_value = []
    pool.get_dropped_stocks.return_value = []

    monkeypatch.setattr(
        "src.reports.morning_report_service.settings.admin_user_open_id",
        "ou_admin",
    )
    service = MorningReportService(
        report_agent=agent,
        feishu_client=feishu,
        stock_screener=screener,
        strong_stock_analyzer=analyzer,
        observation_pool=pool,
    )

    assert service.send() is True

    content = feishu.send_markdown.call_args.args[1]
    analyzer.analyze_candidates.assert_called_once_with(candidates[:3], limit=3)
    assert "### ③ 龙头池（Top3）" in content
    assert "★★★★★ 300000 股票0" in content
    assert "量价齐升" in content
    assert "冲高回落" in content
    assert "### ④ 潜力接力池" in content
