"""Sector context provider for the Investment Rating Engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import httpx

from src.market.akshare_source import StockInfo

logger = logging.getLogger(__name__)

EASTMONEY_STOCK_URL = "https://push2.eastmoney.com/api/qt/stock/get"
EASTMONEY_HOT_KEYWORD_URL = (
    "https://emappdata.eastmoney.com/stockrank/getHotStockRankList"
)


class StockInfoProvider(Protocol):
    def get_stock_info(self, symbol: str) -> StockInfo:
        ...


class SectorContextProvider(Protocol):
    def get_sector_context(self, symbol: str) -> "SectorContext":
        ...


@dataclass(frozen=True)
class SectorContext:
    """Industry and concept data used before sector scoring is enabled."""

    name: str = ""
    industry: str = ""
    region_sector: str = ""
    concepts: list[str] = field(default_factory=list)
    data_source: str = ""
    industry_score: Optional[float] = None
    concept_score: Optional[float] = None
    sector_heat_score: Optional[float] = None
    sector_continuity_score: Optional[float] = None
    is_main_sector: Optional[bool] = None
    sector_linkage_score: Optional[float] = None
    warning: str = "行业/概念数据暂不可用，板块评分暂未纳入。"

    @property
    def available(self) -> bool:
        return self.industry_available or self.concepts_available

    @property
    def industry_available(self) -> bool:
        return bool(self.industry)

    @property
    def concepts_available(self) -> bool:
        return bool(self.concepts)

    @property
    def sector_status(self) -> str:
        if self.industry_available and self.concepts_available:
            return "已纳入"
        if self.available:
            return "部分纳入"
        return "暂未纳入"


class EastMoneyRawSectorSource:
    """Read industry and concepts directly from EastMoney raw endpoints."""

    def __init__(
        self,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.timeout = timeout
        self.client = client

    def get_sector_context(self, symbol: str) -> SectorContext:
        industry_context = self._fetch_industry(symbol)
        concepts = self._fetch_concepts(symbol)
        return _merge_contexts(
            SectorContext(data_source="EastMoneyRaw"),
            industry_context,
            SectorContext(
                concepts=concepts,
                data_source="EastMoneyHotKeyword" if concepts else "",
            ),
        )

    def _fetch_industry(self, symbol: str) -> SectorContext:
        market_code = "1" if str(symbol).startswith(("6", "688")) else "0"
        try:
            response = self._get(
                EASTMONEY_STOCK_URL,
                params={
                    "fltt": "2",
                    "invt": "2",
                    "fields": "f57,f58,f127,f128",
                    "secid": f"{market_code}.{symbol}",
                },
            )
            response.raise_for_status()
            data = (response.json() or {}).get("data") or {}
        except Exception as exc:
            logger.warning(
                "EastMoney raw sector industry failed: symbol=%s error=%s",
                symbol,
                exc,
            )
            return SectorContext()

        industry = str(data.get("f127") or "").strip()
        return SectorContext(
            name=str(data.get("f58") or "").strip(),
            industry=industry,
            region_sector=str(data.get("f128") or "").strip(),
            data_source="EastMoneyRaw" if industry else "",
        )

    def _fetch_concepts(self, symbol: str) -> list[str]:
        prefixed_symbol = to_eastmoney_security_code(symbol)
        try:
            response = self._post(
                EASTMONEY_HOT_KEYWORD_URL,
                json={
                    "appId": "appId01",
                    "globalId": "786e4c21-70dc-435a-93bb-38",
                    "srcSecurityCode": prefixed_symbol,
                },
            )
            response.raise_for_status()
            rows = (response.json() or {}).get("data") or []
        except Exception as exc:
            logger.warning(
                "EastMoney raw sector concepts failed: symbol=%s prefixed=%s error=%s",
                symbol,
                prefixed_symbol,
                exc,
            )
            return []

        concepts: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            concept = str(row.get("conceptName") or "").strip()
            if concept and concept not in concepts:
                concepts.append(concept)
        return concepts

    def _get(self, url: str, params: dict[str, str]) -> httpx.Response:
        if self.client is not None:
            return self.client.get(url, params=params)
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            return client.get(url, params=params)

    def _post(self, url: str, json: dict[str, str]) -> httpx.Response:
        if self.client is not None:
            return self.client.post(url, json=json)
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            return client.post(url, json=json)


class SectorProvider:
    """Fetch sector context by source priority without deciding scores."""

    def __init__(
        self,
        akshare_provider: Optional[StockInfoProvider] = None,
        eastmoney_provider: Optional[StockInfoProvider] = None,
        astock_provider: Optional[StockInfoProvider] = None,
        eastmoney_raw_source: Optional[SectorContextProvider] = None,
    ) -> None:
        self.context_sources: list[tuple[str, Optional[SectorContextProvider]]] = [
            ("EastMoneyRaw", eastmoney_raw_source),
        ]
        self.stock_info_sources: list[tuple[str, Optional[StockInfoProvider]]] = [
            ("AkShare", akshare_provider),
            ("EastMoney", eastmoney_provider),
            ("A-Stock-Data", astock_provider),
        ]

    def get_sector_context(self, symbol: str) -> SectorContext:
        """Return merged industry/concept context from available sources."""
        contexts: list[SectorContext] = []
        missing_reasons: list[str] = []
        for source_name, provider in self.context_sources:
            if provider is None:
                continue
            try:
                context = provider.get_sector_context(symbol)
            except Exception as exc:
                logger.warning(
                    "SectorProvider %s failed: symbol=%s error=%s",
                    source_name,
                    symbol,
                    exc,
                )
                continue
            if context.available:
                contexts.append(context)
            else:
                missing_reasons.append(f"{source_name} unavailable")

        for source_name, provider in self.stock_info_sources:
            if provider is None:
                continue
            try:
                info = provider.get_stock_info(symbol)
            except Exception as exc:
                logger.warning(
                    "SectorProvider %s failed: symbol=%s error=%s",
                    source_name,
                    symbol,
                    exc,
                )
                continue

            context = _context_from_stock_info(info, source_name)
            if context.available:
                contexts.append(context)
                continue
            missing_reasons.append(
                f"{source_name} incomplete: industry={bool(context.industry)} "
                f"concepts={bool(context.concepts)}"
            )

        merged = _merge_contexts(*contexts)
        if merged.available:
            return merged
        if missing_reasons:
            logger.info(
                "SectorProvider incomplete sector data: symbol=%s details=%s",
                symbol,
                "; ".join(missing_reasons),
            )
        return SectorContext()


def _context_from_stock_info(info: StockInfo, source_name: str) -> SectorContext:
    industry = str(getattr(info, "industry", "") or "").strip()
    concepts = [
        str(item).strip()
        for item in (getattr(info, "concepts", None) or [])
        if str(item).strip()
    ]
    return SectorContext(
        name=str(getattr(info, "name", "") or "").strip(),
        industry=industry,
        concepts=concepts,
        data_source=source_name,
        warning=_sector_warning(bool(industry), bool(concepts)),
    )


def _merge_contexts(*contexts: SectorContext) -> SectorContext:
    name = ""
    industry = ""
    region_sector = ""
    concepts: list[str] = []
    sources: list[str] = []
    for context in contexts:
        if not context:
            continue
        if context.name and not name:
            name = context.name
        if context.industry and not industry:
            industry = context.industry
        if context.region_sector and not region_sector:
            region_sector = context.region_sector
        for concept in context.concepts:
            if concept and concept not in concepts:
                concepts.append(concept)
        if context.data_source:
            for item in context.data_source.split(","):
                source = item.strip()
                if source and source not in sources:
                    sources.append(source)

    industry_score = 10.0 if industry else None
    concept_score = 10.0 if concepts else None
    return SectorContext(
        name=name,
        industry=industry,
        region_sector=region_sector,
        concepts=concepts,
        data_source=", ".join(sources),
        industry_score=industry_score,
        concept_score=concept_score,
        warning=_sector_warning(bool(industry), bool(concepts)),
    )


def _sector_warning(industry_available: bool, concepts_available: bool) -> str:
    if industry_available and concepts_available:
        return ""
    if industry_available:
        return "概念数据暂不可用，板块评分部分纳入。"
    if concepts_available:
        return "行业数据暂不可用，板块评分部分纳入。"
    return "行业/概念数据暂不可用，板块评分暂未纳入。"


def to_eastmoney_security_code(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper()
    if normalized.startswith(("SZ", "SH")):
        return normalized
    if normalized.startswith(("6", "688")):
        return f"SH{normalized}"
    return f"SZ{normalized}"
