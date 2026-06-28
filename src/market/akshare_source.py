"""
AkShare 数据源封装。

EastMoney 继续负责实时行情；本模块只负责历史数据、技术指标和基础信息。
所有对外返回值统一为 Pydantic Model，不向业务层暴露 DataFrame。
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

EASTMONEY_QUOTE_URL = "https://push2.eastmoney.com/api/qt/stock/get"


class AkShareError(Exception):
    """AkShare 数据获取或解析失败。"""


class HistoryBar(BaseModel):
    """历史 K 线。"""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


class MASnapshot(BaseModel):
    """均线快照。"""

    symbol: str
    MA5: Optional[float] = None
    MA10: Optional[float] = None
    MA20: Optional[float] = None
    MA60: Optional[float] = None


class MACDSnapshot(BaseModel):
    """MACD 快照。"""

    symbol: str
    DIF: Optional[float] = None
    DEA: Optional[float] = None
    MACD: Optional[float] = None


class StockInfo(BaseModel):
    """股票基础信息。"""

    symbol: str
    name: str = ""
    industry: str = ""
    concepts: list[str] = Field(default_factory=list)


class AkShareSource:
    """AkShare 统一访问层。"""

    def __init__(self, ak_module: Any = None) -> None:
        self._ak = ak_module

    def get_history(self, symbol: str, period: int = 60) -> list[HistoryBar]:
        """获取最近 period 条日 K 线。"""
        self._log_request("stock_zh_a_hist", symbol, period=period)
        try:
            frame = self._akshare().stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                adjust="qfq",
            )
            records = self._records(frame, tail=period)
            bars = [
                self._parse_history_row(row)
                for row in records
                if self._has_history_fields(row)
            ]
            self._log_success("stock_zh_a_hist", symbol, rows=len(bars))
            return bars
        except Exception as exc:
            self._log_failed("stock_zh_a_hist", symbol, exc)
            raise AkShareError(f"AkShare 历史 K 线获取失败: {exc}") from exc

    def get_ma(self, symbol: str) -> MASnapshot:
        """计算 MA5 / MA10 / MA20 / MA60。"""
        self._log_request("ma", symbol)
        try:
            bars = self.get_history(symbol, period=60)
            closes = [bar.close for bar in bars]
            snapshot = MASnapshot(
                symbol=symbol,
                MA5=self._moving_average(closes, 5),
                MA10=self._moving_average(closes, 10),
                MA20=self._moving_average(closes, 20),
                MA60=self._moving_average(closes, 60),
            )
            self._log_success("ma", symbol)
            return snapshot
        except Exception as exc:
            self._log_failed("ma", symbol, exc)
            if isinstance(exc, AkShareError):
                raise
            raise AkShareError(f"AkShare 均线计算失败: {exc}") from exc

    def get_macd(self, symbol: str) -> MACDSnapshot:
        """计算 MACD，返回最新一条 DIF / DEA / MACD。"""
        self._log_request("macd", symbol)
        try:
            bars = self.get_history(symbol, period=60)
            closes = [bar.close for bar in bars]
            snapshot = self._calculate_macd(symbol, closes)
            self._log_success("macd", symbol)
            return snapshot
        except Exception as exc:
            self._log_failed("macd", symbol, exc)
            if isinstance(exc, AkShareError):
                raise
            raise AkShareError(f"AkShare MACD 计算失败: {exc}") from exc

    def get_stock_info(self, symbol: str) -> StockInfo:
        """获取股票名称、行业、概念板块。"""
        self._log_request("stock_individual_info_em", symbol)
        try:
            try:
                frame = self._akshare().stock_individual_info_em(symbol=symbol)
                info = self._key_value_map(frame)
            except Exception as exc:
                self._log_failed("stock_individual_info_em", symbol, exc)
                self._log_request("stock_individual_info_em_fallback", symbol)
                info = self._fetch_stock_info_fallback(symbol)

            concepts = self._parse_concepts(info)
            if not concepts:
                concepts = self._fetch_hot_concepts(symbol)

            stock_info = StockInfo(
                symbol=symbol,
                name=self._first_value(
                    info,
                    ("股票简称", "股票名称", "证券简称", "名称"),
                ),
                industry=self._first_value(
                    info,
                    ("行业", "所属行业", "行业板块"),
                ),
                concepts=concepts,
            )
            self._log_success("stock_individual_info_em", symbol)
            return stock_info
        except Exception as exc:
            self._log_failed("stock_individual_info_em", symbol, exc)
            raise AkShareError(f"AkShare 股票基础信息获取失败: {exc}") from exc

    def _akshare(self) -> Any:
        if self._ak is None:
            try:
                self._ak = importlib.import_module("akshare")
            except ImportError as exc:
                raise AkShareError("未安装 akshare，请先安装依赖") from exc
        return self._ak

    def _fetch_hot_concepts(self, symbol: str) -> list[str]:
        ak = self._akshare()
        if not hasattr(ak, "stock_hot_keyword_em"):
            return []

        self._log_request("stock_hot_keyword_em", symbol)
        try:
            frame = ak.stock_hot_keyword_em(symbol=symbol)
            records = self._records(frame)
            concepts: list[str] = []
            for row in records:
                value = self._row_first_value(
                    row,
                    ("概念名称", "关键词", "概念", "板块名称", "名称"),
                )
                if value and value not in concepts:
                    concepts.append(value)
            self._log_success("stock_hot_keyword_em", symbol, rows=len(concepts))
            return concepts
        except Exception as exc:
            self._log_failed("stock_hot_keyword_em", symbol, exc)
            return []

    def _fetch_stock_info_fallback(self, symbol: str) -> dict[str, str]:
        market_code = "1" if str(symbol).startswith("6") else "0"
        response = httpx.get(
            EASTMONEY_QUOTE_URL,
            params={
                "fltt": "2",
                "invt": "2",
                "fields": "f57,f58,f127",
                "secid": f"{market_code}.{symbol}",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        data = (response.json() or {}).get("data") or {}
        return {
            "股票代码": str(data.get("f57") or "").strip(),
            "股票简称": str(data.get("f58") or "").strip(),
            "行业": str(data.get("f127") or "").strip(),
        }

    @staticmethod
    def _records(frame: Any, tail: Optional[int] = None) -> list[dict[str, Any]]:
        if frame is None:
            return []
        if isinstance(frame, list):
            records = frame
        elif isinstance(frame, dict):
            records = [frame]
        else:
            if getattr(frame, "empty", False):
                return []
            if tail is not None and hasattr(frame, "tail"):
                frame = frame.tail(tail)
            if hasattr(frame, "to_dict"):
                records = frame.to_dict("records")
            else:
                records = []

        return [
            dict(row)
            for row in records
            if isinstance(row, dict)
        ]

    @classmethod
    def _key_value_map(cls, frame: Any) -> dict[str, str]:
        info: dict[str, str] = {}
        for row in cls._records(frame):
            key = cls._row_first_value(row, ("item", "项目", "指标", "name"))
            value = cls._row_first_value(row, ("value", "值", "内容", "data"))
            if key:
                info[key] = value
        return info

    @classmethod
    def _parse_history_row(cls, row: dict[str, Any]) -> HistoryBar:
        return HistoryBar(
            date=str(cls._row_first_value(row, ("日期", "date", "交易日期"))),
            open=cls._to_float(cls._row_first_value(row, ("开盘", "open"))),
            high=cls._to_float(cls._row_first_value(row, ("最高", "high"))),
            low=cls._to_float(cls._row_first_value(row, ("最低", "low"))),
            close=cls._to_float(cls._row_first_value(row, ("收盘", "close"))),
            volume=cls._to_float(cls._row_first_value(row, ("成交量", "volume"))),
            amount=cls._to_float(cls._row_first_value(row, ("成交额", "amount"))),
        )

    @classmethod
    def _has_history_fields(cls, row: dict[str, Any]) -> bool:
        return bool(cls._row_first_value(row, ("日期", "date", "交易日期")))

    @staticmethod
    def _row_first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            if key in row and row[key] not in (None, ""):
                return str(row[key]).strip()
        return ""

    @staticmethod
    def _first_value(info: dict[str, str], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = info.get(key)
            if value:
                return str(value).strip()
        return ""

    @staticmethod
    def _parse_concepts(info: dict[str, str]) -> list[str]:
        raw = AkShareSource._first_value(
            info,
            ("概念板块", "所属概念", "所属概念板块", "题材概念"),
        )
        if not raw:
            return []
        normalized = raw.replace("，", ",").replace("、", ",").replace(";", ",")
        return [
            item.strip()
            for item in normalized.split(",")
            if item.strip()
        ]

    @staticmethod
    def _moving_average(closes: list[float], window: int) -> Optional[float]:
        if len(closes) < window:
            return None
        return round(sum(closes[-window:]) / window, 4)

    @staticmethod
    def _calculate_macd(symbol: str, closes: list[float]) -> MACDSnapshot:
        if not closes:
            return MACDSnapshot(symbol=symbol)

        ema12 = closes[0]
        ema26 = closes[0]
        dea = 0.0
        dif = 0.0
        for close in closes:
            ema12 = ema12 * (11 / 13) + close * (2 / 13)
            ema26 = ema26 * (25 / 27) + close * (2 / 27)
            dif = ema12 - ema26
            dea = dea * (8 / 10) + dif * (2 / 10)

        macd = (dif - dea) * 2
        return MACDSnapshot(
            symbol=symbol,
            DIF=round(dif, 4),
            DEA=round(dea, 4),
            MACD=round(macd, 4),
        )

    @staticmethod
    def _to_float(raw: Any) -> float:
        if raw in (None, "", "-", "--"):
            return 0.0
        return float(str(raw).replace(",", ""))

    @staticmethod
    def _log_request(function: str, symbol: str, **extra: Any) -> None:
        logger.info(
            "AkShare request: function=%s symbol=%s extra=%s",
            function,
            symbol,
            extra,
        )

    @staticmethod
    def _log_success(function: str, symbol: str, **extra: Any) -> None:
        logger.info(
            "AkShare success: function=%s symbol=%s extra=%s",
            function,
            symbol,
            extra,
        )

    @staticmethod
    def _log_failed(function: str, symbol: str, exc: Exception) -> None:
        logger.warning(
            "AkShare failed: function=%s symbol=%s error=%s",
            function,
            symbol,
            exc,
        )
