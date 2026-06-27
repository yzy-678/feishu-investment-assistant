"""市场数据服务。"""

from src.market.akshare_source import (
    AkShareError,
    AkShareSource,
    HistoryBar,
    MACDSnapshot,
    MASnapshot,
    StockInfo,
)
from src.market.service import (
    DailyBar,
    MAX_QUOTE_AGE_SECONDS,
    MarketDataError,
    MarketDataService,
    QuoteSnapshot,
    get_market_data_service,
)

__all__ = [
    "AkShareError",
    "AkShareSource",
    "DailyBar",
    "HistoryBar",
    "MACDSnapshot",
    "MASnapshot",
    "MAX_QUOTE_AGE_SECONDS",
    "MarketDataError",
    "MarketDataService",
    "QuoteSnapshot",
    "StockInfo",
    "get_market_data_service",
]
