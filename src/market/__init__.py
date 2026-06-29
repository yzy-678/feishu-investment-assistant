"""市场数据服务。"""

from src.market.aggregator import (
    MarketDataAggregator,
    MarketDataSnapshot,
    get_market_data_aggregator,
)
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
from src.market.stock_resolver import (
    ResolvedStock,
    StockResolver,
    StockResolveResult,
    get_stock_resolver,
)
from src.market.observation_pool import (
    ObservationPoolManager,
    get_observation_pool_manager,
)
from src.market.stock_screener import (
    AkShareProvider,
    RealtimeQuote,
    StockCandidate,
    StrongStockScreener,
    get_stock_screener,
)
from src.market.strong_stock_analyzer import (
    StrongStockAnalyzer,
    StrongStockPick,
    get_strong_stock_analyzer,
)
from src.market.layered_observation import (
    AStockDataProvider,
    LayeredObservationBuilder,
    LayeredObservationReport,
    PotentialRelayPick,
    SectorObservation,
)

__all__ = [
    "AkShareError",
    "AkShareProvider",
    "AkShareSource",
    "AStockDataProvider",
    "DailyBar",
    "HistoryBar",
    "MACDSnapshot",
    "MASnapshot",
    "MAX_QUOTE_AGE_SECONDS",
    "MarketDataError",
    "MarketDataAggregator",
    "MarketDataSnapshot",
    "MarketDataService",
    "LayeredObservationBuilder",
    "LayeredObservationReport",
    "ObservationPoolManager",
    "PotentialRelayPick",
    "QuoteSnapshot",
    "RealtimeQuote",
    "ResolvedStock",
    "StockCandidate",
    "SectorObservation",
    "StrongStockAnalyzer",
    "StrongStockPick",
    "StockResolver",
    "StockResolveResult",
    "StrongStockScreener",
    "StockInfo",
    "get_market_data_service",
    "get_market_data_aggregator",
    "get_observation_pool_manager",
    "get_stock_resolver",
    "get_stock_screener",
    "get_strong_stock_analyzer",
]
