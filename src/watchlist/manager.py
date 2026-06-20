"""
自选股管理器

通过 SQLite watchlist 表持久化，提供完整的自选股 CRUD 操作。
支持按 symbol、market、tag 进行查询。
"""

import logging
import threading
from datetime import datetime
from typing import Optional

from src.db import get_database
from src.db.models import WatchlistItem

logger = logging.getLogger(__name__)


class WatchlistError(Exception):
    """自选股操作异常"""


class WatchlistManager:
    """自选股管理器（线程安全单例）

    所有读写直接操作 SQLite watchlist 表，不缓存。

    用法::

        wm = WatchlistManager()
        item = wm.add_stock("000001", "平安银行", "CN", tags=["银行", "蓝筹"])
        wm.list_stocks()
    """

    _instance: Optional["WatchlistManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "WatchlistManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized: bool = False  # type: ignore[assignment]
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._db = get_database()
        self._ensure_schema()
        self._initialized = True
        logger.info("WatchlistManager initialized")

    # ── Schema 迁移 ───────────────────────────────────────

    def _ensure_schema(self) -> None:
        """确保 watchlist 表包含 updated_at 列"""
        try:
            conn = self._db.get_connection()
            conn.execute("ALTER TABLE watchlist ADD COLUMN updated_at TIMESTAMP")
            conn.commit()
            logger.info("Added 'updated_at' column to watchlist table")
        except Exception:
            pass  # 列已存在，忽略

    # ── CRUD ──────────────────────────────────────────────

    def add_stock(
        self,
        symbol: str,
        name: str,
        market: str = "CN",
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> WatchlistItem:
        """添加自选股

        Args:
            symbol: 股票代码（不区分大小写，自动转为大写）
            name: 股票名称
            market: 市场（CN / HK / US）
            tags: 标签列表
            notes: 用户备注

        Returns:
            新增的 WatchlistItem

        Raises:
            WatchlistError: 市场无效或股票已存在时抛出
        """
        market = market.upper()
        if market not in ("CN", "HK", "US"):
            raise WatchlistError(
                f"不支持的市场: {market}，可选: CN, HK, US"
        )
        # CN/HK/US → a/hk/us（enum 值）
        _MARKET_MAP = {"CN": "a", "HK": "hk", "US": "us"}
        market_enum = _MARKET_MAP[market]

        symbol = symbol.upper()
        existing = self.get_stock(symbol)
        if existing is not None:
            raise WatchlistError(
                f"自选股已存在: {symbol} ({existing.name})"
            )

        tags_str = ",".join(tags) if tags else ""
        notes_str = notes or ""
        now = datetime.now()

        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                """INSERT INTO watchlist (symbol, name, market, tags, notes, added_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, name, market_enum, tags_str, notes_str, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM watchlist WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            logger.info("Added stock to watchlist: %s (%s)", symbol, name)
            return self._row_to_model(row)
        except Exception as exc:
            logger.error("Failed to add stock '%s': %s", symbol, exc)
            raise WatchlistError(f"添加自选股失败: {symbol}") from exc

    def remove_stock(self, symbol: str) -> bool:
        """删除自选股

        Returns:
            是否成功删除（False 表示该股票不在自选列表中）
        """
        symbol = symbol.upper()
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "DELETE FROM watchlist WHERE symbol = ?", (symbol,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info("Removed stock from watchlist: %s", symbol)
            return deleted
        except Exception as exc:
            logger.error("Failed to remove stock '%s': %s", symbol, exc)
            raise WatchlistError(f"删除自选股失败: {symbol}") from exc

    def get_stock(self, symbol: str) -> Optional[WatchlistItem]:
        """查询单只自选股详情

        Returns:
            WatchlistItem 或 None（不存在时）
        """
        symbol = symbol.upper()
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT * FROM watchlist WHERE symbol = ?", (symbol,)
            ).fetchone()
            return self._row_to_model(row) if row else None
        except Exception as exc:
            logger.error("Failed to get stock '%s': %s", symbol, exc)
            raise WatchlistError(f"查询自选股失败: {symbol}") from exc

    def list_stocks(self, market: Optional[str] = None) -> list[WatchlistItem]:
        """列出所有自选股

        Args:
            market: 可选，按市场过滤（CN / HK / US）

        Returns:
            WatchlistItem 列表（按添加时间倒序）
        """
        try:
            conn = self._db.get_connection()
            if market is not None:
                market_key = market.upper()
                if market_key not in ("CN", "HK", "US"):
                    raise WatchlistError(f"不支持的市场: {market_key}")
                _MAP = {"CN": "a", "HK": "hk", "US": "us"}
                db_market = _MAP[market_key]
                rows = conn.execute(
                    "SELECT * FROM watchlist WHERE market = ? ORDER BY added_at DESC",
                    (db_market,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM watchlist ORDER BY added_at DESC"
                ).fetchall()
            return [self._row_to_model(r) for r in rows]
        except Exception as exc:
            logger.error("Failed to list watchlist: %s", exc)
            raise WatchlistError("查询自选股列表失败") from exc

    def update_tags(self, symbol: str, tags: list[str]) -> WatchlistItem:
        """更新自选股标签

        Args:
            symbol: 股票代码
            tags: 新标签列表

        Returns:
            更新后的 WatchlistItem
        """
        return self._update_field(symbol, "tags", ",".join(tags))

    def update_notes(self, symbol: str, notes: str) -> WatchlistItem:
        """更新自选股备注

        Args:
            symbol: 股票代码
            notes: 新备注内容

        Returns:
            更新后的 WatchlistItem
        """
        return self._update_field(symbol, "notes", notes)

    def _update_field(self, symbol: str, field: str, value: str) -> WatchlistItem:
        """通用字段更新方法"""
        symbol = symbol.upper()
        existing = self.get_stock(symbol)
        if existing is None:
            raise WatchlistError(f"自选股不存在: {symbol}")

        try:
            conn = self._db.get_connection()
            now = datetime.now()
            conn.execute(
                f"UPDATE watchlist SET {field} = ?, updated_at = ? WHERE symbol = ?",
                (value, now, symbol),
            )
            conn.commit()
            logger.info("Updated '%s' for stock '%s'", field, symbol)
            return self.get_stock(symbol)  # type: ignore[return-value]
        except Exception as exc:
            logger.error("Failed to update %s for '%s': %s", field, symbol, exc)
            raise WatchlistError(f"更新{field}失败: {symbol}") from exc

    # ── 查询 ──────────────────────────────────────────────

    def search_by_tag(self, tag: str) -> list[WatchlistItem]:
        """按标签搜索自选股

        Args:
            tag: 标签名称（模糊匹配）

        Returns:
            匹配的自选股列表（按添加时间倒序）
        """
        try:
            conn = self._db.get_connection()
            rows = conn.execute(
                "SELECT * FROM watchlist WHERE tags LIKE ? ORDER BY added_at DESC",
                (f"%{tag}%",),
            ).fetchall()
            return [self._row_to_model(r) for r in rows]
        except Exception as exc:
            logger.error("Failed to search by tag '%s': %s", tag, exc)
            raise WatchlistError(f"按标签搜索失败: {tag}") from exc

    def count(self) -> int:
        """统计自选股数量"""
        try:
            conn = self._db.get_connection()
            row = conn.execute("SELECT COUNT(*) as cnt FROM watchlist").fetchone()
            return row["cnt"]
        except Exception as exc:
            logger.error("Failed to count watchlist: %s", exc)
            raise WatchlistError("统计自选股数量失败") from exc

    def clear(self) -> int:
        """清空所有自选股（谨慎操作）

        Returns:
            被删除的记录数
        """
        try:
            conn = self._db.get_connection()
            cursor = conn.execute("DELETE FROM watchlist")
            conn.commit()
            count = cursor.rowcount
            logger.warning("Cleared all %d watchlist items", count)
            return count
        except Exception as exc:
            logger.error("Failed to clear watchlist: %s", exc)
            raise WatchlistError("清空自选股失败") from exc

    # ── 行转换 ────────────────────────────────────────────

    @staticmethod
    def _row_to_model(row) -> WatchlistItem:
        """将 SQLite Row 转换为 Pydantic 模型"""
        return WatchlistItem(
            id=row["id"],
            symbol=row["symbol"],
            name=row["name"],
            market=row["market"],
            tags=row["tags"],
            notes=row["notes"],
            added_at=row["added_at"],
            updated_at=row["updated_at"] if "updated_at" in row.keys() else None,
        )


# ── 全局单例访问函数 ─────────────────────────────────────

_watchlist_instance: Optional[WatchlistManager] = None
_watchlist_lock: threading.Lock = threading.Lock()


def get_watchlist() -> WatchlistManager:
    """获取 WatchlistManager 单例"""
    global _watchlist_instance  # noqa: PLW0603
    if _watchlist_instance is None:
        with _watchlist_lock:
            if _watchlist_instance is None:
                _watchlist_instance = WatchlistManager()
    return _watchlist_instance
