"""Observation Pool for tracking strong stock persistence."""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Optional

from src.db import get_database
from src.db.models import ObservationPoolEntry, ObservationStatus
from src.market.strong_stock_analyzer import StrongStockPick
from src.time_utils import shanghai_today

logger = logging.getLogger(__name__)


class ObservationPoolManager:
    """Manage the daily strong stock observation pool."""

    _instance: Optional["ObservationPoolManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ObservationPoolManager":
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
        self._db.init_db()
        self._write_lock = threading.Lock()
        self._initialized = True
        logger.info("ObservationPoolManager initialized")

    def _get_connection(self) -> sqlite3.Connection:
        self._db = get_database()
        return self._db.get_connection()

    def update_daily_picks(
        self,
        picks: list[StrongStockPick],
    ) -> list[ObservationPoolEntry]:
        """Update observation_pool with today's Top3 strong stock picks."""
        today = shanghai_today().isoformat()
        today_symbols = {pick.symbol for pick in picks}

        with self._write_lock:
            conn = self._get_connection()
            previous_active = conn.execute(
                "SELECT * FROM observation_pool WHERE status IN ('active', 'watching')"
            ).fetchall()

            for pick in picks:
                existing = conn.execute(
                    "SELECT * FROM observation_pool WHERE symbol = ?",
                    (pick.symbol,),
                ).fetchone()
                if existing is None:
                    self._insert_pick(conn, pick, today)
                else:
                    self._update_existing_pick(conn, pick, existing, today)

            for row in previous_active:
                if row["symbol"] not in today_symbols:
                    conn.execute(
                        "UPDATE observation_pool SET status = 'dropped' WHERE symbol = ?",
                        (row["symbol"],),
                    )

            conn.commit()
            logger.info(
                "Observation pool updated: date=%s picks=%d",
                today,
                len(picks),
            )
            return self.get_active_pool()

    def get_active_pool(self) -> list[ObservationPoolEntry]:
        """Return currently active observation pool entries."""
        rows = self._get_connection().execute(
            """SELECT * FROM observation_pool
               WHERE status = 'active'
               ORDER BY consecutive_days DESC, latest_rank ASC, latest_score DESC"""
        ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def get_continuous_leaders(
        self,
        min_days: int = 2,
    ) -> list[ObservationPoolEntry]:
        """Return active entries that have appeared for at least min_days."""
        rows = self._get_connection().execute(
            """SELECT * FROM observation_pool
               WHERE status = 'active' AND consecutive_days >= ?
               ORDER BY consecutive_days DESC, latest_rank ASC""",
            (min_days,),
        ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def get_new_entries(self) -> list[ObservationPoolEntry]:
        """Return today's first-time Top3 entries."""
        today = shanghai_today().isoformat()
        rows = self._get_connection().execute(
            """SELECT * FROM observation_pool
               WHERE status = 'active'
                 AND first_seen = ?
                 AND last_seen = ?
               ORDER BY latest_rank ASC""",
            (today, today),
        ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def get_dropped_stocks(self) -> list[ObservationPoolEntry]:
        """Return stocks dropped from today's observation pool."""
        rows = self._get_connection().execute(
            """SELECT * FROM observation_pool
               WHERE status = 'dropped'
               ORDER BY last_seen DESC, consecutive_days DESC"""
        ).fetchall()
        return [self._row_to_model(row) for row in rows]

    def clear(self) -> int:
        """Clear the observation pool. Intended for tests and maintenance."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM observation_pool")
        conn.commit()
        return cursor.rowcount

    @staticmethod
    def _insert_pick(
        conn: sqlite3.Connection,
        pick: StrongStockPick,
        today: str,
    ) -> None:
        conn.execute(
            """INSERT INTO observation_pool
               (symbol, name, industry, first_seen, last_seen, consecutive_days,
                highest_score, latest_score, latest_rank, latest_reason, status)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 'active')""",
            (
                pick.symbol,
                pick.name,
                pick.industry,
                today,
                today,
                pick.score,
                pick.score,
                pick.rank,
                pick.reason,
            ),
        )

    @staticmethod
    def _update_existing_pick(
        conn: sqlite3.Connection,
        pick: StrongStockPick,
        row: sqlite3.Row,
        today: str,
    ) -> None:
        previous_last_seen = str(row["last_seen"])
        if previous_last_seen == today:
            consecutive_days = int(row["consecutive_days"] or 0)
        else:
            consecutive_days = int(row["consecutive_days"] or 0) + 1

        highest_score = max(float(row["highest_score"] or 0.0), pick.score)
        conn.execute(
            """UPDATE observation_pool
               SET name = ?,
                   industry = ?,
                   last_seen = ?,
                   consecutive_days = ?,
                   highest_score = ?,
                   latest_score = ?,
                   latest_rank = ?,
                   latest_reason = ?,
                   status = 'active'
               WHERE symbol = ?""",
            (
                pick.name,
                pick.industry,
                today,
                consecutive_days,
                highest_score,
                pick.score,
                pick.rank,
                pick.reason,
                pick.symbol,
            ),
        )

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> ObservationPoolEntry:
        return ObservationPoolEntry(
            symbol=row["symbol"],
            name=row["name"],
            industry=row["industry"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            consecutive_days=row["consecutive_days"],
            highest_score=row["highest_score"],
            latest_score=row["latest_score"],
            latest_rank=row["latest_rank"],
            latest_reason=row["latest_reason"],
            status=ObservationStatus(row["status"]),
        )


_observation_pool_manager: Optional[ObservationPoolManager] = None


def get_observation_pool_manager() -> ObservationPoolManager:
    """Return the ObservationPoolManager singleton."""
    global _observation_pool_manager  # noqa: PLW0603
    if _observation_pool_manager is None:
        _observation_pool_manager = ObservationPoolManager()
    return _observation_pool_manager
