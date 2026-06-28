"""
SQLite 数据库管理模块

提供 DatabaseManager 单例，管理所有数据库连接和表结构。
支持 FastAPI 生命周期集成（startup / shutdown）。
每个线程独立管理自己的连接，使用 WAL 模式提升并发性能。
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from src.config.settings import settings


class DatabaseManager:
    """SQLite 数据库管理器（线程安全单例）

    用法::

        db = DatabaseManager()
        conn = db.get_connection()
        cursor = conn.execute("SELECT * FROM app_config")
    """

    _instance: Optional["DatabaseManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "DatabaseManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized: bool = False  # type: ignore[assignment]
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """初始化数据库路径和线程本地存储"""
        if getattr(self, "_initialized", False):
            return

        # 解析数据库文件绝对路径
        self.db_path: Path = Path(settings.database_path).resolve()
        # 确保父目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local: threading.local = threading.local()
        self._initialized = True

    # ── 连接管理 ──────────────────────────────────────────────

    def get_connection(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接

        每个线程持有独立的连接，避免 SQLite 线程安全问题。
        连接启用 WAL 模式 + 外键约束。
        """
        conn: Optional[sqlite3.Connection] = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.connection = conn
        return conn

    def close_connection(self) -> None:
        """关闭当前线程的数据库连接"""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    # ── 初始化 ────────────────────────────────────────────────

    def init_db(self) -> None:
        """初始化数据库：创建所有基础表

        幂等操作，可安全重复调用。
        """
        conn = self.get_connection()
        self._create_tables(conn)

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """执行建表 DDL"""
        conn.executescript("""
            -- app_config：运行时配置
            CREATE TABLE IF NOT EXISTS app_config (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- watchlist：自选股
            CREATE TABLE IF NOT EXISTS watchlist (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT NOT NULL,
                name        TEXT NOT NULL,
                market      TEXT NOT NULL DEFAULT 'a',
                tags        TEXT DEFAULT '',
                notes       TEXT DEFAULT '',
                added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, market)
            );

            -- conversations：短期对话记录
            CREATE TABLE IF NOT EXISTS conversations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                metadata        TEXT DEFAULT '{}',
                conversation_id TEXT DEFAULT '',
                is_summarized   INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_conversations_session
                ON conversations(session_id, created_at DESC);

            -- conversation_summaries：对话摘要
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                summary       TEXT NOT NULL,
                summary_type  TEXT DEFAULT 'conversation',
                token_count   INTEGER DEFAULT 0,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_session
                ON conversation_summaries(session_id, created_at DESC);

            -- alert_events：预警事件存储
            CREATE TABLE IF NOT EXISTS alert_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id        TEXT NOT NULL,
                alert_type      TEXT NOT NULL,
                title           TEXT NOT NULL,
                content         TEXT NOT NULL,
                severity        TEXT DEFAULT 'info',
                related_code    TEXT DEFAULT '',
                related_sector  TEXT DEFAULT '',
                strength        REAL DEFAULT 0.0,
                peak_strength   REAL DEFAULT 0.0,
                first_seen      TIMESTAMP,
                last_sent       TIMESTAMP,
                sent_count      INTEGER DEFAULT 0,
                resolved        INTEGER DEFAULT 0,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_alert_events_event_id
                ON alert_events(event_id);
            CREATE INDEX IF NOT EXISTS idx_alert_events_unresolved
                ON alert_events(resolved, alert_type);

            -- observation_pool：强势观察池连续跟踪
            CREATE TABLE IF NOT EXISTS observation_pool (
                symbol            TEXT PRIMARY KEY,
                name              TEXT NOT NULL,
                industry          TEXT DEFAULT '',
                first_seen        TEXT NOT NULL,
                last_seen         TEXT NOT NULL,
                consecutive_days  INTEGER DEFAULT 1,
                highest_score     REAL DEFAULT 0.0,
                latest_score      REAL DEFAULT 0.0,
                latest_rank       INTEGER DEFAULT 0,
                latest_reason     TEXT DEFAULT '',
                status            TEXT DEFAULT 'active'
                                  CHECK(status IN ('active', 'dropped', 'watching'))
            );
            CREATE INDEX IF NOT EXISTS idx_observation_pool_status
                ON observation_pool(status, last_seen DESC);

            -- investment_rating_history：投资评级每日快照
            CREATE TABLE IF NOT EXISTS investment_rating_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT NOT NULL,
                rating_date     TEXT NOT NULL,
                name            TEXT DEFAULT '',
                total_score     REAL DEFAULT 0.0,
                rating_level    TEXT DEFAULT 'D',
                trend_score     REAL DEFAULT 0.0,
                volume_score    REAL DEFAULT 0.0,
                sector_score    REAL DEFAULT 0.0,
                breakout_score  REAL DEFAULT 0.0,
                strength_score  REAL DEFAULT 0.0,
                summary         TEXT DEFAULT '',
                warning         TEXT DEFAULT '',
                data_source     TEXT DEFAULT '',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, rating_date)
            );
            CREATE INDEX IF NOT EXISTS idx_investment_rating_history_symbol_date
                ON investment_rating_history(symbol, rating_date DESC);
        """)
        conn.commit()

    def close_all(self) -> None:
        """关闭所有连接（应用退出时调用）"""
        self.close_connection()


# ── 模块级便捷函数 ────────────────────────────────────────────

_db_instance: Optional[DatabaseManager] = None
_db_lock: threading.Lock = threading.Lock()


def get_database() -> DatabaseManager:
    """获取 DatabaseManager 单例"""
    global _db_instance  # noqa: PLW0603
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                _db_instance = DatabaseManager()
    return _db_instance


def init_database() -> None:
    """初始化数据库（供 FastAPI startup 事件调用）"""
    db = get_database()
    db.init_db()


def close_database() -> None:
    """关闭数据库（供 FastAPI shutdown 事件调用）"""
    db = get_database()
    db.close_all()
