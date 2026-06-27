"""
运行时配置管理器

通过 SQLite app_config 表持久化，每次读写直接操作数据库，
不依赖内存状态，确保多进程/线程环境下的配置一致性。
"""

import logging
import threading
from datetime import datetime
from typing import Optional

from src.db.models import AppConfig

logger = logging.getLogger(__name__)

# ── 默认配置 ──────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, str] = {
    "enabled": "true",
    "market": "CN",
    "scan_interval": "1800",  # 秒，即 30 分钟
}


class ConfigError(Exception):
    """配置操作异常"""


class ConfigManager:
    """运行时配置管理器（SQLite 持久化）

    所有读写直接操作 app_config 表，不缓存，保证一致性。

    用法::

        cfg = ConfigManager()
        cfg.get_enabled()          # → bool
        cfg.set_market("HK")       # None
        cfg.get_all()              # → dict
    """

    _instance: Optional["ConfigManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ConfigManager":
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
        # 延迟导入避免循环依赖（config ↔ db）
        from src.db import get_database
        self._db = get_database()
        self._init_defaults()
        self._initialized = True
        logger.info("ConfigManager initialized with SQLite backend")

    # ── 内部方法 ───────────────────────────────────────────

    def _init_defaults(self) -> None:
        """写入默认配置（仅当键不存在时）"""
        for key, value in DEFAULT_CONFIG.items():
            existing = self._get_raw(key)
            if existing is None:
                self._set_raw(key, value)
                logger.debug("Initialized default config: %s = %s", key, value)

    def _get_raw(self, key: str) -> Optional[str]:
        """直接查询 app_config 表"""
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT value FROM app_config WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None
        except Exception as exc:
            logger.error("Failed to read config key '%s': %s", key, exc)
            raise ConfigError(f"读取配置失败: {key}") from exc

    def _set_raw(self, key: str, value: str) -> None:
        """写入 app_config 表（UPSERT 语义）"""
        try:
            conn = self._db.get_connection()
            conn.execute(
                """INSERT INTO app_config (key, value, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value = excluded.value,
                       updated_at = excluded.updated_at""",
                (key, value, datetime.now()),
            )
            conn.commit()
        except Exception as exc:
            logger.error("Failed to write config key '%s': %s", key, exc)
            raise ConfigError(f"写入配置失败: {key}") from exc

    # ── 通用接口 ───────────────────────────────────────────

    def get_value(self, key: str) -> Optional[str]:
        """获取原始配置值（字符串）

        Args:
            key: 配置键名

        Returns:
            配置值的字符串形式；键不存在时返回 None
        """
        return self._get_raw(key)

    def set_value(self, key: str, value: str) -> None:
        """设置配置值

        Args:
            key: 配置键名
            value: 配置值（字符串）
        """
        self._set_raw(key, value)

    def get_all(self) -> dict[str, str]:
        """获取全部配置键值对"""
        try:
            conn = self._db.get_connection()
            rows = conn.execute(
                "SELECT key, value FROM app_config ORDER BY key"
            ).fetchall()
            return {row["key"]: row["value"] for row in rows}
        except Exception as exc:
            logger.error("Failed to list all config: %s", exc)
            raise ConfigError("获取全部配置失败") from exc

    # ── 类型安全接口 ───────────────────────────────────────

    def get_enabled(self) -> bool:
        """系统是否已启用"""
        raw = self._get_raw("enabled")
        if raw is None:
            return True
        return raw.strip().lower() == "true"

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用系统"""
        self._set_raw("enabled", "true" if enabled else "false")
        logger.info("System %s", "enabled" if enabled else "disabled")

    def get_market(self) -> str:
        """获取当前市场（CN / HK / US）"""
        raw = self._get_raw("market")
        return raw if raw else "CN"

    def set_market(self, market: str) -> None:
        """切换市场

        Args:
            market: "CN" / "HK" / "US"
        """
        normalized = market.strip().upper()
        if normalized not in ("CN", "HK", "US"):
            raise ValueError(f"不支持的市场: {market}，可选: CN, HK, US")
        self._set_raw("market", normalized)
        logger.info("Market switched to %s", normalized)

    def get_scan_interval(self) -> int:
        """获取盘中扫描间隔（秒）"""
        raw = self._get_raw("scan_interval")
        if raw is None:
            return 1800
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid scan_interval value '%s', fallback to 1800", raw)
            return 1800

    def set_scan_interval(self, interval_seconds: int) -> None:
        """设置扫描间隔

        Args:
            interval_seconds: 间隔秒数（>= 60）
        """
        if interval_seconds < 60:
            raise ValueError(f"扫描间隔不能小于 60 秒: {interval_seconds}")
        self._set_raw("scan_interval", str(interval_seconds))
        logger.info("Scan interval set to %d seconds", interval_seconds)

    def as_app_config(self, key: str) -> Optional[AppConfig]:
        """以 Pydantic 模型形式返回单条配置"""
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT key, value, updated_at FROM app_config WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return AppConfig(
                key=row["key"],
                value=row["value"],
                updated_at=row["updated_at"],
            )
        except Exception as exc:
            logger.error("Failed to read AppConfig model for '%s': %s", key, exc)
            return None


# ── 全局单例访问函数 ─────────────────────────────────────

_config_instance: Optional[ConfigManager] = None
_config_lock: threading.Lock = threading.Lock()


def get_config() -> ConfigManager:
    """获取 ConfigManager 单例"""
    global _config_instance  # noqa: PLW0603
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = ConfigManager()
    return _config_instance
