"""
预警事件 Agent

负责盘中预警的检测、记录、去重和推送。
使用 alert_events 表持久化，支持 30 分钟冷却和强度升级。
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Optional

from src.agents.base import BaseAgent, AgentType, AgentResponse
from src.config.manager import get_config
from src.db import get_database
from src.db.models import AlertEvent, AlertSeverity
from src.watchlist.manager import get_watchlist

logger = logging.getLogger(__name__)

# ── 去重策略常量 ─────────────────────────────────────────

COOLDOWN_MINUTES: int = 30
"""同事件冷却时间（分钟）"""

STRENGTH_ESCALATION_RATIO: float = 1.3
"""强度超过峰值的 130% 时允许突破冷却"""

_HANDLE_KEYWORDS: list[str] = [
    "预警", "异动", "监控", "提醒", "扫描",
    "警报", "告警",
]


class AlertAgent(BaseAgent):
    """预警事件 Agent

    职责：
    1. 记录新预警事件
    2. 基于冷却窗口和强度审判定推送策略
    3. 管理事件生命周期（活跃→解除）
    """

    _instance: Optional["AlertAgent"] = None

    def __new__(cls) -> "AlertAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized: bool = False  # type: ignore[assignment]
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._db = get_database()
        self.config = get_config()
        self.watchlist = get_watchlist()
        self._lock = threading.Lock()
        self._initialized = True
        logger.info("AlertAgent initialized (cooldown=%dmin, escalation=%.0f%%)",
                     COOLDOWN_MINUTES, STRENGTH_ESCALATION_RATIO * 100)

    # ── BaseAgent 接口 ────────────────────────────────────

    @property
    def agent_type(self) -> AgentType:
        return AgentType.ALERT

    def can_handle(self, message: str) -> bool:
        if not message or not message.strip():
            return False
        return any(kw in message for kw in _HANDLE_KEYWORDS)

    def handle(self, session_id: str, message: str) -> AgentResponse:
        """处理用户对预警状态的查询"""
        try:
            active = self.get_active_events()
            if not active:
                return AgentResponse(
                    success=True,
                    agent=AgentType.ALERT,
                    message="当前无活跃预警事件，一切正常。",
                    metadata={"active_count": 0},
                )

            lines: list[str] = [
                f"共 {len(active)} 个活跃预警：",
            ]
            for e in active:
                sent_info = (
                    f" | 上次推送: {e.last_sent.strftime('%H:%M')}"
                    if e.last_sent else ""
                )
                lines.append(
                    f"  [{e.alert_type.value}] {e.title}（强度 {e.strength:.1f}）{sent_info}"
                )

            return AgentResponse(
                success=True,
                agent=AgentType.ALERT,
                message="\n".join(lines),
                metadata={"active_count": len(active)},
            )
        except Exception as exc:
            logger.error("AlertAgent handle error: %s", exc)
            return AgentResponse(
                success=False,
                agent=AgentType.ALERT,
                message="查询预警信息时出现错误，请稍后再试。",
                metadata={"error": str(exc)},
            )

    # ── 事件管理 ─────────────────────────────────────────

    def record_event(
        self,
        event_id: str,
        alert_type: str,
        title: str,
        content: str,
        strength: float,
        severity: str = "info",
        related_code: str = "",
        related_sector: str = "",
    ) -> AlertEvent:
        """记录预警事件（新建或更新强度）

        1. 查询 event_id 是否存在
        2. 不存在 → 插入新记录，设置 first_seen
        3. 存在 → 更新 strength / peak_strength / severity

        Args:
            event_id: 事件唯一标识（如 "price_spike:000001"）
            alert_type: 事件类型字符串
            title: 预警标题
            content: 预警详情
            strength: 当前强度值
            severity: 级别（info / warning / critical）
            related_code: 关联股票代码
            related_sector: 关联板块

        Returns:
            更新后的 AlertEvent
        """
        with self._lock:
            existing = self._find_event(event_id)
            now = datetime.now()

            try:
                conn = self._db.get_connection()

                if existing is None:
                    # 新建事件
                    cursor = conn.execute(
                        """INSERT INTO alert_events
                           (event_id, alert_type, title, content, severity,
                            related_code, related_sector, strength, peak_strength,
                            first_seen, last_sent, sent_count, resolved, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0, ?)""",
                        (event_id, alert_type, title, content, severity,
                         related_code, related_sector, strength, strength,
                         now, now),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT * FROM alert_events WHERE id = ?",
                        (cursor.lastrowid,),
                    ).fetchone()
                    logger.info("New alert event: %s | %s (strength=%.1f)",
                                 event_id, title, strength)
                else:
                    # 更新强度
                    # new_peak = max(strength, existing.peak_strength)  # peak updated in mark_sent
                    conn.execute(
                        """UPDATE alert_events
                           SET strength = ?, severity = ?,
                               title = ?, content = ?
                           WHERE event_id = ?""",
                        (strength, severity, title, content, event_id),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT * FROM alert_events WHERE event_id = ?",
                        (event_id,),
                    ).fetchone()
                    if strength > existing.peak_strength * STRENGTH_ESCALATION_RATIO:
                        logger.info(
                            "Alert escalated: %s strength %.1f -> %.1f (peak=%.1f)",
                            event_id, existing.strength, strength, max(existing.peak_strength, strength),
                        )

                return self._row_to_model(row)

            except Exception as exc:
                logger.error("Failed to record event '%s': %s", event_id, exc)
                raise

    def should_alert(self, event_id: str) -> tuple[bool, str]:
        """判断是否需要发送预警推送

        去重策略：
        1. 新事件（从未发送）→ True
        2. 已解除事件 → True
        3. 冷却期内 & 强度未升级 → False
        4. 冷却期内 & 强度强升级 → True（突破冷却）
        5. 冷却期已过 → True

        Returns:
            (should_send: bool, reason: str)
        """
        existing = self._find_event(event_id)

        if existing is None:
            return (True, "new_event")

        if existing.resolved:
            return (True, "resolved")

        now = datetime.now()

        if existing.last_sent is not None:
            elapsed = (now - existing.last_sent).total_seconds() / 60.0

            if elapsed < COOLDOWN_MINUTES:
                # 冷却期内
                if existing.strength > existing.peak_strength * STRENGTH_ESCALATION_RATIO:
                    return (True, "strength_escalated")
                remaining = COOLDOWN_MINUTES - elapsed
                return (False, f"cooldown ({remaining:.0f}min remaining)")

        # 已过冷却期或从未发送
        return (True, "cooldown_expired")

    def mark_sent(self, event_id: str) -> None:
        """标记事件已推送，更新 last_sent 和 sent_count"""
        with self._lock:
            try:
                conn = self._db.get_connection()
                now = datetime.now()
                conn.execute(
                    """UPDATE alert_events
                       SET last_sent = ?, sent_count = sent_count + 1
                       WHERE event_id = ?""",
                    (now, event_id),
                )
                conn.commit()
                logger.debug("Alert sent: %s at %s", event_id, now.isoformat())
            except Exception as exc:
                logger.error("Failed to mark sent for '%s': %s", event_id, exc)
                raise

    def resolve_event(self, event_id: str) -> bool:
        """将事件标记为已解除

        Returns:
            True 表示事件存在且已解除
        """
        with self._lock:
            try:
                conn = self._db.get_connection()
                cursor = conn.execute(
                    "UPDATE alert_events SET resolved = 1 WHERE event_id = ? AND resolved = 0",
                    (event_id,),
                )
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info("Alert resolved: %s", event_id)
                    return True
                return False
            except Exception as exc:
                logger.error("Failed to resolve event '%s': %s", event_id, exc)
                raise

    def get_active_events(self) -> list[AlertEvent]:
        """获取所有活跃（未解除）的预警事件

        Returns:
            按 first_seen 降序排列的活跃事件列表
        """
        try:
            conn = self._db.get_connection()
            rows = conn.execute(
                "SELECT * FROM alert_events WHERE resolved = 0 ORDER BY first_seen DESC"
            ).fetchall()
            return [self._row_to_model(r) for r in rows]
        except Exception as exc:
            logger.error("Failed to get active events: %s", exc)
            raise

    # ── 内部方法 ─────────────────────────────────────────

    def _find_event(self, event_id: str) -> Optional[AlertEvent]:
        """按 event_id 查找事件"""
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT * FROM alert_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            return self._row_to_model(row) if row else None
        except Exception as exc:
            logger.error("Failed to find event '%s': %s", event_id, exc)
            raise

    @staticmethod
    def make_event_id(alert_type: str, symbol: str = "") -> str:
        """生成标准格式的 event_id

        格式: "{alert_type}:{symbol}" 或 "{alert_type}:{short_hash}"
        """
        if symbol:
            return f"{alert_type}:{symbol}"
        import hashlib
        import time
        short = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"{alert_type}:{short}"

    @staticmethod
    def _row_to_model(row) -> AlertEvent:
        """将 SQLite Row 转换为 AlertEvent Pydantic 模型"""
        return AlertEvent(
            id=row["id"],
            event_id=row["event_id"],
            alert_type=row["alert_type"],
            title=row["title"],
            content=row["content"],
            severity=row["severity"],
            related_code=row["related_code"],
            related_sector=row["related_sector"],
            strength=row["strength"],
            peak_strength=row["peak_strength"],
            first_seen=row["first_seen"],
            last_sent=row["last_sent"],
            sent_count=row["sent_count"],
            resolved=row["resolved"],
            created_at=row["created_at"],
        )


# ── 全局单例访问函数 ─────────────────────────────────────

_alert_instance: Optional[AlertAgent] = None


def get_alert_agent() -> AlertAgent:
    """获取 AlertAgent 单例"""
    global _alert_instance  # noqa: PLW0603
    if _alert_instance is None:
        _alert_instance = AlertAgent()
    return _alert_instance
