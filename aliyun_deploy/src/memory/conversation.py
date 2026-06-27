"""
对话记忆管理器

分层记忆机制：
1. Short-Term Memory（conversations 表）
   - 保存每条对话记录，按 session_id 管理
   - is_summarized 标记已被摘要折叠的消息

2. Long-Term Summary（conversation_summaries 表）
   - 超过 50 条未摘要消息时自动触发
   - 保留最近 20 条，将更早的消息压缩为文本摘要
   - 第一版使用结构化文本折叠，后续接入 DeepSeek 智能摘要
"""

import logging
import threading
from datetime import datetime
from typing import Optional

from src.db import get_database
from src.db.models import (
    ConversationMessage,
    ConversationMessageCreate,
    ConversationSummary,
    RoleType,
)

logger = logging.getLogger(__name__)

# ── 阈值常量 ─────────────────────────────────────────────

MAX_MESSAGES_BEFORE_SUMMARY = 50
"""触发摘要的消息数量阈值"""

KEEP_AFTER_SUMMARY = 20
"""摘要后保留的最近消息数"""


class MemoryError(Exception):
    """记忆模块操作异常"""


class ConversationMemory:
    """对话记忆管理器（线程安全单例）

    用法::

        mem = ConversationMemory()
        mem.add_message("user_123", "user", "今天大盘怎么样")
        context = mem.get_context("user_123")
    """

    _instance: Optional["ConversationMemory"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ConversationMemory":
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
        self._initialized = True
        logger.info("ConversationMemory initialized")

    # ── 消息写入 ─────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str) -> ConversationMessage:
        """保存一条对话消息

        Args:
            session_id: 会话 ID（通常为用户 open_id）
            role: 角色（user / assistant / system）
            content: 消息内容

        Returns:
            新增的 ConversationMessage

        Raises:
            MemoryError: role 无效时抛出
        """
        # 校验 role
        try:
            role_enum = RoleType(role.strip().lower())
        except ValueError:
            raise MemoryError(
                f"无效的角色类型: {role}，仅支持: {', '.join(r.value for r in RoleType)}"
            )

        now = datetime.now()

        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                """INSERT INTO conversations (session_id, role, content, metadata,
                                              conversation_id, is_summarized, created_at)
                   VALUES (?, ?, ?, '{}', '', 0, ?)""",
                (session_id, role_enum.value, content, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            logger.debug("Added %s message to session '%s' (%d chars)",
                         role, session_id, len(content))
            return self._row_to_message(row)
        except Exception as exc:
            logger.error("Failed to add message for session '%s': %s",
                         session_id, exc)
            raise MemoryError(f"保存消息失败") from exc

    # ── 消息读取 ─────────────────────────────────────────

    def get_recent_messages(
        self, session_id: str, limit: int = 50
    ) -> list[ConversationMessage]:
        """获取最近 N 条未摘要的消息

        Args:
            session_id: 会话 ID
            limit: 最大返回条数

        Returns:
            按时间正序排列的消息列表
        """
        try:
            conn = self._db.get_connection()
            rows = conn.execute(
                """SELECT * FROM conversations
                   WHERE session_id = ? AND is_summarized = 0
                   ORDER BY created_at DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
            rows.reverse()  # 转为时间正序
            return [self._row_to_message(r) for r in rows]
        except Exception as exc:
            logger.error("Failed to get recent messages for '%s': %s",
                         session_id, exc)
            raise MemoryError(f"获取最近消息失败") from exc

    def get_context(self, session_id: str) -> list[dict[str, str]]:
        """获取完整的 LLM 对话上下文

        返回格式适合直接传递给 DeepSeek chat API::

            [
                {"role": "system", "content": "对话摘要..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
            ]

        Args:
            session_id: 会话 ID

        Returns:
            LLM 消息列表
        """
        context: list[dict[str, str]] = []

        # 1. 如果有摘要，作为第一条 system 消息
        summary = self.get_summary(session_id)
        if summary is not None:
            context.append({
                "role": "system",
                "content": f"以下是本次对话的简要回顾，供参考：\n{summary.summary}",
            })

        # 2. 追加最近消息
        recent = self.get_recent_messages(session_id, limit=MAX_MESSAGES_BEFORE_SUMMARY)
        for msg in recent:
            context.append({
                "role": msg.role,
                "content": msg.content,
            })

        return context

    # ── 摘要管理 ─────────────────────────────────────────

    def summarize_if_needed(self, session_id: str) -> bool:
        """检查是否需要触发摘要，需要则执行

        条件：未摘要消息数量超过 MAX_MESSAGES_BEFORE_SUMMARY 时触发。
        执行：保留最近 KEEP_AFTER_SUMMARY 条，将更早的消息压缩为摘要文本。

        第一版使用文本枚举方式的简单摘要，后续可替换为 DeepSeek 智能摘要。

        Returns:
            是否执行了摘要操作
        """
        try:
            conn = self._db.get_connection()
            rows = conn.execute(
                """SELECT * FROM conversations
                   WHERE session_id = ? AND is_summarized = 0
                   ORDER BY created_at ASC""",
                (session_id,),
            ).fetchall()
        except Exception as exc:
            logger.error("Failed to check messages for summary '%s': %s",
                         session_id, exc)
            raise MemoryError(f"检查摘要条件失败") from exc

        total = len(rows)
        if total <= MAX_MESSAGES_BEFORE_SUMMARY:
            return False

        # 保留最近 KEEP_AFTER_SUMMARY 条，折叠更早的
        to_summarize = rows[:-KEEP_AFTER_SUMMARY]
        # to_keep = rows[-KEEP_AFTER_SUMMARY:]（隐式保留）

        # 获取已有摘要
        existing_summary = self.get_summary(session_id)
        summary_text = self._build_compact_summary(
            to_summarize, existing_summary
        )

        try:
            conn = self._db.get_connection()
            # 保存新摘要
            now = datetime.now()
            conn.execute(
                """INSERT INTO conversation_summaries
                   (session_id, summary, summary_type, token_count, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, summary_text, "conversation", 0, now),
            )

            # 标记被折叠的消息
            for r in to_summarize:
                conn.execute(
                    "UPDATE conversations SET is_summarized = 1 WHERE id = ?",
                    (r["id"],),
                )
            conn.commit()
            logger.info(
                "Summarized session '%s': folded %d messages, kept %d",
                session_id, len(to_summarize), KEEP_AFTER_SUMMARY,
            )
            return True
        except Exception as exc:
            logger.error("Failed to save summary for '%s': %s",
                         session_id, exc)
            raise MemoryError(f"保存摘要失败") from exc

    def get_summary(self, session_id: str) -> Optional[ConversationSummary]:
        """获取该会话的最新摘要

        Returns:
            ConversationSummary 或 None（无摘要时）
        """
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                """SELECT * FROM conversation_summaries
                   WHERE session_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return ConversationSummary(
                id=row["id"],
                session_id=row["session_id"],
                summary=row["summary"],
                summary_type=row["summary_type"],
                token_count=row["token_count"],
                created_at=row["created_at"],
            )
        except Exception as exc:
            logger.error("Failed to get summary for '%s': %s",
                         session_id, exc)
            raise MemoryError(f"获取摘要失败") from exc

    # ── 会话管理 ─────────────────────────────────────────

    def clear_session(self, session_id: str) -> bool:
        """清空指定会话的所有消息和摘要

        Returns:
            是否成功清空（True 表示有数据被删除）
        """
        try:
            conn = self._db.get_connection()
            del_msgs = conn.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,),
            ).rowcount
            del_summaries = conn.execute(
                "DELETE FROM conversation_summaries WHERE session_id = ?",
                (session_id,),
            ).rowcount
            conn.commit()
            if del_msgs > 0 or del_summaries > 0:
                logger.info("Cleared session '%s': %d msgs, %d summaries",
                            session_id, del_msgs, del_summaries)
                return True
            return False
        except Exception as exc:
            logger.error("Failed to clear session '%s': %s",
                         session_id, exc)
            raise MemoryError(f"清空会话失败") from exc

    def count_messages(self, session_id: str) -> int:
        """统计该会话中未摘要的消息数量"""
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM conversations WHERE session_id = ? AND is_summarized = 0",
                (session_id,),
            ).fetchone()
            return row["cnt"]
        except Exception as exc:
            logger.error("Failed to count messages for '%s': %s",
                         session_id, exc)
            raise MemoryError(f"统计消息数量失败") from exc

    # ── 内部方法 ─────────────────────────────────────────

    def _build_compact_summary(
        self,
        to_summarize: list,
        existing_summary: Optional[ConversationSummary],
    ) -> str:
        """构建紧凑的文本摘要（第一版：结构化文本）

        后续可替换为 DeepSeek 调用：:

            client.chat([{
                "role": "system",
                "content": "请对以下对话内容进行摘要..."
            }, ...])

        Args:
            to_summarize: 需折叠的 SQLite Row 列表
            existing_summary: 已有的历史摘要（可选）

        Returns:
            摘要文本
        """
        parts: list[str] = []

        # 旧摘要
        if existing_summary is not None:
            parts.append(f"[历史摘要] {existing_summary.summary}")

        # 统计
        user_msgs = [r for r in to_summarize if r["role"] == "user"]
        assistant_msgs = [r for r in to_summarize if r["role"] == "assistant"]
        system_msgs = [r for r in to_summarize if r["role"] == "system"]

        fold_info = (
            f"[消息折叠] 以下 {len(to_summarize)} 条历史记录被折叠——"
            f"用户 {len(user_msgs)} 条 / 助手 {len(assistant_msgs)} 条"
            f"{f' / 系统 {len(system_msgs)} 条' if system_msgs else ''}"
        )
        parts.append(fold_info)

        # 用户问题要点
        if user_msgs:
            recent_user = user_msgs[-8:]
            queries = [(r["content"][:80], len(r["content"]) > 80) for r in recent_user]
            parts.append("[用户问题摘要]")
            for i, (q, is_long) in enumerate(queries, 1):
                suffix = "…" if is_long else ""
                parts.append(f"  {i}. {q}{suffix}")

        return "\n".join(parts)

    # ── 行转换 ───────────────────────────────────────────

    @staticmethod
    def _row_to_message(row) -> ConversationMessage:
        """将 SQLite Row 转换为 ConversationMessage"""
        return ConversationMessage(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=row["metadata"],
            conversation_id=row["conversation_id"],
            is_summarized=row["is_summarized"],
            created_at=row["created_at"],
        )


# ── 全局单例访问函数 ─────────────────────────────────────

_memory_instance: Optional[ConversationMemory] = None
_memory_lock: threading.Lock = threading.Lock()


def get_memory() -> ConversationMemory:
    """获取 ConversationMemory 单例"""
    global _memory_instance  # noqa: PLW0603
    if _memory_instance is None:
        with _memory_lock:
            if _memory_instance is None:
                _memory_instance = ConversationMemory()
    return _memory_instance
