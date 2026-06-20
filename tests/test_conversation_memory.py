"""
ConversationMemory 单元测试

测试覆盖：
- 消息 CRUD（添加、查询、计数）
- 角色校验（user/assistant/system）
- 上下文构建（含摘要和未摘要消息）
- 摘要触发（阈值、折叠保留）
- 会话管理（清空、异常处理）
- 单例 / 线程安全
"""

import threading
from datetime import datetime

import pytest

from src.db import init_database
from src.memory.conversation import (
    ConversationMemory,
    MemoryError,
    get_memory,
)
from src.db.models import ConversationMessage, ConversationSummary


@pytest.fixture(autouse=True)
def reset_memory():
    """每个测试前重置数据库和记忆"""
    init_database()
    mem = get_memory()
    # 清空所有见过的 session
    for sid in ("test_session", "session_a", "session_b", "empty_session"):
        try:
            mem.clear_session(sid)
        except MemoryError:
            pass
    return mem


class TestConversationMemory:
    """对话记忆管理器测试套件"""

    # ── 基础消息操作 ─────────────────────────────────────

    def test_singleton(self):
        """验证单例模式"""
        m1 = get_memory()
        m2 = get_memory()
        assert m1 is m2

    def test_add_message(self, reset_memory):
        """添加一条用户消息"""
        mem = reset_memory
        msg = mem.add_message("test_session", "user", "今天大盘怎么样")
        assert isinstance(msg, ConversationMessage)
        assert msg.session_id == "test_session"
        assert msg.role == "user"
        assert msg.content == "今天大盘怎么样"
        assert isinstance(msg.id, int)
        assert isinstance(msg.created_at, datetime)
        assert msg.is_summarized == 0
        assert mem.count_messages("test_session") == 1

    def test_add_assistant_message(self, reset_memory):
        """添加一条助手回复"""
        mem = reset_memory
        msg = mem.add_message("test_session", "assistant", "今日市场震荡上行")
        assert msg.role == "assistant"
        assert mem.count_messages("test_session") == 1

    def test_add_system_message(self, reset_memory):
        """添加一条系统消息"""
        mem = reset_memory
        msg = mem.add_message("test_session", "system", "初始化会话")
        assert msg.role == "system"
        assert mem.count_messages("test_session") == 1

    def test_add_message_invalid_role(self, reset_memory):
        """无效角色应抛异常"""
        mem = reset_memory
        with pytest.raises(MemoryError, match="无效的角色"):
            mem.add_message("test_session", "admin", "test")

    def test_add_message_empty_content(self, reset_memory):
        """允许空内容消息"""
        mem = reset_memory
        msg = mem.add_message("test_session", "user", "")
        assert msg.content == ""

    def test_add_message_with_case_variation(self, reset_memory):
        """角色不区分大小写"""
        mem = reset_memory
        msg = mem.add_message("test_session", "USER", "Hello")
        assert msg.role == "user"

    # ── 多 session 隔离 ──────────────────────────────────

    def test_multi_session_isolation(self, reset_memory):
        """不同会话的消息互不干扰"""
        mem = reset_memory
        mem.add_message("session_a", "user", "消息A")
        mem.add_message("session_b", "user", "消息B")
        assert mem.count_messages("session_a") == 1
        assert mem.count_messages("session_b") == 1

    # ── 获取最近消息 ─────────────────────────────────────

    def test_get_recent_messages(self, reset_memory):
        """获取最近消息"""
        mem = reset_memory
        for i in range(10):
            mem.add_message("test_session", "user", f"query_{i}")
            mem.add_message("test_session", "assistant", f"reply_{i}")
        msgs = mem.get_recent_messages("test_session", limit=20)
        assert len(msgs) == 20
        assert msgs[0].role == "user"  # 时间正序，第一条是 user

    def test_get_recent_messages_custom_limit(self, reset_memory):
        """自定义 limit"""
        mem = reset_memory
        for i in range(10):
            mem.add_message("test_session", "user", f"q_{i}")
        msgs = mem.get_recent_messages("test_session", limit=5)
        assert len(msgs) == 5

    def test_get_recent_messages_empty(self, reset_memory):
        """空会话返回空列表"""
        mem = reset_memory
        msgs = mem.get_recent_messages("empty_session")
        assert msgs == []

    def test_get_recent_messages_excludes_summarized(self, reset_memory):
        """已摘要的消息不应返回"""
        mem = reset_memory
        for i in range(60):
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        # 摘要后只有 20 条保留
        msgs = mem.get_recent_messages("test_session")
        assert len(msgs) == 20

    # ── 计数 ─────────────────────────────────────────────

    def test_count_messages(self, reset_memory):
        """统计消息数量"""
        mem = reset_memory
        assert mem.count_messages("test_session") == 0
        mem.add_message("test_session", "user", "hello")
        assert mem.count_messages("test_session") == 1
        mem.add_message("test_session", "assistant", "hi")
        assert mem.count_messages("test_session") == 2

    def test_count_messages_excludes_summarized(self, reset_memory):
        """已摘要的消息不应计入"""
        mem = reset_memory
        for i in range(55):
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        assert mem.count_messages("test_session") == 20

    # ── 摘要 ─────────────────────────────────────────────

    def test_summarize_not_triggered_under_threshold(self, reset_memory):
        """低于阈值不触发摘要"""
        mem = reset_memory
        for i in range(30):
            mem.add_message("test_session", "user", f"msg_{i}")
        result = mem.summarize_if_needed("test_session")
        assert result is False
        assert mem.count_messages("test_session") == 30

    def test_summarize_triggered_at_threshold(self, reset_memory):
        """达到阈值触发摘要"""
        mem = reset_memory
        for i in range(51):
            mem.add_message("test_session", "user", f"msg_{i}")
        result = mem.summarize_if_needed("test_session")
        assert result is True

    def test_summarize_keeps_recent_messages(self, reset_memory):
        """摘要后保留最近 20 条"""
        mem = reset_memory
        for i in range(51):
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        assert mem.count_messages("test_session") == 20
        msgs = mem.get_recent_messages("test_session")
        assert len(msgs) == 20
        # 验证保留的是最新的消息
        assert msgs[-1].content == "msg_50"

    def test_get_summary_after_summarize(self, reset_memory):
        """摘要后能获取到摘要内容"""
        mem = reset_memory
        for i in range(55):
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        summary = mem.get_summary("test_session")
        assert summary is not None
        assert isinstance(summary, ConversationSummary)
        assert "消息折叠" in summary.summary

    def test_get_summary_before_summarize(self, reset_memory):
        """未摘要时返回 None"""
        mem = reset_memory
        summary = mem.get_summary("test_session")
        assert summary is None

    def test_summarize_idempotent(self, reset_memory):
        """多次触发摘要不应重复压缩"""
        mem = reset_memory
        for i in range(55):
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        assert mem.count_messages("test_session") == 20

        # 再加一些消息，第二次触发（需要超过50条阈值）
        for i in range(56, 87):  # 20 + 31 = 51 > 50
            mem.add_message("test_session", "user", f"msg_{i}")
        result2 = mem.summarize_if_needed("test_session")
        assert result2 is True
        assert mem.count_messages("test_session") == 20

    def test_summarize_accumulates_old_summary(self, reset_memory):
        """多次摘要应叠加旧摘要"""
        mem = reset_memory
        for i in range(55):
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        s1 = mem.get_summary("test_session")
        assert s1 is not None
        assert "历史摘要" not in s1.summary  # 第一次无历史

        for i in range(56, 87):  # 20 + 31 = 51 > 50
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        s2 = mem.get_summary("test_session")
        assert "历史摘要" in s2.summary  # 第二次包含第一次的摘要

    # ── 上下文构建 ───────────────────────────────────────

    def test_get_context_no_summary(self, reset_memory):
        """无摘要时仅返回消息列表"""
        mem = reset_memory
        mem.add_message("test_session", "user", "你好")
        mem.add_message("test_session", "assistant", "你好！")
        ctx = mem.get_context("test_session")
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"
        assert ctx[1]["role"] == "assistant"

    def test_get_context_with_summary(self, reset_memory):
        """有摘要时第一条为 system 消息"""
        mem = reset_memory
        for i in range(55):
            mem.add_message("test_session", "user", f"msg_{i}")
        mem.summarize_if_needed("test_session")
        ctx = mem.get_context("test_session")
        assert ctx[0]["role"] == "system"
        assert "消息折叠" in ctx[0].get("content", "")

    # ── 会话管理 ─────────────────────────────────────────

    def test_clear_session(self, reset_memory):
        """清空会话"""
        mem = reset_memory
        mem.add_message("test_session", "user", "测试")
        result = mem.clear_session("test_session")
        assert result is True
        assert mem.count_messages("test_session") == 0
        assert mem.get_summary("test_session") is None

    def test_clear_nonexistent_session(self, reset_memory):
        """清空不存在的会话返回 False"""
        mem = reset_memory
        result = mem.clear_session("nonexistent")
        assert result is False

    # ── 线程安全 ─────────────────────────────────────────

    def test_thread_safety(self, reset_memory):
        """并发写入线程安全"""
        mem = reset_memory
        errors: list[Exception] = []

        def add_msg(i: int):
            try:
                mem.add_message("test_session", "user", f"thread_msg_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_msg, args=(i,))
            for i in range(30)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert mem.count_messages("test_session") == 30

    # ── 边界条件 ─────────────────────────────────────────

    def test_exactly_50_messages_no_summary(self, reset_memory):
        """恰好 50 条不触发摘要"""
        mem = reset_memory
        for i in range(50):
            mem.add_message("test_session", "user", f"msg_{i}")
        assert mem.summarize_if_needed("test_session") is False
        assert mem.count_messages("test_session") == 50

    def test_51_messages_triggers_summary(self, reset_memory):
        """51 条触发摘要"""
        mem = reset_memory
        for i in range(51):
            mem.add_message("test_session", "user", f"msg_{i}")
        assert mem.summarize_if_needed("test_session") is True
        assert mem.count_messages("test_session") == 20

    def test_long_content_message(self, reset_memory):
        """长文本消息"""
        mem = reset_memory
        long_text = "测试" * 1000
        msg = mem.add_message("test_session", "user", long_text)
        assert len(msg.content) == 2000
        assert mem.count_messages("test_session") == 1

    def test_context_format(self, reset_memory):
        """get_context 返回格式适配 LLM"""
        mem = reset_memory
        mem.add_message("test_session", "user", "分析大盘")
        ctx = mem.get_context("test_session")
        assert isinstance(ctx, list)
        assert all(isinstance(m, dict) for m in ctx)
        assert all("role" in m and "content" in m for m in ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
