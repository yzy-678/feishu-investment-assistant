"""
Agent 协调器

职责：
1. 管理 Agent 注册
2. 消息路由（按注册顺序匹配第一个可处理的 Agent）
3. 无匹配时兜底回复
4. 日志记录
"""

import logging
import threading
from typing import Optional

from src.agents.base import BaseAgent, AgentType, AgentResponse

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    '抱歉，我没有理解您的问题。您可以尝试以下指令：\n\n'
    '📊 **市场查询**：\n'
    '  \u201c今天市场怎么样\u201d \u201c分析 平安银行\u201d \u201c银行板块怎么样\u201d\n\n'
    '📋 **自选股管理**：\n'
    '  \u201c添加自选 000001\u201d \u201c我的自选\u201d \u201c删除自选 平安银行\u201d\n\n'
    '⚙️ **系统控制**：\n'
    '  \u201c启动\u201d \u201c暂停\u201d \u201c状态\u201d \u201c切换A股\u201d \u201c扫描频率 30\u201d'
)


class AgentCoordinator:
    """Agent 协调器

    按注册顺序依次检查每个 Agent 的 can_handle()，
    将消息路由到第一个匹配的 Agent 处理。

    用法::

        coordinator = AgentCoordinator()
        coordinator.register(market_agent)
        coordinator.register(report_agent)
        response = coordinator.route("user123", "今天市场怎么样")
    """

    def __init__(self) -> None:
        self._agents: list[BaseAgent] = []
        self._lock: threading.Lock = threading.Lock()
        logger.info("AgentCoordinator initialized")

    # ── Agent 管理 ───────────────────────────────────────

    def register(self, agent: BaseAgent) -> None:
        """注册一个 Agent

        Args:
            agent: 实现了 BaseAgent 的具体 Agent 实例

        Raises:
            TypeError: agent 不是 BaseAgent 实例时抛出
        """
        if not isinstance(agent, BaseAgent):
            raise TypeError(
                f"Agent 必须是 BaseAgent 实例，收到: {type(agent).__name__}"
            )

        with self._lock:
            if agent in self._agents:
                logger.debug(
                    "Agent already registered: %s (%s)",
                    agent.__class__.__name__,
                    agent.agent_type.value,
                )
                return

            if agent.agent_type == AgentType.GENERAL:
                self._agents.append(agent)
            else:
                general_index = next(
                    (
                        i for i, registered in enumerate(self._agents)
                        if registered.agent_type == AgentType.GENERAL
                    ),
                    None,
                )
                if general_index is None:
                    self._agents.append(agent)
                else:
                    self._agents.insert(general_index, agent)

        logger.info(
            "Registered agent: %s (%s)",
            agent.__class__.__name__,
            agent.agent_type.value,
        )

    def list_agents(self) -> list[dict]:
        """列出所有已注册的 Agent

        Returns:
            每个 Agent 的摘要信息列表
        """
        with self._lock:
            return [
                {
                    "type": agent.agent_type.value,
                    "class": agent.__class__.__name__,
                    "order": i,
                }
                for i, agent in enumerate(self._agents)
            ]

    # ── 消息路由 ─────────────────────────────────────────

    def route(self, session_id: str, message: str) -> AgentResponse:
        """路由消息到合适的 Agent

        按注册顺序遍历 Agent，返回第一个 can_handle() 返回 True 的 Agent 的处理结果。
        若该 Agent 的 handle() 抛出异常，返回标记为失败的 AgentResponse。
        若无任何 Agent 能处理，返回兜底回复。

        Args:
            session_id: 会话 ID（用户 open_id）
            message: 用户消息文本

        Returns:
            AgentResponse 处理结果
        """
        if not message or not message.strip():
            return AgentResponse(
                success=False,
                agent=AgentType.REPORT,
                message="消息不能为空，请重新输入。",
                metadata={"fallback": True, "reason": "empty_message"},
            )

        agents_copy: list[BaseAgent] = []
        with self._lock:
            agents_copy = list(self._agents)

        for agent in agents_copy:
            if agent.can_handle(message):
                logger.info(
                    "Route matched: %s (%s) ← %s",
                    agent.__class__.__name__,
                    agent.agent_type.value,
                    message[:50],
                )
                try:
                    response = agent.handle(session_id, message)
                    logger.info(
                        "AgentResponse before return: agent=%s message_repr=%r",
                        response.agent.value,
                        response.message,
                    )
                    return response
                except Exception as exc:
                    logger.exception(
                        "Agent %s handle() failed: %s",
                        agent.__class__.__name__,
                        exc,
                    )
                    return AgentResponse(
                        success=False,
                        agent=agent.agent_type,
                        message=f"处理您的请求时出现错误，请稍后再试。",
                        metadata={
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )

        # 无 Agent 命中 → 兜底
        logger.info("No agent matched, returning fallback: %s", message[:50])
        return AgentResponse(
            success=False,
            agent=AgentType.REPORT,
            message=FALLBACK_MESSAGE,
            metadata={"fallback": True, "reason": "no_agent_matched"},
        )


# ── 全局单例访问函数 ─────────────────────────────────────

_coordinator_instance: Optional[AgentCoordinator] = None
_coordinator_lock: threading.Lock = threading.Lock()


def get_coordinator() -> AgentCoordinator:
    """获取 AgentCoordinator 单例"""
    global _coordinator_instance  # noqa: PLW0603
    if _coordinator_instance is None:
        with _coordinator_lock:
            if _coordinator_instance is None:
                _coordinator_instance = AgentCoordinator()
    return _coordinator_instance
