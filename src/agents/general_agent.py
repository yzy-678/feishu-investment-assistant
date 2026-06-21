"""
通用兜底 Agent

负责处理投资专用 Agent 未命中的普通聊天、代码、解释类问题。
"""

import logging
from typing import Optional

from src.agents.base import BaseAgent, AgentResponse, AgentType
from src.ai.deepseek import DeepSeekClient, get_deepseek

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    """通用 AI 助手 Agent，必须作为最后兜底注册。"""

    _instance: Optional["GeneralAgent"] = None

    def __new__(cls) -> "GeneralAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # type: ignore[attr-defined]
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.deepseek: DeepSeekClient = get_deepseek()
        self._initialized = True
        logger.info("GeneralAgent initialized")

    @property
    def agent_type(self) -> AgentType:
        return AgentType.GENERAL

    def can_handle(self, message: str) -> bool:
        return True

    def handle(self, session_id: str, message: str) -> AgentResponse:
        try:
            reply = self.deepseek.chat_with_memory(session_id, message)
            return AgentResponse(
                success=True,
                agent=AgentType.GENERAL,
                message=reply,
                metadata={"type": "general_chat"},
            )
        except Exception as exc:
            logger.warning("GeneralAgent error: %s", exc)
            return AgentResponse(
                success=False,
                agent=AgentType.GENERAL,
                message="处理通用问题时遇到问题，请稍后再试。",
                metadata={
                    "type": "general_chat",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )


_general_agent_instance: Optional[GeneralAgent] = None


def get_general_agent() -> GeneralAgent:
    """获取 GeneralAgent 单例。"""
    global _general_agent_instance  # noqa: PLW0603
    if _general_agent_instance is None:
        _general_agent_instance = GeneralAgent()
    return _general_agent_instance
