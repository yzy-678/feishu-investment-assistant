"""
Agent 系统抽象定义

定义 AgentType、AgentResponse 和 BaseAgent 抽象基类。
所有具体 Agent（market/report/alert）继承 BaseAgent 实现。
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Agent 类型枚举"""
    MARKET = "market"
    """市场分析 Agent"""
    REPORT = "report"
    """报告生成 Agent"""
    ALERT = "alert"
    """实时预警 Agent"""
    GENERAL = "general"
    """通用聊天 Agent"""


class AgentResponse(BaseModel):
    """Agent 处理结果"""
    success: bool = Field(..., description="是否处理成功")
    agent: AgentType = Field(..., description="处理的 Agent 类型")
    message: str = Field(..., description="回复内容")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据（日志/调试）",
    )


class BaseAgent(ABC):
    """Agent 抽象基类

    所有具体 Agent 必须实现 can_handle 和 handle 方法。

    用法::

        class MyAgent(BaseAgent):
            @property
            def agent_type(self) -> AgentType:
                return AgentType.MARKET

            def can_handle(self, message: str) -> bool:
                return "大盘" in message

            def handle(self, session_id: str, message: str) -> AgentResponse:
                ...
    """

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """返回此 Agent 的类型标识"""

    @abstractmethod
    def can_handle(self, message: str) -> bool:
        """判断此 Agent 是否能处理该消息

        Args:
            message: 用户输入的消息

        Returns:
            True 表示此 Agent 可以处理该消息
        """

    @abstractmethod
    def handle(self, session_id: str, message: str) -> AgentResponse:
        """处理消息并返回结果

        Args:
            session_id: 会话 ID（用户 open_id）
            message: 用户输入的消息

        Returns:
            AgentResponse 包含处理结果
        """
