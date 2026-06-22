"""GeneralAgent 单元测试。"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import AgentResponse, AgentType, BaseAgent
from src.agents.coordinator import AgentCoordinator
from src.agents.general_agent import GeneralAgent
from src.ai.deepseek import DeepSeekError
from src.ai.prompts import INVESTMENT_ASSISTANT_SYSTEM_PROMPT


@pytest.fixture(autouse=True)
def reset_singleton():
    GeneralAgent._instance = None
    GeneralAgent._initialized = False  # type: ignore[attr-defined]


@pytest.fixture
def mock_deps():
    with (
        patch("src.agents.general_agent.get_deepseek") as mocked_get,
        patch("src.agents.general_agent.get_memory") as mocked_memory,
    ):
        client = MagicMock()
        client.chat_with_memory.return_value = "通用回复"
        mocked_get.return_value = client
        memory = MagicMock()
        mocked_memory.return_value = memory
        yield {"deepseek": client, "memory": memory}


@pytest.fixture
def mock_deepseek(mock_deps):
    return mock_deps["deepseek"]


@pytest.fixture
def mock_memory(mock_deps):
    return mock_deps["memory"]


@pytest.fixture
def agent(mock_deps):
    return GeneralAgent()


class FakeMarketAgent(BaseAgent):
    @property
    def agent_type(self) -> AgentType:
        return AgentType.MARKET

    def can_handle(self, message: str) -> bool:
        return "分析" in message or "市场" in message

    def handle(self, session_id: str, message: str) -> AgentResponse:
        return AgentResponse(
            success=True,
            agent=AgentType.MARKET,
            message="市场回复",
            metadata={"type": "market"},
        )


class FakeReportAgent(BaseAgent):
    @property
    def agent_type(self) -> AgentType:
        return AgentType.REPORT

    def can_handle(self, message: str) -> bool:
        return "早报" in message or "报告" in message

    def handle(self, session_id: str, message: str) -> AgentResponse:
        return AgentResponse(
            success=True,
            agent=AgentType.REPORT,
            message="报告回复",
            metadata={"type": "report"},
        )


class FakeAlertAgent(BaseAgent):
    @property
    def agent_type(self) -> AgentType:
        return AgentType.ALERT

    def can_handle(self, message: str) -> bool:
        return "预警" in message

    def handle(self, session_id: str, message: str) -> AgentResponse:
        return AgentResponse(
            success=True,
            agent=AgentType.ALERT,
            message="预警回复",
            metadata={"type": "alert"},
        )


class TestGeneralAgent:
    def test_can_handle_always_true(self, agent):
        assert agent.can_handle("你好") is True
        assert agent.can_handle("帮我写Python代码") is True
        assert agent.can_handle("") is True

    def test_handle_calls_chat_with_memory(self, agent, mock_deepseek):
        resp = agent.handle("ou_x", "你好")

        assert resp.success is True
        assert resp.agent == AgentType.GENERAL
        assert resp.message == "通用回复"
        assert resp.metadata == {"type": "general_chat"}
        mock_deepseek.chat_with_memory.assert_called_once_with("ou_x", "你好")

    def test_handle_injects_system_prompt(self, agent, mock_memory):
        agent.handle("ou_x", "你好")

        mock_memory.add_message.assert_called_once_with(
            "ou_x",
            "system",
            INVESTMENT_ASSISTANT_SYSTEM_PROMPT,
        )

    def test_session_id_passed_to_deepseek(self, agent, mock_deepseek):
        agent.handle("session_123", "你是谁")

        session_id = mock_deepseek.chat_with_memory.call_args[0][0]
        assert session_id == "session_123"

    def test_deepseek_error_wrapped(self, agent, mock_deepseek):
        mock_deepseek.chat_with_memory.side_effect = DeepSeekError("API failed")

        resp = agent.handle("ou_x", "你好")

        assert resp.success is False
        assert resp.agent == AgentType.GENERAL
        assert resp.metadata["type"] == "general_chat"
        assert resp.metadata["error_type"] == "DeepSeekError"


class TestCoordinatorWithGeneralAgent:
    def test_general_agent_is_last_fallback(self, agent):
        coordinator = AgentCoordinator()
        coordinator.register(FakeMarketAgent())
        coordinator.register(FakeReportAgent())
        coordinator.register(FakeAlertAgent())
        coordinator.register(agent)

        agents = coordinator.list_agents()
        assert [item["type"] for item in agents] == [
            "market",
            "report",
            "alert",
            "general",
        ]

        assert coordinator.route("ou_x", "分析信维通信").agent == AgentType.MARKET
        assert coordinator.route("ou_x", "生成早报").agent == AgentType.REPORT
        assert coordinator.route("ou_x", "查看预警").agent == AgentType.ALERT
        assert coordinator.route("ou_x", "你好").agent == AgentType.GENERAL

    def test_non_general_registered_after_general_still_takes_priority(self, agent):
        coordinator = AgentCoordinator()
        coordinator.register(agent)
        coordinator.register(FakeMarketAgent())

        agents = coordinator.list_agents()
        assert [item["type"] for item in agents] == ["market", "general"]
        assert coordinator.route("ou_x", "分析信维通信").agent == AgentType.MARKET
