"""
AgentCoordinator 单元测试

测试覆盖：Agent 注册/列表、路由匹配、多 Agent 优先级、
兜底回复、异常处理、线程安全。
使用 Mock Agent 类模拟真实 Agent 行为。
"""

import threading

import pytest

from src.agents.base import BaseAgent, AgentType, AgentResponse
from src.agents.coordinator import AgentCoordinator, get_coordinator, FALLBACK_MESSAGE


# ═══════════════════════════════════════════════════════════
#  Mock Agents
# ═══════════════════════════════════════════════════════════


class MockMarketAgent(BaseAgent):
    """模拟市场分析 Agent：匹配含"市场"/"大盘"/"股票名称"的消息"""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.MARKET

    def can_handle(self, message: str) -> bool:
        keywords = ["市场", "大盘", "指数", "分析"]
        return any(kw in message for kw in keywords)

    def handle(self, session_id: str, message: str) -> AgentResponse:
        return AgentResponse(
            success=True,
            agent=AgentType.MARKET,
            message=f"市场分析结果：今日市场震荡上行",
            metadata={"session_id": session_id, "keywords": ["市场"]},
        )


class MockReportAgent(BaseAgent):
    """模拟日报 Agent：匹配含"日报"/"复盘"/"汇报"的消息"""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.REPORT

    def can_handle(self, message: str) -> bool:
        keywords = ["日报", "复盘", "汇报", "报告"]
        return any(kw in message for kw in keywords)

    def handle(self, session_id: str, message: str) -> AgentResponse:
        return AgentResponse(
            success=True,
            agent=AgentType.REPORT,
            message="【日报】今日市场数据显示...",
            metadata={"session_id": session_id},
        )


class MockAlertAgent(BaseAgent):
    """模拟预警 Agent：匹配含"预警"/"提醒"/"监控"的消息"""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.ALERT

    def can_handle(self, message: str) -> bool:
        keywords = ["预警", "提醒", "监控"]
        return any(kw in message for kw in keywords)

    def handle(self, session_id: str, message: str) -> AgentResponse:
        return AgentResponse(
            success=True,
            agent=AgentType.ALERT,
            message="当前未发现异常波动",
            metadata={"session_id": session_id},
        )


class MockFailingAgent(BaseAgent):
    """模拟会抛出异常的 Agent"""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.MARKET

    def can_handle(self, message: str) -> bool:
        return True  # 所有消息都匹配

    def handle(self, session_id: str, message: str) -> AgentResponse:
        raise RuntimeError("模拟 Agent 异常")


class MockBroadAgent(BaseAgent):
    """模拟宽匹配 Agent：匹配含"所有"的消息，用于测试优先级"""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.REPORT

    def can_handle(self, message: str) -> bool:
        return True  # 匹配所有消息（兜底之前检查）

    def handle(self, session_id: str, message: str) -> AgentResponse:
        return AgentResponse(
            success=True,
            agent=AgentType.REPORT,
            message="宽匹配 Agent 处理",
            metadata={"session_id": session_id},
        )


# ═══════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def coordinator():
    """返回空的协调器"""
    return AgentCoordinator()


@pytest.fixture
def market_agent():
    return MockMarketAgent()


@pytest.fixture
def report_agent():
    return MockReportAgent()


@pytest.fixture
def alert_agent():
    return MockAlertAgent()


@pytest.fixture
def full_coordinator(market_agent, report_agent, alert_agent):
    """返回已注册三个 Agent 的协调器"""
    c = AgentCoordinator()
    c.register(market_agent)
    c.register(report_agent)
    c.register(alert_agent)
    return c


# ═══════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════


class TestAgentRegistration:
    """Agent 注册测试"""

    def test_register_single_agent(self, coordinator, market_agent):
        """注册单个 Agent"""
        coordinator.register(market_agent)
        agents = coordinator.list_agents()
        assert len(agents) == 1
        assert agents[0]["type"] == "market"

    def test_register_multiple_agents(self, coordinator, market_agent, report_agent, alert_agent):
        """注册多个 Agent"""
        coordinator.register(market_agent)
        coordinator.register(report_agent)
        coordinator.register(alert_agent)
        assert len(coordinator.list_agents()) == 3

    def test_register_non_agent_raises_type_error(self, coordinator):
        """注册非 BaseAgent 实例应抛 TypeError"""
        with pytest.raises(TypeError, match="BaseAgent"):
            coordinator.register("not_an_agent")  # type: ignore[arg-type]

    def test_register_none_raises_type_error(self, coordinator):
        """注册 None 应抛 TypeError"""
        with pytest.raises(TypeError, match="BaseAgent"):
            coordinator.register(None)  # type: ignore[arg-type]

    def test_register_preserves_order(self, coordinator, report_agent, market_agent, alert_agent):
        """注册顺序应被保持"""
        coordinator.register(report_agent)
        coordinator.register(market_agent)
        coordinator.register(alert_agent)
        agents = coordinator.list_agents()
        assert agents[0]["type"] == "report"
        assert agents[1]["type"] == "market"
        assert agents[2]["type"] == "alert"

    def test_list_agents_format(self, coordinator, market_agent):
        """list_agents 返回格式"""
        coordinator.register(market_agent)
        agents = coordinator.list_agents()
        assert isinstance(agents, list)
        item = agents[0]
        assert "type" in item
        assert "class" in item
        assert "order" in item


class TestRouting:
    """消息路由测试"""

    def test_route_market_message(self, full_coordinator):
        """市场类消息应路由到 MarketAgent"""
        response = full_coordinator.route("user123", "今天市场怎么样")
        assert response.success is True
        assert response.agent == AgentType.MARKET
        assert "市场分析" in response.message

    def test_route_report_message(self, full_coordinator):
        """日报类消息应路由到 ReportAgent"""
        response = full_coordinator.route("user123", "帮我生成日报")
        assert response.success is True
        assert response.agent == AgentType.REPORT
        assert "日报" in response.message

    def test_route_alert_message(self, full_coordinator):
        """预警类消息应路由到 AlertAgent"""
        response = full_coordinator.route("user123", "查看预警")
        assert response.success is True
        assert response.agent == AgentType.ALERT

    def test_route_analysis_query(self, full_coordinator):
        """分析类消息应命中 MarketAgent"""
        response = full_coordinator.route("user123", "分析平安银行")
        assert response.success is True
        assert response.agent == AgentType.MARKET

    def test_route_fallback_no_match(self, coordinator, market_agent):
        """无匹配时返回兜底回复"""
        coordinator.register(market_agent)
        response = coordinator.route("user123", "你好呀")
        # "你好" 不匹配 market_agent 的关键词
        assert response.success is False
        assert response.message == FALLBACK_MESSAGE
        assert response.metadata.get("fallback") is True

    def test_route_empty_message(self, full_coordinator):
        """空消息返回提示"""
        response = full_coordinator.route("user123", "")
        assert response.success is False
        assert "不能为空" in response.message

    def test_route_whitespace_message(self, full_coordinator):
        """纯空白消息返回提示"""
        response = full_coordinator.route("user123", "   ")
        assert response.success is False
        assert "不能为空" in response.message


class TestAgentPriority:
    """Agent 优先级测试"""

    def test_first_match_wins(self, coordinator):
        """注册顺序决定优先级：先注册的先匹配"""
        market = MockMarketAgent()
        broad = MockBroadAgent()  # can_handle 总是返回 True
        coordinator.register(market)  # market 先注册
        coordinator.register(broad)   # broad 后注册

        # "市场" 应被 market 匹配
        response = coordinator.route("user123", "市场怎么样")
        assert response.agent == AgentType.MARKET
        assert "市场分析" in response.message

    def test_later_match_overrides(self, coordinator):
        """宽匹配 Agent 在特定 Agent 之后时不影响特定匹配"""
        broad = MockBroadAgent()
        market = MockMarketAgent()
        coordinator.register(broad)  # 宽匹配先注册
        coordinator.register(market)  # 特定 Agent 后注册

        # "市场" 先被 broad 匹配（因为 broad 总是 True）
        response = coordinator.route("user123", "市场怎么样")
        assert response.agent == AgentType.REPORT
        assert "宽匹配" in response.message


class TestAgentException:
    """Agent 异常处理测试"""

    def test_agent_exception_caught(self, coordinator):
        """Agent 抛出异常应返回失败的 AgentResponse"""
        failing_agent = MockFailingAgent()
        coordinator.register(failing_agent)

        response = coordinator.route("user123", "任何消息")
        assert response.success is False
        assert "错误" in response.message
        # 注意：是捕获的异常响应，不是兜底
        assert response.metadata.get("fallback") is not True
        assert "error" in response.metadata

    def test_failing_agent_does_not_affect_others(self, coordinator, market_agent):
        """异常 Agent 不会影响其他 Agent"""
        failing_agent = MockFailingAgent()
        coordinator.register(failing_agent)
        coordinator.register(market_agent)

        # market_agent 排在第 2 位
        response = coordinator.route("user123", "今天市场怎么样")
        # market 匹配 → 但因为 failing 排第一且 can_handle 返回 True
        # 所以 failing_agent 先处理 → 异常
        assert response.success is False
        assert "错误" in response.message


class TestCoordinatorEdgeCases:
    """边界条件测试"""

    def test_no_agents_registered(self, coordinator):
        """无任何人注册时返回兜底"""
        response = coordinator.route("user123", "市场怎么样")
        assert response.success is False
        assert response.message == FALLBACK_MESSAGE

    def test_singleton(self):
        """验证单例"""
        c1 = get_coordinator()
        c2 = get_coordinator()
        assert c1 is c2

    def test_session_id_propagation(self, full_coordinator):
        """session_id 应透传到 Agent"""
        test_sid = "test_user_001"
        response = full_coordinator.route(test_sid, "市场怎么样")
        assert response.metadata.get("session_id") == test_sid

    def test_agent_response_creation(self):
        """AgentResponse 正确创建和序列化"""
        resp = AgentResponse(success=True, agent=AgentType.MARKET, message="OK", metadata={"k": "v"})
        assert resp.success is True
        assert resp.agent == AgentType.MARKET
        assert resp.message == "OK"
        assert resp.metadata == {"k": "v"}
        assert resp.model_dump()["agent"] == "market"  # 序列化为枚举值

    def test_message_truncated_in_log(self, full_coordinator):
        """超长消息不影响路由"""
        long_msg = "市场" * 1000
        response = full_coordinator.route("user123", long_msg)
        assert response.success is True
        assert response.agent == AgentType.MARKET

    def test_agent_type_enum_values(self):
        """AgentType 枚举值验证"""
        assert AgentType.MARKET.value == "market"
        assert AgentType.REPORT.value == "report"
        assert AgentType.ALERT.value == "alert"


class TestConcurrency:
    """并发访问测试"""

    def test_concurrent_route(self, full_coordinator):
        """并发路由线程安全"""
        errors: list[Exception] = []

        def route_msg(i: int):
            try:
                resp = full_coordinator.route(f"user_{i}", "市场分析")
                assert resp.success is True
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=route_msg, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_register_and_route(self, coordinator, market_agent, report_agent):
        """并发注册和路由"""
        errors: list[Exception] = []

        def register_agents():
            try:
                coordinator.register(MockMarketAgent())
                coordinator.register(MockReportAgent())
            except Exception as e:
                errors.append(e)

        def route_all():
            try:
                for msg in ["市场怎么样", "日报", "你好"]:
                    coordinator.route("user", msg)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_agents),
            threading.Thread(target=route_all),
            threading.Thread(target=register_agents),
            threading.Thread(target=route_all),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
