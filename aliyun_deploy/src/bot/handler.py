"""
飞书消息处理器

统一处理飞书机器人收到的所有消息：
1. 系统命令（启动/暂停/状态/切换市场等）
2. 自选股管理（添加/删除/查看/清空）
3. 市场问答（通过 Coordinator 路由到 Agent）
"""

import json
import logging
import re
from typing import Optional

from src.ai.deepseek import DeepSeekError
from src.bot.client import FeishuClient, FeishuError, get_feishu_client
from src.config.manager import ConfigManager, get_config
from src.watchlist.manager import WatchlistError, WatchlistManager, get_watchlist
from src.agents.coordinator import AgentCoordinator, get_coordinator
from src.agents.market_agent import get_market_agent
from src.agents.report_agent import get_report_agent
from src.agents.alert_agent import get_alert_agent

logger = logging.getLogger(__name__)

MAX_REPLY_LENGTH = 1500
"""飞书单条消息最大字符数"""


class CommandError(Exception):
    """命令处理异常"""


class MessageHandler:
    """飞书消息处理器

    处理流程：
        收到消息 → 解析事件 → 识别命令/自选股操作 → 处理
                    → 未匹配 → Coordinator.route() → Agent 处理
                    → 发送回复
    """

    def __init__(self) -> None:
        self.feishu: FeishuClient = get_feishu_client()
        self.coordinator: AgentCoordinator = get_coordinator()
        self.config: ConfigManager = get_config()
        self.watchlist: WatchlistManager = get_watchlist()
        self._register_agents()
        logger.info("MessageHandler initialized")

    # ── Agent 注册 ─────────────────────────────────────────

    def _register_agents(self) -> None:
        """向 Coordinator 注册所有 Agent"""
        agents = [
            get_market_agent(),
            get_report_agent(),
            get_alert_agent(),
        ]
        for agent in agents:
            self.coordinator.register(agent)
        logger.info(
            "Registered %d agents: %s",
            len(agents),
            ", ".join(a.__class__.__name__ for a in agents),
        )

    # ── 事件入口 ─────────────────────────────────────────

    def handle_event(self, raw_event: dict) -> Optional[str]:
        """处理飞书事件回调

        解析 event 数据，提取消息内容，
        处理后通过 FeishuClient 回复。

        Args:
            raw_event: 飞书事件回调的完整 JSON 数据

        Returns:
            回复文本（用于日志/调试），None 表示无需回复
        """
        try:
            # 只处理消息接收事件
            event_type = (
                raw_event.get("header", {}).get("event_type", "")
            )
            if event_type != "im.message.receive_v1":
                return None

            event = raw_event.get("event", {})
            message = event.get("message", {})
            sender = event.get("sender", {}).get("sender_id", {})

            message_id: str = message.get("message_id", "")
            content_str: str = message.get("content", "{}")
            open_id: str = sender.get("open_id", "")

            if not message_id or not open_id:
                logger.warning("Missing message_id or open_id in event")
                return None

            # 解析消息内容
            try:
                content_data = json.loads(content_str)
                text = content_data.get("text", "").strip()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to parse message content: %s", exc)
                return None

            if not text:
                logger.debug("Empty message content, skipping")
                return None

            # 处理消息
            reply = self.process_message(open_id, message_id, text)

            # 发送回复
            if reply:
                truncated = reply[:MAX_REPLY_LENGTH]
                self.feishu.reply_text(message_id, truncated)
                logger.info(
                    "Replied to %s: %.40s... (%d chars)",
                    open_id[:8], text, len(truncated),
                )
                return truncated

            return None

        except (FeishuError, DeepSeekError) as exc:
            logger.error("Handler error: %s", exc)
            return None

    # ── 消息处理 ─────────────────────────────────────────

    def process_message(self, open_id: str, message_id: str, text: str) -> str:
        """处理单条消息并生成回复

        优先级：
        1. 系统命令
        2. 自选股操作
        3. Coordinator 路由到 Agent
        """
        # 1. 系统命令
        reply = self._handle_commands(text)
        if reply is not None:
            return reply

        # 2. 自选股操作
        reply = self._handle_watchlist(text)
        if reply is not None:
            return reply

        # 3. 通过 Coordinator 路由
        response = self.coordinator.route(open_id, text)
        return response.message

    # ── 系统命令 ─────────────────────────────────────────

    def _handle_commands(self, text: str) -> Optional[str]:
        """处理系统控制命令"""

        # 启动
        if text in ("启动", "开机", "开启"):
            self.config.set_enabled(True)
            return "✅ 系统已启动，盘中预警和日报已恢复。"

        # 暂停
        if text in ("暂停", "关机", "停止"):
            self.config.set_enabled(False)
            return "⏸️ 系统已暂停，不再推送日报和预警。"

        # 状态
        if text == "状态":
            enabled = self.config.get_enabled()
            market = self.config.get_market()
            interval = self.config.get_scan_interval()
            return (
                f"📊 **系统状态**\n\n"
                f"运行状态: {'✅ 运行中' if enabled else '⏸️ 已暂停'}\n"
                f"当前市场: {market}\n"
                f"扫描间隔: {interval} 秒"
            )

        # 切换市场
        market_switch = {"切换A股": "CN", "切换港股": "HK", "切换美股": "US"}
        if text in market_switch:
            market = market_switch[text]
            self.config.set_market(market)
            labels = {"CN": "A股", "HK": "港股", "US": "美股"}
            return f"✅ 已切换到{labels[market]}市场。"

        # 设置扫描间隔
        match = re.match(r"^扫描频率\s*(\d+)\s*$", text)
        if match:
            interval = int(match.group(1))
            try:
                self.config.set_scan_interval(interval)
                return f"✅ 扫描间隔已设为 {interval} 秒。"
            except ValueError as exc:
                return f"❌ 设置失败: {exc}"

        return None  # 不是系统命令

    # ── 自选股管理 ───────────────────────────────────────

    def _handle_watchlist(self, text: str) -> Optional[str]:
        """处理自选股相关命令"""

        try:
            # 添加自选
            if text.startswith("添加自选"):
                code = text.replace("添加自选", "").strip()
                if not code:
                    return "📝 使用方法: 添加自选 股票代码\n例如: 添加自选 000001"
                try:
                    # 添加成功后通过 coordinator 获取详细分析
                    item = self.watchlist.add_stock(
                        symbol=code, name=code, market="CN"
                    )
                    return (
                        f"✅ 已添加 {code} 到自选股。\n"
                        f'您可以发送 分析 {code} 查看详情。'
                    )
                except WatchlistError as exc:
                    return f"❌ 添加失败: {exc}"

            # 删除自选
            if text.startswith("删除自选"):
                code = text.replace("删除自选", "").strip()
                if not code:
                    return "📝 使用方法: 删除自选 股票代码\n例如: 删除自选 000001"
                if self.watchlist.remove_stock(code):
                    return f"✅ 已从自选股中删除 {code}。"
                return f"❌ 未找到 {code}，请检查股票代码是否正确。"

            # 查看自选
            if text in ("我的自选", "自选股", "查看自选"):
                items = self.watchlist.list_stocks()
                if not items:
                    return '📋 您的自选股列表为空。\n使用[添加自选]指令添加股票，例如: 添加自选 000001'
                lines = [f"📋 共 {len(items)} 只自选股："]
                for i, item in enumerate(items, 1):
                    tag_info = f" [{item.tags}]" if item.tags else ""
                    lines.append(f"{i}. {item.symbol} {item.name}{tag_info}")
                return "\n".join(lines)

            # 清空自选
            if text == "清空自选":
                count = self.watchlist.clear()
                if count > 0:
                    return f"🗑️ 已清空 {count} 只自选股。"
                return "📋 自选股列表已为空。"

        except WatchlistError as exc:
            logger.error("Watchlist operation failed: %s", exc)
            return f"❌ 操作失败: {exc}"

        return None  # 不是自选股命令


# ── 全局单例访问函数 ─────────────────────────────────────

_handler_instance: Optional[MessageHandler] = None


def get_handler() -> MessageHandler:
    """获取 MessageHandler 单例"""
    global _handler_instance  # noqa: PLW0603
    if _handler_instance is None:
        _handler_instance = MessageHandler()
    return _handler_instance
