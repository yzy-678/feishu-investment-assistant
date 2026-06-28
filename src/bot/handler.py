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
import threading
import time
from typing import Optional

from src.ai.deepseek import DeepSeekError
from src.agents.alert_agent import AlertAgent, get_alert_agent
from src.bot.client import FeishuClient, FeishuError, get_feishu_client
from src.bot.text_utils import sanitize_markdown_for_text, sanitize_text
from src.config.manager import ConfigManager, get_config
from src.config.settings import settings
from src.watchlist.manager import WatchlistError, WatchlistManager, get_watchlist
from src.agents.base import BaseAgent
from src.agents.coordinator import AgentCoordinator, get_coordinator
from src.agents.general_agent import get_general_agent
from src.agents.market_agent import get_market_agent
from src.agents.news_agent import get_news_agent
from src.agents.report_agent import get_report_agent

logger = logging.getLogger(__name__)

MAX_REPLY_LENGTH = 1500
"""飞书单条消息最大字符数"""

MESSAGE_DEDUP_TTL_SECONDS = 600
"""同一 message_id 的去重保留时间。"""

MESSAGE_DEDUP_MAX_SIZE = 2048
"""内存中最多保留的 message_id 数量。"""


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
        self.alert_agent: AlertAgent = get_alert_agent()
        self._dedup_lock = threading.Lock()
        self._inflight_message_ids: set[str] = set()
        self._processed_message_ids: dict[str, float] = {}
        self._register_agents()
        logger.info("MessageHandler initialized")

    # ── Agent 注册 ────────────────────────────────────────

    def _register_agents(self) -> None:
        """注册所有 Agent 到 Coordinator

        确保各 Agent 的 can_handle() 能被 Coordinator 依次检查，
        匹配到第一个能处理的消息后交由对应 Agent 处理。
        """
        agents: list[BaseAgent] = [
            get_market_agent(),
            get_report_agent(),
            self.alert_agent,
            get_news_agent(),
            get_general_agent(),
        ]

        for agent in agents:
            self.coordinator.register(agent)

        logger.info(
            "Registered %d agents: %s",
            len(agents),
            [a.__class__.__name__ for a in agents],
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
        message_id = ""
        claimed = False
        success = False

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

            message_id = message.get("message_id", "")
            chat_id = message.get("chat_id", "")
            content_str: str = message.get("content", "{}")
            open_id: str = sender.get("open_id", "")
            logger.info(
                "Feishu handler raw event payload: %s",
                json.dumps(raw_event, ensure_ascii=False, default=str),
            )
            logger.info(
                "Feishu event.message.content raw: %s",
                content_str,
            )

            if not message_id or not open_id:
                logger.warning("Missing message_id or open_id in event")
                return None

            if not self._claim_message(message_id):
                logger.info("Duplicate Feishu message ignored: %s", message_id)
                return None

            claimed = True

            # 解析消息内容
            try:
                content_data = json.loads(content_str)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to parse message content: %s", exc)
                success = True
                return None

            logger.info(
                "Feishu json.loads(content) parsed: %s",
                json.dumps(content_data, ensure_ascii=False, default=str),
            )
            text = sanitize_text(self._extract_message_text(message, content_data))
            logger.info(
                "Feishu final message_text for coordinator: %s",
                text,
            )

            if not text:
                reply = "消息不能为空，请重新输入。"
                self._send_reply(
                    message_id=message_id,
                    chat_id=chat_id,
                    open_id=open_id,
                    content=reply,
                )
                success = True
                return reply

            # 处理消息
            reply = self.process_message(open_id, message_id, text)

            # 发送回复
            if reply:
                raw_reply = reply[:MAX_REPLY_LENGTH]
                logger.info("BEFORE_SANITIZE %r", raw_reply)
                truncated = sanitize_markdown_for_text(raw_reply)
                logger.info("AFTER_SANITIZE %r", truncated)
                logger.info("SEND_TO_FEISHU %r", truncated)
                logger.info(
                    "Final reply to user data: %s",
                    json.dumps(
                        {
                            "message_id": message_id,
                            "open_id": open_id[:8],
                            "max_reply_length": MAX_REPLY_LENGTH,
                            "original_length": len(reply),
                            "sent_length": len(truncated),
                            "final_response": truncated,
                        },
                        ensure_ascii=False,
                    ),
                )
                self._send_reply(
                    message_id=message_id,
                    chat_id=chat_id,
                    open_id=open_id,
                    content=truncated,
                )
                logger.info(
                    "Replied to %s: %.40s... (%d chars)",
                    open_id[:8], text, len(truncated),
                )
                success = True
                return truncated

            success = True
            return None

        except (FeishuError, DeepSeekError) as exc:
            logger.error("Handler error: %s", exc)
            return None
        finally:
            if claimed and message_id:
                self._finalize_message(message_id, success)

    def _send_reply(
        self,
        message_id: str,
        chat_id: str,
        open_id: str,
        content: str,
    ) -> None:
        """发送飞书回复。

        默认不使用 message reply，避免飞书客户端引用预览乱码。
        有 chat_id 时发送普通群消息；缺失 chat_id 时回退到原 reply_text。
        """
        logger.info("SEND_TO_FEISHU %r", content)
        if settings.use_reply_message:
            self.feishu.reply_text(message_id, content)
            return

        if chat_id:
            self.feishu.send_text(chat_id, content, receive_id_type="chat_id")
            return

        logger.warning(
            "Feishu chat_id missing, fallback to reply_text: message_id=%s open_id=%s",
            message_id,
            open_id[:8],
        )
        self.feishu.reply_text(message_id, content)

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

    @classmethod
    def _extract_message_text(cls, message: dict, content_data: object) -> str:
        """从飞书 message.content 中提取最终用户文本。

        飞书文本消息的 content 是 JSON 字符串，常见结构为
        {"text": "..."}。群聊 @ 机器人时 text 中可能包含 <at ...>...</at>
        或 mentions[*].key。这里只移除 @ 标签/占位符，不改写中文正文。
        """
        if not isinstance(content_data, dict):
            return ""

        raw_text = content_data.get("text", "")
        if raw_text is None:
            return ""
        text = str(raw_text)

        mentions = message.get("mentions", [])
        if not isinstance(mentions, list):
            mentions = []

        text = cls._remove_feishu_mentions(text, mentions)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _remove_feishu_mentions(text: str, mentions: list[dict]) -> str:
        """只删除飞书 @ 标签/占位符，保留其余中文。"""
        cleaned = re.sub(r"<at\b[^>]*>.*?</at>", " ", text, flags=re.I | re.S)

        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            candidates = [
                mention.get("key"),
                mention.get("name"),
                mention.get("id", {}).get("open_id")
                if isinstance(mention.get("id"), dict)
                else None,
            ]
            for candidate in candidates:
                if not candidate:
                    continue
                value = str(candidate)
                cleaned = cleaned.replace(value, " ")
                if not value.startswith("@"):
                    cleaned = cleaned.replace(f"@{value}", " ")

        return cleaned

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

        if text in ("立即扫描", "执行扫描", "扫描预警", "盘中扫描"):
            return self._handle_manual_scan()

        return None  # 不是系统命令

    def _handle_manual_scan(self) -> str:
        """手动触发一次盘中预警扫描。"""
        result = self.alert_agent.scan_watchlist()
        alerts = result.get("alerts", [])
        deliverable = [item for item in alerts if item.get("should_send")]

        lines = [
            "📡 盘中扫描完成",
            f"数据源: {result['data_source']}",
            f"扫描标的: {result['scanned']}",
            f"触发预警: {result['triggered']}",
            f"可推送预警: {result['deliverable']}",
        ]

        if deliverable:
            for item in deliverable[:5]:
                event = item["event"]
                lines.append(
                    f"- {event.related_code or event.title}: {event.title}（强度 {event.strength:.1f}）"
                )
            self.alert_agent.mark_delivered(
                [item["event"].event_id for item in deliverable]
            )
        else:
            lines.append(result["message"])

        return "\n".join(lines)

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

    # ── 消息去重 ─────────────────────────────────────────

    def _claim_message(self, message_id: str) -> bool:
        """登记待处理 message_id，防止飞书重试导致重复回复。"""
        now = time.time()

        with self._dedup_lock:
            self._cleanup_processed_messages(now)

            if (
                message_id in self._inflight_message_ids
                or message_id in self._processed_message_ids
            ):
                return False

            self._inflight_message_ids.add(message_id)
            return True

    def _finalize_message(self, message_id: str, success: bool) -> None:
        """处理完成后更新去重状态。"""
        with self._dedup_lock:
            self._inflight_message_ids.discard(message_id)

            if not success:
                self._processed_message_ids.pop(message_id, None)
                return

            self._processed_message_ids[message_id] = time.time()
            while len(self._processed_message_ids) > MESSAGE_DEDUP_MAX_SIZE:
                oldest_message_id = next(iter(self._processed_message_ids))
                self._processed_message_ids.pop(oldest_message_id, None)

    def _cleanup_processed_messages(self, now: float) -> None:
        """清理过期 message_id，避免内存无限增长。"""
        expired_ids = [
            message_id
            for message_id, seen_at in self._processed_message_ids.items()
            if now - seen_at > MESSAGE_DEDUP_TTL_SECONDS
        ]
        for message_id in expired_ids:
            self._processed_message_ids.pop(message_id, None)


# ── 全局单例访问函数 ─────────────────────────────────────

_handler_instance: Optional[MessageHandler] = None


def get_handler() -> MessageHandler:
    """获取 MessageHandler 单例"""
    global _handler_instance  # noqa: PLW0603
    if _handler_instance is None:
        _handler_instance = MessageHandler()
    return _handler_instance
