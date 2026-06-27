"""
新闻催化 Agent

基于东方财富新闻/公告搜索，分析个股或板块的新闻催化与事件驱动。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html import unescape
from typing import Any, Optional

import httpx

from src.agents.base import BaseAgent, AgentResponse, AgentType
from src.ai.deepseek import DeepSeekClient, DeepSeekError, get_deepseek
from src.ai.prompts import INVESTMENT_ASSISTANT_SYSTEM_PROMPT
from src.market import get_market_data_service

logger = logging.getLogger(__name__)

EASTMONEY_SEARCH_URL = "https://search-api-web.eastmoney.com/search/jsonp"
DEFAULT_TIMEOUT = 8.0

NEWS_INTENT_KEYWORDS: tuple[str, ...] = (
    "新闻",
    "消息",
    "公告",
    "催化",
    "事件",
    "驱动",
    "为什么涨",
    "为何涨",
    "为啥涨",
    "最近有什么",
    "今天有什么",
)

OUTPUT_FORMAT_RULES = """
请严格按以下格式输出，不要新增其它一级标题：
【核心事件】
【事件解读】
【对板块影响】
【对个股影响】
【持续性判断】

硬规则：
- 只能基于【东方财富新闻/公告】里列出的内容分析。
- 不允许编造新闻、公告、日期、公司动作或政策信息。
- 如果证据不足，请明确写“未获取到相关新闻”或“现有新闻不足以判断”。
""".strip()


@dataclass(frozen=True)
class EastMoneyNewsItem:
    """东方财富新闻/公告搜索结果。"""

    title: str
    content: str
    date: str
    source: str
    url: str
    item_type: str


class NewsAgent(BaseAgent):
    """新闻催化与事件驱动分析 Agent。"""

    _instance: Optional["NewsAgent"] = None

    def __new__(cls) -> "NewsAgent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # type: ignore[attr-defined]
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self.deepseek: DeepSeekClient = get_deepseek()
        self.market_data = get_market_data_service()
        self.timeout = DEFAULT_TIMEOUT
        self._initialized = True
        logger.info("NewsAgent initialized")

    @property
    def agent_type(self) -> AgentType:
        return AgentType.NEWS

    def can_handle(self, message: str) -> bool:
        if not message or not message.strip():
            return False
        return is_news_intent(message)

    def handle(self, session_id: str, message: str) -> AgentResponse:
        try:
            keyword = self._extract_keyword(message)
            items = self.search_eastmoney(keyword)
            logger.info(
                "NewsAgent searched: session=%s keyword=%s news_count=%d",
                session_id[:8],
                keyword,
                len(items),
            )

            if not items:
                return AgentResponse(
                    success=True,
                    agent=AgentType.NEWS,
                    message=self._no_news_message(keyword),
                    metadata={
                        "session_id": session_id,
                        "keyword": keyword,
                        "news_count": 0,
                    },
                )

            prompt = self._build_prompt(message, keyword, items)
            reply = self.deepseek.chat([
                {
                    "role": "system",
                    "content": INVESTMENT_ASSISTANT_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ])
            reply = self._ensure_output_format(reply)
            return AgentResponse(
                success=True,
                agent=AgentType.NEWS,
                message=reply,
                metadata={
                    "session_id": session_id,
                    "keyword": keyword,
                    "news_count": len(items),
                },
            )

        except DeepSeekError as exc:
            logger.warning("NewsAgent DeepSeek error: %s", exc)
            return AgentResponse(
                success=False,
                agent=AgentType.NEWS,
                message="分析新闻时遇到问题，请稍后再试。",
                metadata={
                    "session_id": session_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
        except Exception as exc:
            logger.warning("NewsAgent error: %s", exc)
            return AgentResponse(
                success=True,
                agent=AgentType.NEWS,
                message="未获取到相关新闻",
                metadata={
                    "session_id": session_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

    def search_eastmoney(
        self,
        keyword: str,
        page_size: int = 5,
    ) -> list[EastMoneyNewsItem]:
        """搜索东方财富新闻和公告。"""
        news_items = self._search_eastmoney_type(
            keyword=keyword,
            type_name="cmsArticleWebOld",
            item_type="新闻",
            page_size=page_size,
        )
        notice_items = self._search_eastmoney_type(
            keyword=keyword,
            type_name="noticeWeb",
            item_type="公告",
            page_size=page_size,
        )
        return self._deduplicate_items(news_items + notice_items)

    def _search_eastmoney_type(
        self,
        keyword: str,
        type_name: str,
        item_type: str,
        page_size: int,
    ) -> list[EastMoneyNewsItem]:
        param = self._build_search_param(keyword, type_name, page_size)
        headers = {
            "Referer": "https://so.eastmoney.com/",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
        }
        try:
            with httpx.Client(timeout=self.timeout, headers=headers) as client:
                response = client.get(
                    EASTMONEY_SEARCH_URL,
                    params={
                        "cb": "jQuery",
                        "param": json.dumps(
                            param,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "EastMoney news search failed: keyword=%s type=%s error=%s",
                keyword,
                type_name,
                exc,
            )
            return []

        payload = self._parse_jsonp(response.text)
        rows = payload.get("result", {}).get(type_name, [])
        if not isinstance(rows, list):
            return []

        return [
            self._parse_item(row, item_type)
            for row in rows
            if isinstance(row, dict) and self._item_has_title(row)
        ]

    @staticmethod
    def _build_search_param(
        keyword: str,
        type_name: str,
        page_size: int,
    ) -> dict[str, Any]:
        type_params: dict[str, Any] = {
            "pageIndex": 1,
            "pageSize": page_size,
            "preTag": "",
            "postTag": "",
        }
        if type_name == "cmsArticleWebOld":
            type_params.update({"searchScope": "ALL", "sort": "time"})

        return {
            "uid": "",
            "keyword": keyword,
            "type": [type_name],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {type_name: type_params},
        }

    @staticmethod
    def _parse_jsonp(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            return {}
        match = re.match(r"^[^(]*\((.*)\)\s*;?$", stripped, re.S)
        raw_json = match.group(1) if match else stripped
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _parse_item(row: dict[str, Any], item_type: str) -> EastMoneyNewsItem:
        return EastMoneyNewsItem(
            title=clean_text(str(row.get("title", ""))),
            content=clean_text(str(row.get("content", ""))),
            date=clean_text(str(row.get("date", ""))),
            source=clean_text(
                str(row.get("mediaName") or row.get("securityFullName") or "东方财富")
            ),
            url=clean_text(str(row.get("url", ""))),
            item_type=item_type,
        )

    @staticmethod
    def _item_has_title(row: dict[str, Any]) -> bool:
        return bool(clean_text(str(row.get("title", ""))))

    @staticmethod
    def _deduplicate_items(
        items: list[EastMoneyNewsItem],
    ) -> list[EastMoneyNewsItem]:
        seen: set[tuple[str, str]] = set()
        result: list[EastMoneyNewsItem] = []
        for item in items:
            key = (item.title, item.url)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _extract_keyword(self, message: str) -> str:
        cleaned = message.strip()
        symbol = self.market_data.extract_symbol(cleaned)
        if symbol:
            return symbol

        keyword = re.sub(
            r"(今天|最近|近期|有什么|有啥|哪些|什么|新闻|消息|公告|催化|事件|驱动|为什么涨|为何涨|为啥涨|板块)",
            " ",
            cleaned,
        )
        keyword = re.sub(r"[？?！!。,.，、]", " ", keyword)
        keyword = re.sub(r"\s+", " ", keyword).strip()
        return keyword or cleaned

    @staticmethod
    def _build_prompt(
        original_message: str,
        keyword: str,
        items: list[EastMoneyNewsItem],
    ) -> str:
        lines = [
            f"用户问题：{original_message}",
            f"搜索关键词：{keyword}",
            "",
            "【东方财富新闻/公告】",
        ]
        for index, item in enumerate(items, start=1):
            lines.extend([
                f"{index}. [{item.item_type}] {item.title}",
                f"   日期：{item.date or '未提供'}",
                f"   来源：{item.source or '东方财富'}",
                f"   摘要：{item.content or '未提供'}",
                f"   链接：{item.url or '未提供'}",
            ])

        lines.extend(["", OUTPUT_FORMAT_RULES])
        return "\n".join(lines)

    @staticmethod
    def _ensure_output_format(reply: str) -> str:
        text = str(reply or "").strip()
        if not text:
            return NewsAgent._no_news_message("")

        required_headings = (
            "【核心事件】",
            "【事件解读】",
            "【对板块影响】",
            "【对个股影响】",
            "【持续性判断】",
        )
        if all(heading in text for heading in required_headings):
            return text

        return "\n".join(
            [
                "【核心事件】",
                text,
                "【事件解读】",
                "以上内容仅基于东方财富新闻/公告搜索结果。",
                "【对板块影响】",
                "现有新闻不足以判断。",
                "【对个股影响】",
                "现有新闻不足以判断。",
                "【持续性判断】",
                "现有新闻不足以判断。",
            ]
        )

    @staticmethod
    def _no_news_message(keyword: str) -> str:
        suffix = f"：{keyword}" if keyword else ""
        return "\n".join([
            "【核心事件】",
            f"未获取到相关新闻{suffix}",
            "【事件解读】",
            "未获取到相关新闻，不能编造事件解读。",
            "【对板块影响】",
            "未获取到相关新闻。",
            "【对个股影响】",
            "未获取到相关新闻。",
            "【持续性判断】",
            "未获取到相关新闻。",
        ])


def clean_text(text: str) -> str:
    """清理东方财富搜索结果里的 HTML 标签和高亮标记。"""
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_news_intent(message: str) -> bool:
    """判断是否为新闻/催化/事件驱动类问题。"""
    if not message or not message.strip():
        return False
    return any(keyword in message for keyword in NEWS_INTENT_KEYWORDS)


_news_agent_instance: Optional[NewsAgent] = None


def get_news_agent() -> NewsAgent:
    """获取 NewsAgent 单例。"""
    global _news_agent_instance  # noqa: PLW0603
    if _news_agent_instance is None:
        _news_agent_instance = NewsAgent()
    return _news_agent_instance
