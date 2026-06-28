"""
飞书 Open API 客户端

封装飞书机器人核心 API：
- 自动获取/缓存/刷新 tenant_access_token
- 发送文本消息
- 回复消息
- 发送 Markdown（post）消息
- 健康检查
"""

import json
import logging
import re
import threading
import time
from typing import Optional

import httpx

from src.config.settings import settings
from src.bot.text_utils import sanitize_markdown_for_text

logger = logging.getLogger(__name__)

# ── API 基础地址 ─────────────────────────────────────────

FEISHU_BASE_URL = "https://open.feishu.cn/open-apis"
"""飞书 Open API 基础地址"""

TOKEN_REFRESH_MARGIN = 600
"""Token 过期前 600 秒（10 分钟）刷新"""


class FeishuError(Exception):
    """飞书 API 调用异常"""


class FeishuClient:
    """飞书 Open API 客户端

    封装 token 管理、消息发送、消息回复。

    用法::

        client = FeishuClient()
        client.send_text("ou_xxx", "Hello 飞书")
        client.reply_text("om_xxx", "回复内容")
        client.send_markdown("ou_xxx", "**粗体** 文本")
    """

    _instance: Optional["FeishuClient"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FeishuClient":
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

        self.app_id: str = settings.feishu_app_id
        self.app_secret: str = settings.feishu_app_secret
        self.base_url: str = FEISHU_BASE_URL
        self.max_retries: int = 3
        self.backoff: float = 1.0
        self.timeout: float = 15.0

        # Token 缓存
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._token_lock: threading.Lock = threading.Lock()

        self._initialized = True
        if self.app_id and self.app_secret:
            logger.info("FeishuClient initialized (app_id=%s...)", self.app_id[:8])
        else:
            logger.warning("FeishuClient initialized without credentials")

    # ── Token 管理 ───────────────────────────────────────

    def _ensure_token(self) -> str:
        """获取有效的 tenant_access_token

        若 token 未过期（含 10 分钟缓冲），直接返回缓存。
        否则自动调用飞书 API 刷新。
        """
        with self._token_lock:
            if self._token and time.time() < self._expires_at - TOKEN_REFRESH_MARGIN:
                return self._token

            if not self.app_id or not self.app_secret:
                raise FeishuError("飞书 App ID 或 App Secret 未配置")

            try:
                timeout = httpx.Timeout(10.0, connect=5.0)
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/auth/v3/tenant_access_token/internal",
                        json={
                            "app_id": self.app_id,
                            "app_secret": self.app_secret,
                        },
                    )
                    data = resp.json()
            except httpx.RequestError as exc:
                raise FeishuError(f"获取 token 网络错误: {exc}") from exc

            if data.get("code") != 0:
                raise FeishuError(
                    f"获取 token 失败: code={data.get('code')} msg={data.get('msg')}"
                )

            self._token = data["tenant_access_token"]
            expires_in = data.get("expire", 7200)
            self._expires_at = time.time() + expires_in
            logger.debug("Token refreshed (expires in %ds)", expires_in)
            return self._token

    # ── 消息发送 ─────────────────────────────────────────

    def send_text(
        self,
        receive_id: str,
        content: str,
        receive_id_type: str = "open_id",
    ) -> dict:
        """发送文本消息

        Args:
            receive_id: 接收方 open_id 或 chat_id
            content: 消息文本
            receive_id_type: open_id / chat_id 等飞书接收 ID 类型

        Returns:
            API 响应 JSON
        """
        logger.info("BEFORE_SANITIZE %r", content)
        safe_content = sanitize_markdown_for_text(content)
        logger.info("AFTER_SANITIZE %r", safe_content)
        logger.info("SEND_TO_FEISHU %r", safe_content)
        body = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": json.dumps({"text": safe_content}, ensure_ascii=False),
        }
        logger.info("FeishuClient.send_text content repr: %r", safe_content)
        logger.info("FeishuClient.send_text post body repr: %r", body)
        return self._post(
            f"/im/v1/messages?receive_id_type={receive_id_type}",
            body,
        )

    def reply_text(self, message_id: str, content: str) -> dict:
        """回复消息

        Args:
            message_id: 被回复消息的 message_id
            content: 回复文本

        Returns:
            API 响应 JSON
        """
        logger.info("BEFORE_SANITIZE %r", content)
        safe_content = sanitize_markdown_for_text(content)
        logger.info("AFTER_SANITIZE %r", safe_content)
        logger.info("SEND_TO_FEISHU %r", safe_content)
        body = {
            "msg_type": "text",
            "content": json.dumps({"text": safe_content}, ensure_ascii=False),
        }
        logger.info("FeishuClient.reply_text content repr: %r", safe_content)
        logger.info("FeishuClient.reply_text post body repr: %r", body)
        return self._post(f"/im/v1/messages/{message_id}/reply", body)

    def send_markdown(self, receive_id: str, markdown: str) -> dict:
        """发送 Markdown 格式消息（使用 post 消息类型）

        Args:
            receive_id: 接收方 open_id 或 chat_id
            markdown: Markdown 文本（支持 **粗体** [链接](url)）

        Returns:
            API 响应 JSON
        """
        post_content = self._md_to_post_content(markdown)
        body = {
            "receive_id": receive_id,
            "msg_type": "post",
            "content": json.dumps(post_content, ensure_ascii=False),
        }
        return self._post("/im/v1/messages?receive_id_type=open_id", body)

    def health_check(self) -> bool:
        """检查飞书 API 连通性

        通过尝试获取 token 验证配置和网络是否正常。
        不实际发送消息，不消耗配额。

        Returns:
            True 表示 API 可用
        """
        try:
            token = self._ensure_token()
            return bool(token)
        except FeishuError as exc:
            logger.warning("Health check failed: %s", exc)
            return False

    # ── 内部 HTTP 调用 ──────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        """发送 POST 请求（含 token 注入、重试、错误处理）"""
        token = self._ensure_token()
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        timeout = httpx.Timeout(self.timeout, connect=10.0)

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, headers=headers, json=body)
                    data = resp.json()

                if data.get("code") != 0:
                    raise FeishuError(
                        f"API code={data.get('code')} msg={data.get('msg')}"
                    )

                return data

            except FeishuError:
                raise  # API 层面的错误不重试
            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "Timeout (attempt %d/%d): %s", attempt + 1, self.max_retries, path
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "Network error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, exc,
                )

            if attempt < self.max_retries - 1:
                wait = self.backoff * (2 ** attempt)
                logger.info("Retrying in %.1fs...", wait)
                time.sleep(wait)

        raise FeishuError(f"请求失败（重试 {self.max_retries} 次）: {last_exc}")

    # ── Markdown 转换 ────────────────────────────────────

    def _md_to_post_content(self, md_text: str) -> dict:
        """将 Markdown 文本转换为飞书 post 消息格式

        支持：
        - **粗体**
        - [链接文字](url)
        - 普通文本
        - 空行分段
        """
        lines = md_text.strip().split("\n")
        content_lines: list[list[dict]] = []

        for line in lines:
            if not line.strip():
                content_lines.append([{"tag": "text", "text": ""}])
                continue
            tags = self._parse_md_line(line)
            content_lines.append(tags)

        # 从首行提取标题
        first_line = md_text.strip().split("\n")[0] if md_text.strip() else ""
        title = (
            first_line.replace("##", "").replace("#", "").replace("**", "").strip()[:50]
        )

        return {
            "zh_cn": {
                "title": title or "投资助手",
                "content": content_lines,
            }
        }

    @staticmethod
    def _parse_md_line(line: str) -> list[dict]:
        """解析单行 Markdown，返回飞书 post tag 列表"""
        tags: list[dict] = []
        # 匹配 **bold** 或 [text](url)
        pattern = r"(\*\*(.+?)\*\*)|(\[(.+?)\]\((.+?)\))"
        last_end = 0

        for match in re.finditer(pattern, line):
            start = match.start()

            # 匹配前的纯文本
            if start > last_end:
                text = line[last_end:start]
                if text:
                    tags.append({"tag": "text", "text": text})

            if match.group(1):  # **bold**
                tags.append({"tag": "bold", "text": match.group(2)})
            elif match.group(3):  # [text](url)
                tags.append(
                    {"tag": "a", "text": match.group(4), "href": match.group(5)}
                )

            last_end = match.end()

        # 匹配后的剩余文本
        if last_end < len(line):
            remaining = line[last_end:]
            if remaining:
                tags.append({"tag": "text", "text": remaining})

        return tags if tags else [{"tag": "text", "text": line}]


# ── 全局单例访问函数 ─────────────────────────────────────

_feishu_instance: Optional[FeishuClient] = None


def get_feishu_client() -> FeishuClient:
    """获取 FeishuClient 单例"""
    global _feishu_instance  # noqa: PLW0603
    if _feishu_instance is None:
        _feishu_instance = FeishuClient()
    return _feishu_instance
