"""
DeepSeek API 客户端

使用 OpenAI 兼容接口调用 DeepSeek，集成 ConversationMemory。
支持重试、超时、健康检查、Token 估算。
"""

import logging
import random
import time
import re
from typing import Optional

import httpx

from src.config.settings import settings
from src.memory import get_memory, MemoryError

logger = logging.getLogger(__name__)


class DeepSeekError(Exception):
    """DeepSeek API 调用异常"""


class DeepSeekClient:
    """DeepSeek API 统一封装

    用法::

        client = DeepSeekClient()
        reply = client.chat([{"role": "user", "content": "今天市场怎么样"}])
        reply = client.chat_with_memory("user_open_id", "分析平安银行")
        ok = client.health_check()
    """

    _instance: Optional["DeepSeekClient"] = None

    def __new__(cls) -> "DeepSeekClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized: bool = False  # type: ignore[assignment]
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.api_key: str = settings.deepseek_api_key
        self.base_url: str = settings.deepseek_base_url.rstrip("/")
        self.model: str = settings.deepseek_model

        self.max_retries: int = 3
        """最大重试次数（遇到 5xx 或网络错误时）"""
        self.backoff: float = 1.0
        """指数退避初始等待秒数"""
        self.timeout: float = 60.0
        """HTTP 请求超时秒数（含连接、读写）"""

        self._memory = get_memory()
        self._initialized = True
        logger.info("DeepSeekClient ready: model=%s, base=%s", self.model, self.base_url)

    # ── 对外接口 ─────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """发送对话请求

        Args:
            messages: 消息列表，格式 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数 (0.0 ~ 2.0)
            max_tokens: 最大生成 token 数（None 表示使用模型默认值）

        Returns:
            AI 回复文本

        Raises:
            DeepSeekError: API 调用失败时抛出
        """
        if not messages:
            raise DeepSeekError("消息列表不能为空")

        return self._call_with_retry(messages, temperature, max_tokens)

    def chat_with_memory(self, session_id: str, user_message: str) -> str:
        """结合对话记忆的智能回复

        自动完成：
        1. 保存用户消息到记忆
        2. 提取历史上下文
        3. 调用 AI 获取回复
        4. 保存回复到记忆
        5. 检查是否需要触发摘要

        Args:
            session_id: 会话 ID（用户 open_id）
            user_message: 用户最新消息

        Returns:
            AI 回复文本

        Raises:
            DeepSeekError: 调用失败时抛出
        """
        try:
            # 1. 保存用户消息
            self._memory.add_message(session_id, "user", user_message)

            # 2. 获取带摘要的上下文
            context = self._memory.get_context(session_id)

            # 3. 调用 AI
            response = self.chat(context)

            # 4. 保存回复
            self._memory.add_message(session_id, "assistant", response)

            # 5. 检查摘要
            self._memory.summarize_if_needed(session_id)

            logger.info("chat_with_memory: session=%s, user=%d chars, reply=%d chars",
                        session_id, len(user_message), len(response))
            return response

        except MemoryError as exc:
            raise DeepSeekError(f"记忆模块错误: {exc}") from exc

    def health_check(self) -> bool:
        """检查 API 连通性

        调用 /v1/models 接口验证 API Key 和网络可达性。
        不消耗 Token 额度。
        """
        try:
            timeout = httpx.Timeout(10.0, connect=5.0)
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code == 200:
                    logger.info("Health check passed")
                    return True
                logger.warning("Health check failed: HTTP %d", resp.status_code)
                return False
        except httpx.RequestError as exc:
            logger.warning("Health check network error: %s", exc)
            return False

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的 Token 数量

        使用字符级启发式估算：
        - 中文字符 ≈ 2 tokens
        - 其他字符 ≈ 0.3 tokens

        Args:
            text: 输入文本

        Returns:
            估算的 Token 数（至少返回 1）
        """
        if not text:
            return 0

        chinese_chars = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", text))
        other_chars = len(text) - chinese_chars

        tokens = chinese_chars * 2.0 + other_chars * 0.3
        return max(1, int(tokens))

    # ── 内部 HTTP 调用 ─────────────────────────────────

    def _call_with_retry(self, messages: list[dict], temperature: float,
                         max_tokens: Optional[int]) -> str:
        """带重试机制的 API 调用

        重试策略：
        - 4xx 客户端错误 → 不重试，立即抛出
        - 5xx 服务端错误 → 指数退避重试
        - 网络错误 → 指数退避重试
        - 重试耗尽 → 抛出最后一次异常
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._call_single(messages, temperature, max_tokens)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code < 500:
                    # 4xx 错误不重试
                    raise DeepSeekError(
                        f"API {exc.response.status_code}: {exc.response.text[:200]}"
                    ) from exc
                logger.warning(
                    "Server error (attempt %d/%d): HTTP %d",
                    attempt + 1, self.max_retries + 1, exc.response.status_code,
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "Network error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
            except Exception as exc:
                raise DeepSeekError(f"未预期的错误: {exc}") from exc

            if attempt < self.max_retries:
                wait = self.backoff * (2 ** attempt) + random.uniform(0, 0.5)
                logger.info("Retrying in %.1fs...", wait)
                time.sleep(wait)

        raise DeepSeekError(f"重试耗尽: {last_exc}") from last_exc

    def _call_single(self, messages: list[dict], temperature: float,
                     max_tokens: Optional[int]) -> str:
        """单次 HTTP 调用"""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        timeout = httpx.Timeout(self.timeout, connect=10.0, write=30.0, read=self.timeout)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            content: str = data["choices"][0]["message"]["content"]
            logger.debug("API call success: %d msg, %d tokens in, %d chars out",
                         len(messages),
                         sum(self.estimate_tokens(m.get("content", "")) for m in messages),
                         len(content))
            return content
        except (KeyError, IndexError) as exc:
            raise DeepSeekError(f"API 返回格式异常: {exc}") from exc


# ── 全局单例访问函数 ─────────────────────────────────────

_deepseek_instance: Optional[DeepSeekClient] = None


def get_deepseek() -> DeepSeekClient:
    """获取 DeepSeekClient 单例"""
    global _deepseek_instance  # noqa: PLW0603
    if _deepseek_instance is None:
        _deepseek_instance = DeepSeekClient()
    return _deepseek_instance
