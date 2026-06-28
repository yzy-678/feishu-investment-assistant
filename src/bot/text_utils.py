"""飞书发送前的文本安全清洗。"""

from __future__ import annotations

import unicodedata


def sanitize_text(text: object) -> str:
    """清洗不可见控制字符和异常 surrogate，保留正常中文/英文/数字/标点。"""
    if text is None:
        return ""

    value = str(text)
    value = value.encode("utf-8", "surrogatepass").decode("utf-8", "replace")

    cleaned: list[str] = []
    for char in value:
        if char in ("\n", "\r", "\t"):
            cleaned.append(char)
            continue

        category = unicodedata.category(char)
        if category in {"Cc", "Cf", "Cs"}:
            continue
        cleaned.append(char)

    return "".join(cleaned)
