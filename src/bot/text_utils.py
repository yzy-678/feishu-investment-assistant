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


def sanitize_markdown_for_text(text: object) -> str:
    """把 Markdown 文本降级为飞书纯文本，不改写中文正文。"""
    value = sanitize_text(text)
    if not value:
        return ""

    lines: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.replace("```", "")

        stripped = line.lstrip(" ")
        leading_spaces = len(line) - len(stripped)
        while stripped.startswith("#"):
            stripped = stripped[1:]
        if stripped != line.lstrip(" "):
            stripped = stripped.lstrip(" ")
            line = (" " * leading_spaces) + stripped

        line = line.replace("**", "")
        line = _collapse_ascii_spaces(line)
        lines.append(line.rstrip(" "))

    return "\n".join(lines).strip()


def _collapse_ascii_spaces(text: str) -> str:
    collapsed: list[str] = []
    previous_space = False
    for char in text:
        if char == " ":
            if not previous_space:
                collapsed.append(char)
            previous_space = True
            continue

        previous_space = False
        collapsed.append(char)
    return "".join(collapsed)
