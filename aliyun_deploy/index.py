import asyncio
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("DATA_SOURCE", "mock")
os.environ.setdefault("DATABASE_PATH", "/tmp/investment.db")

from src.main import app  # noqa: E402
from src.db import init_database  # noqa: E402

# 初始化数据库（FC 下不会触发 FastAPI startup 事件）
init_database()


def handler(event, context):
    """Aliyun FC HTTP 触发器 → FastAPI ASGI 适配器

    Aliyun FC 的 HTTP 触发器传入 JSON 字符串格式的事件，
    需转换为 ASGI scope 后调用 FastAPI 应用，再组装 FC 响应格式返回。
    """
    # ── 解析事件 ───────────────────────────────────────────
    if isinstance(event, str):
        event = json.loads(event)

    body = event.get("body") or ""
    if isinstance(body, bytes):
        body = body.decode("utf-8")

    # 将 headers 转为 ASGI 期望的 (bytes, bytes) 列表
    raw_headers = [
        (k.lower().encode("utf-8"), str(v).encode("utf-8"))
        for k, v in event.get("headers", {}).items()
    ]

    method = event.get("httpMethod") or event.get("method", "GET")
    path = event.get("path", "/")

    query_params = event.get("queryParameters") or event.get("queryParams") or {}
    query_string = "&".join(f"{k}={v}" for k, v in query_params.items())

    # ── 构建 ASGI scope ───────────────────────────────────
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query_string.encode("utf-8"),
        "root_path": "",
        "headers": raw_headers,
        "client": ("0.0.0.0", 0),
        "server": ("bot", 443),
    }

    # ── ASGI send / receive ────────────────────────────────
    async def receive():
        return {"type": "http.request", "body": body.encode("utf-8"), "more_body": False}

    response_body = bytearray()
    response_status = 200
    response_headers: list = []

    async def send(message):
        nonlocal response_status, response_headers, response_body
        if message["type"] == "http.response.start":
            response_status = message["status"]
            response_headers = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    # ── 调用 ASGI 应用 ────────────────────────────────────
    try:
        asyncio.run(app(scope, receive, send))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(app(scope, receive, send))
    except Exception:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {"error": "Internal Server Error", "detail": traceback.format_exc()},
                ensure_ascii=False,
            ),
        }

    # ── 组装 FC 响应 ──────────────────────────────────────
    resp_headers = {}
    for name, value in response_headers:
        resp_headers[name.decode("utf-8")] = value.decode("utf-8")

    return {
        "statusCode": response_status,
        "headers": resp_headers,
        "body": response_body.decode("utf-8"),
    }
