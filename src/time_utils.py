"""
时间工具

统一使用 Asia/Shanghai 作为面向用户的业务时区。
"""

from datetime import date, datetime
from zoneinfo import ZoneInfo

SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def shanghai_now() -> datetime:
    """返回上海时区当前时间（aware datetime）。"""
    return datetime.now(SHANGHAI_TZ)


def shanghai_now_naive() -> datetime:
    """返回上海时区当前时间的 naive 版本，便于写入 SQLite。"""
    return shanghai_now().replace(tzinfo=None)


def shanghai_today() -> date:
    """返回上海时区今日日期。"""
    return shanghai_now().date()
