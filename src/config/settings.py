"""
环境变量配置管理

使用 Pydantic v2 BaseSettings 读取 .env 文件，
所有配置项均可通过环境变量覆盖。
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── DeepSeek ──────────────────────────────────────────────
    deepseek_api_key: str = ""
    """DeepSeek API 密钥（必填）"""

    deepseek_base_url: str = "https://api.deepseek.com"
    """DeepSeek API 地址"""

    deepseek_model: str = "deepseek-chat"
    """模型名称"""

    # ── 飞书应用 ──────────────────────────────────────────────
    feishu_app_id: str = ""
    """飞书应用 App ID（必填，如需飞书功能）"""

    feishu_app_secret: str = ""
    """飞书应用 App Secret（必填，如需飞书功能）"""

    feishu_bot_name: str = "投资助手"
    """机器人名称"""

    feishu_event_verify_token: str = ""
    """飞书事件回调验证令牌（订阅事件回调时必填）"""

    feishu_event_encrypt_key: str = ""
    """飞书事件回调加密 Key（可选，启用加密时必填）"""

    # ── 管理员 ────────────────────────────────────────────────
    admin_user_open_id: str = ""
    """管理员飞书 open_id（权限控制用）"""

    # ── API 安全 ──────────────────────────────────────────────
    api_bearer_token: str = ""
    """GitHub Actions 调用内部 API 的鉴权 Token"""

    # ── 应用运行时 ────────────────────────────────────────────
    log_level: str = "INFO"
    """日志级别：DEBUG / INFO / WARNING / ERROR"""

    database_path: str = "config/investment.db"
    """SQLite 数据库文件路径（相对于项目根目录）"""

    data_source: str = "eastmoney"
    """行情数据源：eastmoney / mock / yahoo"""

    model_config = {
        "env_file": str(Path(__file__).resolve().parent.parent.parent / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# 全局单例
settings = Settings()
