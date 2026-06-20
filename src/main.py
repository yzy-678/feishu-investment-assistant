"""
飞书AI投资助手 — FastAPI 应用入口

启动方式:
    uvicorn src.main:app --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI

from src.bot.router import router
from src.config.settings import settings

# ── 日志配置 ─────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用"""
    app = FastAPI(
        title="飞书AI投资助手",
        description="飞书机器人驱动的 AI 投资研究系统",
        version="1.0.0",
    )

    app.include_router(router)

    @app.on_event("startup")
    async def startup() -> None:
        """应用启动时初始化"""
        from src.db import init_database

        init_database()
        logger.info("Database initialized")

        # 后台调度器（后续启用）
        # from src.scheduler.background import start_scheduler
        # start_scheduler()

        logger.info("Application started")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        """应用关闭时清理"""
        from src.db import close_database

        close_database()
        logger.info("Application stopped")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
