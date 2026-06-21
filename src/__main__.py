"""Package entrypoint for `python -m src`."""

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
