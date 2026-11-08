"""Run with `python -m app` (reads PORT, default 8000)."""
import logging

import uvicorn

from .config import Settings

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    uvicorn.run("app.main:create_app", factory=True, host="0.0.0.0", port=settings.port, workers=1)
