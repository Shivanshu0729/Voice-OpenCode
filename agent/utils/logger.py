import logging
import sys

import structlog
from rich.logging import RichHandler

from config.settings import settings


def _configure_stdlib() -> None:
    level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    for lib in ("httpx", "httpcore", "websockets", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def configure_logging() -> None:
    _configure_stdlib()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level, logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)