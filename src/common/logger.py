"""Centralized logging configuration for media extraction pipeline.
This logger can be used across all projects (root, ocr, transcriber).
"""

import logging
import os
import sys
import asyncio
from logging import Logger, FileHandler
from datetime import datetime
from pathlib import Path
from rich.logging import RichHandler

from dotenv import load_dotenv
load_dotenv() # loading .env if available

# Icon mapping for different components
ICON_MAP = {
    "media_preprocessor": "⚙️",
    "media_reporter": "📊",
    "media_vectordb": "🛡️",
    "ocr": "📄",
    "transcriber": "🎙️",
    "pipeline": "⚙️",
    "default": "🤖",
}


class MemoryHandler(logging.Handler):
    """A thread-safe logging handler that keeps the last N log records in memory."""
    def __init__(self, capacity: int = 100):
        super().__init__()
        self.capacity = capacity
        self.buffer: list[str] = []
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

    def emit(self, record):
        try:
            msg = self.format(record)
            self.buffer.append(msg)
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)
        except Exception:
            self.handleError(record)

    def get_logs(self) -> list[str]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()

# Global log buffer for UI/API display
log_buffer = MemoryHandler()
log_buffer.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))

def get_logger(
    name: str = "default",
    level: int = logging.INFO,
    icon: str = "🤖",
    to_file: str = None
) -> Logger:
    """Get a configured logger.
    
    If 'log_to_file' is enabled in config, writes to a file in 'logs/'.
    Otherwise, uses RichHandler for console output.
    
    Args:
        name: Name of the component/module
        level: Logging level (default: INFO)
        icon: Emoji or string to prefix log lines
        to_file: Optional path to log file. If None and file logging is enabled, a new file is generated in 'logs/'.
    Returns:
        Configured logger instance
    """
    # Check for global debug flag
    if os.getenv("DEBUG", "false").lower() in ("true", "1"):
        level = logging.DEBUG

    if os.getenv("LOG__TO_FILE", "false").lower() in ("true", "1"):
        log_to_file = True
    else:
        log_to_file = False

    if not icon:
        icon = ICON_MAP.get(name, ICON_MAP["default"])

    logger = logging.getLogger(f"MediaExtraction.{name}")

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        log_dir_path = "logs"

        if log_to_file:
            # File Logging Strategy
            try:
                if to_file is None:
                    log_dir = Path(log_dir_path)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    to_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                file_handler = FileHandler(to_file, encoding='utf-8')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # Fallback to console if file logging fails
                print(f"Failed to setup file logging: {e}", file=sys.stderr)
            # Always add console handler as well
            _setup_console_handler(logger, name, icon)
        else:
            # Console Logging Strategy (Default)
            _setup_console_handler(logger, name, icon)

    return logger

def _setup_console_handler(logger: Logger, name: str, icon: str):
    """Helper to setup Rich console handler."""
    handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_path=False,
        omit_repeated_times=False,
        show_level=True,
    )
    formatter = logging.Formatter(
        f'{icon} [bold cyan]{name}[/bold cyan] - %(message)s',
        datefmt='%d/%b/%y %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class EndpointFilter(logging.Filter):
    """Filter out specific endpoints from logging."""
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def filter(self, record: logging.LogRecord) -> bool:
        # For uvicorn access logs, the message is the request string
        return record.getMessage().find(self.path) == -1

def suppress_http_logs(paths: list[str] = ["/health", "/logs", "/jobs", "/api/admin/jobs", "/api/admin/all_jobs"]):
    """Suppress access logs for specific HTTP paths in uvicorn."""
    uvicorn_logger = logging.getLogger("uvicorn.access")
    for path in paths:
        uvicorn_logger.addFilter(EndpointFilter(path))