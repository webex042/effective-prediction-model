"""
EPM Logging System
==================

Provides structured, configurable logging for the entire project.

WHY A CUSTOM LOGGER?
    When running experiments, you need to know:
    - WHEN something happened (timestamp)
    - WHAT happened (message)
    - WHERE it happened (module/function)
    - HOW SEVERE it is (level)

    Python's built-in logging module handles all of this. We just need
    to configure it properly once, then use it everywhere.

HOW IT WORKS:
    1. get_logger(name) creates a named logger
    2. The logger outputs to both console and file (if configured)
    3. Console gets colored output for readability
    4. File gets plain text for parsing/searching

USAGE:
    from src.logger import get_logger

    logger = get_logger(__name__)

    logger.info("Training started")
    logger.debug("Batch 1/100, loss=0.542")
    logger.warning("Learning rate seems too high")
    logger.error("CUDA out of memory")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================================
# Colored Console Formatter
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    A logging formatter that adds color codes to console output.

    Colors make it much easier to scan log output visually:
    - DEBUG:    grey (low importance)
    - INFO:     green (normal operations)
    - WARNING:  yellow (something to watch)
    - ERROR:    red (something broke)
    - CRITICAL: bold red (system failure)
    """

    # ANSI color codes
    COLORS = {
        logging.DEBUG:    "\033[90m",     # Grey
        logging.INFO:     "\033[92m",     # Green
        logging.WARNING:  "\033[93m",     # Yellow
        logging.ERROR:    "\033[91m",     # Red
        logging.CRITICAL: "\033[1;91m",   # Bold Red
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str) -> None:
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        # Color only the level name, keep the rest readable
        record.levelname = f"{color}{record.levelname:<8}{self.RESET}"
        return super().format(record)


# =============================================================================
# Logger Setup
# =============================================================================

# Track whether the root EPM logger has been configured
_configured = False


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
    log_format: Optional[str] = None,
) -> None:
    """
    Configure the root EPM logger.

    Call this once at the start of your script or experiment.
    All loggers created with get_logger() will inherit this configuration.

    Parameters
    ----------
    level : str
        Logging level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    log_dir : str
        Directory for log files.
    console : bool
        Whether to output to console.
    file : bool
        Whether to output to a log file.
    log_format : str, optional
        Custom log format string. If None, uses a sensible default.
    """
    global _configured

    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Get the root EPM logger
    root_logger = logging.getLogger("epm")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers (prevents duplicate output on re-setup)
    root_logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(ColoredFormatter(log_format))
        root_logger.addHandler(console_handler)

    # File handler
    if file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"epm_{timestamp}.log"

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # File always captures everything
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    _configured = True
    root_logger.debug(f"Logging initialized (level={level}, file={file})")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a module.

    Creates a child of the 'epm' root logger. If setup_logging() hasn't
    been called yet, it configures logging with defaults automatically.

    Parameters
    ----------
    name : str
        Logger name. Convention: use __name__ to match the module path.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Examples
    --------
    >>> from src.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Model training started")
    2024-01-15 10:30:00 | INFO     | src.training | Model training started
    """
    global _configured

    if not _configured:
        setup_logging()

    # Create child logger under the 'epm' namespace
    # This ensures all EPM loggers share the same configuration
    if name.startswith("src."):
        logger_name = f"epm.{name}"
    elif name == "__main__":
        logger_name = "epm.main"
    else:
        logger_name = f"epm.{name}"

    return logging.getLogger(logger_name)


def setup_from_config(config) -> None:
    """
    Set up logging from a Config object.

    Parameters
    ----------
    config : Config
        Must have a 'logging' section with keys:
        level, console, file, log_dir, format
    """
    log_cfg = config.logging
    setup_logging(
        level=log_cfg.level,
        log_dir=log_cfg.log_dir,
        console=log_cfg.console,
        file=log_cfg.file,
        log_format=log_cfg.format,
    )
