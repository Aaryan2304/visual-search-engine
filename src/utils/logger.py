import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from ..config import Config

# Flag to ensure setup is run only once
_logging_setup_done = False


def setup_logging():
    """
    Configures the root logger for the application.
    It sets up logging to both the console and a rotating file.
    """
    global _logging_setup_done
    if _logging_setup_done:
        return

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(Config.LOG_LEVEL)

    # Create formatter
    formatter = logging.Formatter(Config.LOG_FORMAT)

    # --- Console Handler ---
    # Logs messages to the standard output (your terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- File Handler ---
    # Creates a new log file every day and keeps the last 7 days of logs
    log_file_path = os.path.join(Config.LOG_DIR, "app.log")
    file_handler = TimedRotatingFileHandler(
        log_file_path, when="midnight", interval=1, backupCount=7
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # --- Mark setup as complete ---
    _logging_setup_done = True
    logger.info("Logging configured successfully.")


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance for a specific module.
    It ensures that the logging system is configured before returning the logger.

    Args:
        name (str): The name of the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Ensure logging is set up before any logger is used
    if not _logging_setup_done:
        setup_logging()

    return logging.getLogger(name)