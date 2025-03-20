import logging
import os
from logging.handlers import RotatingFileHandler

from adwersbad.config import config

__all__ = ["setup_logger"]


def setup_logger(
    module_name: str,
    log_level="DEBUG",
    log_folder="logs/",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_max_size=1073741824,
    log_backup_count=2,
    to_console=False,
):
    """Sets up a logger for a specific module."""
    logger = logging.getLogger(module_name)

    if os.getenv("TESTING") == "true":
        logger.addHandler(logging.NullHandler())
        logger.setLevel("CRITICAL")
        return logger

    log_params = config(section="logging")
    logger.setLevel(log_params["log_level"])
    log_folder = log_params["log_folder"]
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    log_file = os.path.join(log_folder, f"{module_name}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=int(log_max_size), backupCount=int(log_backup_count)
    )

    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Prevent the log from propagating to the root logger
    logger.propagate = False

    logger.info(f"Logging setup complete for {module_name}.")
    return logger


# def get_logger(module_name):
#     """Returns a logger instance for the given module."""
#     return logging.getLogger(f'mypackage.{module_name}')


# Null handler to avoid "No handler found" warnings if user doesn't configure logging.
logging.getLogger("adwersbad").addHandler(logging.NullHandler())
