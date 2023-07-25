import os
import logging
from logging.handlers import RotatingFileHandler


def initiate_logger(
    log_location :str,
    log_name: str,
    max_bytes: int = 10000,
    backups: int = 10,
    log_level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
):
    """Initiate logger with custom formatting and rotating file handler.
    Once logger file size reach max_bytes it will move current file to backup,
    adding a prefix.

    Parameters
    ----------
    log_location : str
        Location to save log files.
    log_name : str
        Name of log files.
    max_bytes : int, optional
        Max file size of log file, by default 10000
    backups : int, optional
        Number of backups to keep, by default 10
    log_level : optional
        Log level to use for log messages, by default logging.INFO
    """
    full_name = os.path.join(log_location, log_name)

    logging.basicConfig(
        handlers=[
            RotatingFileHandler(
                filename=full_name,
                maxBytes=max_bytes,
                backupCount=backups,
            ),
            logging.StreamHandler(),
        ],
        level=log_level,
        format=format,
        datefmt=datefmt,
    )
    
