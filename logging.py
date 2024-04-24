import logging
import logging.config
import os

import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

try:
    from .helper_functions import path_prefix
except:
    from helper_functions import path_prefix


def get_log_config(log_name: str) -> dict:
    """Returns a logging config dict for use in logging.config.dictConfig

    Args:
        log_name (str): file that will be using the logger

    Returns:
        dict: dict of config for logging streams
    """

    log_path = f"{path_prefix}logging/{log_name}"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    LOGGING_CONFIG = {
        "version": 1,
        "loggers": {
            "": {  # root logger
                "level": "NOTSET",
                "handlers": [
                    "debug_console_handler",
                    "info_rotating_file_handler",
                    "error_file_handler",
                ],
            },
            "my.package": {
                "level": "WARNING",
                "propagate": False,
                "handlers": [
                    "info_rotating_file_handler",
                    "error_file_handler",
                ],
            },
        },
        "handlers": {
            "debug_console_handler": {
                "level": "DEBUG",
                "formatter": "info",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "info_rotating_file_handler": {
                "level": "INFO",
                "formatter": "info",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": os.path.join(log_path, "info.log"),
                "mode": "a",
                "maxBytes": 1048576,
                "backupCount": 10,
            },
            "error_file_handler": {
                "level": "WARNING",
                "formatter": "error",
                "class": "logging.FileHandler",
                "filename": os.path.join(log_path, "error.log"),
                "mode": "a",
            },
            # "critical_mail_handler": {
            #     "level": "CRITICAL",
            #     "formatter": "error",
            #     "class": "logging.handlers.SMTPHandler",
            #     "mailhost": "smtprelay.horizontherapeutics.local",
            #     "fromaddr": "djosephs@horizontherapeutics.com",
            #     "toaddrs": ["djosephs@horizontherapeutics.com"],
            #     "subject": f"Critical error with {log_name}",
            # },
        },
        "formatters": {
            "info": {
                "format": "%(asctime)s-%(levelname)s-%(name)s::|%(lineno)s:: %(message)s"
            },
            "error": {
                "format": "%(asctime)s-%(levelname)s-%(name)s::|%(lineno)s:: %(message)s"
            },
        },
    }

    return LOGGING_CONFIG


def get_logger(log_name: str) -> logging.Logger:
    """Creates a logging object

    Args:
        log_name (str): name of logger


    Returns:
        logging.Logger: the current logger used by the file
    """

    logging.config.dictConfig(get_log_config(log_name))

    # Get the logger specified in the file
    logger = logging.getLogger(log_name)
    service_key_path = (
        f"{path_prefix}Repositories/prod/keys/logging_service_account.json"
    )

    gcloud_logging_client = google.cloud.logging.Client.from_service_account_json(
        service_key_path
    )
    gcloud_logging_handler = CloudLoggingHandler(gcloud_logging_client, name=log_name)
    logger.addHandler(gcloud_logging_handler)
    return logger
