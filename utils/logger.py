# ==================================================== IMPORTS ====================================================

# Libraries
from colorama import Fore, Style
import colorama
import logging
import os
import sys
from contextlib import contextmanager
from pytorch_lightning.loggers import TensorBoardLogger

# Custom files
from utils.constants import *

# ==================================================== FUNCTIONS ====================================================

def _init_logger(log_level: int = logging.INFO) -> None:
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(log_level)

    colorama.init(autoreset=True)
    
    class CustomFormatter(logging.Formatter):
        """Logging Formatter to add colors and count warning / errors"""

        grey = Fore.LIGHTBLACK_EX
        yellow = Fore.YELLOW
        red = Fore.RED
        green = Fore.GREEN
        bold_blue = Fore.BLUE + Style.BRIGHT
        bold_red = Fore.RED + Style.BRIGHT

        FORMATS = {
            logging.DEBUG: bold_blue + LOGGING_FORMAT,
            logging.INFO: grey + LOGGING_FORMAT,
            logging.WARNING: yellow + LOGGING_FORMAT,
            logging.ERROR: red + LOGGING_FORMAT,
            logging.CRITICAL: bold_red + LOGGING_FORMAT
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    handler_std_out = logging.StreamHandler(sys.stdout)
    handler_std_out.setFormatter(CustomFormatter())
    check_logs_folder()
    handler_file = logging.FileHandler(LOG_FILE_PATH)
    logger.handlers.clear()
    logger.addHandler(handler_std_out)
    logger.addHandler(handler_file)
    logger.propagate = False
    
def check_logs_folder() -> None:
	"""Check if the logs folder exists, if not, create it."""
	if not os.path.exists(LOG_FILE_DIRECTORY):
		os.makedirs(LOG_FILE_DIRECTORY)
	else:
		open(LOG_FILE_PATH, 'w').close()

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Static instanciation of the app logger
_init_logger(logging.DEBUG)
_app_logger = logging.getLogger(APP_LOGGER_NAME)

# Create specific logger for the trainer
_lightning_logger = TensorBoardLogger(
	save_dir=LOG_FILE_DIRECTORY,
	name=LIGHTNING_LOGGER_NAME
)