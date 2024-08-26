# ==================================== IMPORTS ====================================

from pathlib import Path

# ==================================== CONSTANTS ====================================

# Model constants
IMG_SIZE = (512, 512)
ARCHITECTURE = "tf_efficientdet_d0"
EPOCHS = 200
BATCH_SIZE = 32
RESULTS_DIR = Path("output")
MODEL_DIR = Path("model")
GRAPH_DIR = Path("graphs")
IMG_DIR = RESULTS_DIR / Path("images")
MODEL_FULL_PATH = RESULTS_DIR / MODEL_DIR / Path(f"effdet_{ARCHITECTURE}_no_barlow.pth")
LOSS_CURVE_FULL_PATH = RESULTS_DIR / GRAPH_DIR / Path("loss_curve.png")

# Dataset constants
DATASET = Path("data")
TRAIN = Path("train")
VALIDATION = Path("val")
IMAGE = Path("images")
ANNOTATION = Path("annotations")

# Logging constants
LOG_FILE_DIRECTORY = "logs"
LOG_FILE_PATH = f"{LOG_FILE_DIRECTORY}/logs.log"
LOGGING_FORMAT = "%(levelname)s: %(message)s"
APP_LOGGER_NAME = "app_logger"
LIGHTNING_LOGGER_NAME = "lightning_logger"