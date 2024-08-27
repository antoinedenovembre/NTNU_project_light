# ==================================== IMPORTS ====================================

from pathlib import Path

# ==================================== CONSTANTS ====================================

# Model constants
IMG_SIZE = (512, 512)
ARCHITECTURE = "tf_efficientdet_d0"
EPOCHS = 200
BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.5
EARLY_STOPPING = False
VERSION = 0.1

# File paths
RESULTS_DIR = Path("output")
MODEL_DIR = RESULTS_DIR / Path("model")
MODEL_BACKUP_DIR = RESULTS_DIR / Path("model_backup")
GRAPH_DIR = RESULTS_DIR / Path("graphs")
IMG_DIR = RESULTS_DIR / Path("images")
MODEL_NAME = Path(f"effdet_{ARCHITECTURE}_no_barlow_V{VERSION}.pth")
MODEL_FULL_PATH = MODEL_DIR / MODEL_NAME
MODEL_BACKUP_FULL_PATH = MODEL_BACKUP_DIR / MODEL_NAME
LOSS_CURVE_FULL_PATH = GRAPH_DIR / Path("loss_curve.png")
PRECISION_RECALL_FULL_PATH = GRAPH_DIR / Path("precision_recall_curve.png")
DETECTION_PERF_FULL_PATH = GRAPH_DIR / Path("detection_performance.png")
F1_SCORE_FULL_PATH = GRAPH_DIR / Path("f1_score.png")
CONFUSION_MATRIX_FULL_PATH = GRAPH_DIR / Path("confusion_matrix.png")

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