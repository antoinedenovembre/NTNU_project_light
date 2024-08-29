# ==================================== IMPORTS ====================================

from pathlib import Path

# ==================================== CONSTANTS ====================================

# General constants
VERSION = 0.3

# Model constants
IMG_SIZE = (512, 512)
ARCHITECTURE_D0 = "tf_efficientdet_d0"
ARCHITECTURE_D1 = "tf_efficientdet_d1"
MODEL_EPOCHS = 160
MODEL_BATCH_SIZE = 8

BACKBONE_EPOCHS = 50
BACKBONE_BATCH_SIZE = 2
BACKBONE_NUM_WORKERS = 8
BACKBONE_Z_DIM = 4096
BACKBONE_ENC_OUT_DIMS = {"tf_efficientdet_d0": 64, "tf_efficientdet_d1": 88}
CONFIDENCE_THRESHOLD = 0.5
EARLY_STOPPING = False
USE_BACKBONE = True

# File paths
RESULTS_DIR = Path("output")
MODEL_DIR = RESULTS_DIR / Path("model")
MODEL_NO_BARLOW_NAME = Path(f"effdet_{ARCHITECTURE_D0}_no_barlow_V{VERSION}.pth")
MODEL_BARLOW_NAME = Path(f"effdet_{ARCHITECTURE_D0}_barlow_V{VERSION}.pth")
MODEL_BACKUP_DIR = RESULTS_DIR / Path("model_backup")
BACKBONE_NAME = Path(f"backbone_{ARCHITECTURE_D1}_V{VERSION}.pth")
BACKBONE_FULL_PATH = MODEL_DIR / BACKBONE_NAME

GRAPH_DIR = RESULTS_DIR / Path("graphs")
IMG_DIR = RESULTS_DIR / Path("images")
LOSS_CURVE_FULL_PATH = GRAPH_DIR / Path(f"loss_curve_{VERSION}.png")
DETECTION_PERF_FULL_PATH = GRAPH_DIR / Path(f"detection_performance_{VERSION}.png")
F1_SCORE_FULL_PATH = GRAPH_DIR / Path(f"f1_score_{VERSION}.png")
CONFUSION_MATRIX_FULL_PATH = GRAPH_DIR / Path(f"confusion_matrix_{VERSION}.png")
MAP_FULL_PATH = GRAPH_DIR / Path(f"mAPs_{VERSION}.png")

# Dataset constants
DATASET = Path("data")
MODEL = Path("effdet")
BACKBONE = Path("backbone")
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