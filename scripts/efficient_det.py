# ==================================== IMPORTS ==================================== #

# Libraries
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()
import os
import logging
import tqdm

# Custom files
from utils.logger import _app_logger, _lightning_logger
from efficient_det.pigs_dataset_adapter import PigsDatasetAdapter
from efficient_det.effdet_data_module import EfficientDetDataModule
from efficient_det.effdet_model import EfficientDetModel
from efficient_det.loss_plot_callback import LossPlotCallback
from utils.constants import *
from utils.functions import *

# ===================================== GLOBALS ===================================== #

# Paths setup
train_images_path = DATASET / TRAIN / IMAGE
val_images_path = DATASET / VALIDATION / IMAGE
train_anns_path = DATASET / TRAIN / ANNOTATION
val_anns_path = DATASET / VALIDATION / ANNOTATION

# Accelerator and devices
def choose_accelerator_and_devices():
    # Detect if running on a Mac
    if os.name == 'posix' and 'Darwin' in os.uname().sysname:
        _app_logger.debug("Running on MacOS.")
        accelerator = 'cpu'
        devices = 1  # Number of CPU devices
    elif os.name == 'posix' and 'Linux' in os.uname().sysname:
        _app_logger.debug("Running on Linux.")
        # Non-MacOS logic
        if torch.cuda.is_available():
            _app_logger.debug("CUDA is available. Using GPU.")
            accelerator = 'gpu'
            devices = torch.cuda.device_count()  # Number of available GPUs
            _app_logger.debug(f"Using {devices} device(s).")
        else:
            _app_logger.debug("CUDA not available. Using CPU.")
            accelerator = 'cpu'
            devices = 1
    else:
        _app_logger.error("Unsupported OS. Exiting...")
        exit(1)

    _app_logger.debug(f"Chosen accelerator: {accelerator}, Number of devices: {devices}")
    return accelerator, devices

ACCELERATOR, DEVICES = choose_accelerator_and_devices()

# ===================================== FUNCTIONS =====================================

def train_efficient_det(num_sanity_val_steps=1):
    # Logging paths
    _app_logger.debug(f"Training images path: {train_images_path}")
    _app_logger.debug(f"Validation images path: {val_images_path}")
    _app_logger.debug(f"Training annotations path: {train_anns_path}")
    _app_logger.debug(f"Validation annotations path: {val_anns_path}")

    # Dataset setup
    _app_logger.info("Setting up datasets...")
    pigs_train_ds = PigsDatasetAdapter(train_images_path, train_anns_path)
    pigs_val_ds = PigsDatasetAdapter(val_images_path, val_anns_path)
    
    dm = EfficientDetDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        num_workers=4,
        batch_size=BATCH_SIZE,
    )

    _app_logger.info("Datasets and DataModule setup complete.")

    # Model setup
    _app_logger.info(f"Creating EfficientDet model with architecture {ARCHITECTURE}...")
    model = EfficientDetModel(
        num_classes=1,
        img_size=IMG_SIZE[0],
        model_architecture=ARCHITECTURE,
        iou_threshold=0.44,
        prediction_confidence_threshold=CONFIDENCE_THRESHOLD,
        sigma=0.8,
        learning_rate=0.003
    )
    _app_logger.info("Model creation complete.")

    # Trainer setup
    _app_logger.info(f"Starting training for {EPOCHS} epochs...")
    callbacks = [LossPlotCallback()]
    # If early stopping in callback list, log it
    if EARLY_STOPPING:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=5))
        _app_logger.info("Early stopping enabled.")
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    trainer = Trainer(
        logger=_lightning_logger,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        max_epochs=EPOCHS,
        num_sanity_val_steps=num_sanity_val_steps,
        accumulate_grad_batches=3,
        log_every_n_steps=31,
        callbacks=[LossPlotCallback()],
    )
    trainer.fit(model, dm)
    _app_logger.info("Training complete.")

    # Saving model
    torch.save(
        model.state_dict(),
        MODEL_FULL_PATH,
    )
    _app_logger.info(f"Model saved to {MODEL_FULL_PATH}")

def validate_efficient_det(num_sanity_val_steps=1):
    if not os.path.exists(MODEL_FULL_PATH):
        _app_logger.error(f"Model file not found at {MODEL_FULL_PATH}. Validation aborted.")
        return

    _app_logger.info(f"Loading model from {MODEL_FULL_PATH} for validation.")

    pigs_train_ds = PigsDatasetAdapter(train_images_path, train_anns_path)
    pigs_val_ds = PigsDatasetAdapter(val_images_path, val_anns_path)

    dm = EfficientDetDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        num_workers=4,
        batch_size=BATCH_SIZE,
    )

    model = EfficientDetModel(
        num_classes=1,
        img_size=IMG_SIZE[0],
        model_architecture=ARCHITECTURE,
        iou_threshold=0.8,
        prediction_confidence_threshold=CONFIDENCE_THRESHOLD,
        sigma=0.8,
    )

    model.load_state_dict(torch.load(MODEL_FULL_PATH, weights_only=True))
    model.eval()

    all_truths = []
    all_predictions = []

    for i in tqdm.tqdm(range(len(pigs_val_ds) - 230), desc="Validating model", unit="image"):
        image, truth_bboxes, _, _ = pigs_val_ds.get_image_and_labels_by_idx(i)
        all_truths.append(truth_bboxes.tolist())

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict([image])
        all_predictions.append((predicted_bboxes[0], predicted_class_confidences[0], predicted_class_labels[0]))
        
        # Compare predictions with actual bounding boxes for the current image
        show_bboxes_on_image(
                image,
                predicted_bboxes=predicted_bboxes[0],
                predicted_class_confidences=predicted_class_confidences[0],
                actual_bboxes=truth_bboxes.tolist(),
                image_id=i
        )
    
    get_all_metrics(all_predictions, all_truths)
