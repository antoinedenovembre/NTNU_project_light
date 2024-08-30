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
from efficient_det.pigs_dataset_adapter import PigsDatasetAdapterEffDet
from efficient_det.effdet_data_module import EfficientDetDataModule
from efficient_det.effdet_model import EfficientDetModel
from efficient_det.loss_plot_callback import LossPlotCallback
from barlow.pigs_dataset_adapter import PigsDatasetAdapterBackbone
from barlow.barlow_data_module import BarlowTwinsDataModule
from barlow.barlow_twins import BarlowTwins
from barlow.barlow_transform import BarlowTwinsTransform
from barlow.repr_net import ReprNet
from utils.constants import *
from utils.functions import *

# ===================================== GLOBALS ===================================== #

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

def train_efficient_net_backbone(num_sanity_val_steps=1):
    # Create paths
    train_images_path = DATASET / BACKBONE
    val_images_path = DATASET / MODEL / VALIDATION / IMAGE

    # Logging paths
    _app_logger.debug(f"Training images path: {train_images_path}")
    _app_logger.debug(f"Validation images path: {val_images_path}")

    # Dataset setup
    _app_logger.info("Setting up datasets...")
    pigs_train_ds = PigsDatasetAdapterBackbone(train_images_path)
    pigs_val_ds = PigsDatasetAdapterBackbone(val_images_path)
    _app_logger.info("Datasets setup complete.")

    # Model setup
    _app_logger.info("Setting up image augmentation...")
    train_transform = BarlowTwinsTransform(
        train=True,
        input_height=IMG_SIZE[0],
        gaussian_blur=False,
        jitter_strength=0.5,
        normalize=imagenet_normalization(),
    )
    val_transform = BarlowTwinsTransform(
        train=False,
        input_height=IMG_SIZE[0],
        gaussian_blur=False,
        jitter_strength=0.5,
        normalize=imagenet_normalization(),
    )
    _app_logger.info("Image augmentation setup complete.")

    _app_logger.info("Setting up image augmentation...")
    dm = BarlowTwinsDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        train_transforms=train_transform,
        valid_transforms=val_transform,
        num_workers=BACKBONE_NUM_WORKERS,
        batch_size=BACKBONE_BATCH_SIZE,
    )
    _app_logger.info("Datasets and DataModule setup complete.")

    _app_logger.info(f"Creating Barlow Twins model with architecture {ARCHITECTURE_D1}...")
    encoder_out_dim = BACKBONE_ENC_OUT_DIMS[ARCHITECTURE_D1]
    encoder = create_backbone_model(architecture=ARCHITECTURE_D1, image_size=IMG_SIZE[0])
    model = BarlowTwins(
        encoder=encoder,
        encoder_out_dim=encoder_out_dim,
        get_repr=ReprNet(config=get_efficientdet_config(ARCHITECTURE_D1)),
        num_training_samples=len(pigs_train_ds),
        batch_size=BACKBONE_BATCH_SIZE,
        z_dim=BACKBONE_Z_DIM,
    )
    _app_logger.info("Model creation complete.")

    # Trainer setup
    _app_logger.info(f"Starting training for {BACKBONE_EPOCHS} epochs...")
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    trainer = Trainer(
        logger=_lightning_logger,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        max_epochs=BACKBONE_EPOCHS,
        num_sanity_val_steps=num_sanity_val_steps,
        accumulate_grad_batches=8,
        log_every_n_steps=31
    )
    trainer.fit(model, dm)
    _app_logger.info("Training complete.")

    # Train the model
    _app_logger.info(f"Starting training for {BACKBONE_EPOCHS} epochs...")
    trainer.fit(model, dm)
    _app_logger.info("Training complete.")

    # Save the trained model weights
    torch.save(model.state_dict(), BACKBONE_FULL_PATH)
    _app_logger.info(f"Model saved to {BACKBONE_FULL_PATH}")

def train_efficient_det(num_sanity_val_steps=1, use_backbone=True):
    # Create paths
    train_images_path = DATASET / MODEL / TRAIN / IMAGE
    val_images_path = DATASET / MODEL / VALIDATION / IMAGE
    train_anns_path = DATASET / MODEL / TRAIN / ANNOTATION
    val_anns_path = DATASET / MODEL / VALIDATION / ANNOTATION

    # Logging paths
    _app_logger.debug(f"Training images path: {train_images_path}")
    _app_logger.debug(f"Validation images path: {val_images_path}")
    _app_logger.debug(f"Training annotations path: {train_anns_path}")
    _app_logger.debug(f"Validation annotations path: {val_anns_path}")

    # Dataset setup
    _app_logger.info("Setting up datasets...")
    pigs_train_ds = PigsDatasetAdapterEffDet(train_images_path, train_anns_path)
    pigs_val_ds = PigsDatasetAdapterEffDet(val_images_path, val_anns_path)
    
    dm = EfficientDetDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        num_workers=4,
        batch_size=MODEL_BATCH_SIZE,
    )
    _app_logger.info("Datasets and DataModule setup complete.")

    # Model setup
    _app_logger.info(f"Creating EfficientDet model with architecture {ARCHITECTURE_D1}...")
    model = EfficientDetModel(
        num_classes=1,
        img_size=IMG_SIZE[0],
        model_architecture=ARCHITECTURE_D1,
        iou_threshold=0.44,
        prediction_confidence_threshold=CONFIDENCE_THRESHOLD,
        sigma=0.8,
        learning_rate=0.003
    )
    _app_logger.info("Model creation complete.")

    # Trainer setup
    _app_logger.info(f"Starting training for {MODEL_EPOCHS} epochs...")
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
        max_epochs=MODEL_EPOCHS,
        num_sanity_val_steps=num_sanity_val_steps,
        accumulate_grad_batches=3,
        log_every_n_steps=31,
        callbacks=callbacks
    )
    trainer.fit(model, dm)
    _app_logger.info("Training complete.")

    MODEL_FULL_PATH = ""
    if use_backbone:
        MODEL_FULL_PATH = MODEL_DIR / MODEL_BARLOW_NAME
    else:
        MODEL_FULL_PATH = MODEL_DIR / MODEL_NO_BARLOW_NAME
    # Saving model
    torch.save(
        model.state_dict(),
        MODEL_FULL_PATH,
    )
    _app_logger.info(f"Model saved to {MODEL_FULL_PATH}")

def validate_efficient_det(num_sanity_val_steps=1):
    # Create paths
    train_images_path = DATASET / MODEL / TRAIN / IMAGE
    val_images_path = DATASET / MODEL / VALIDATION / IMAGE
    train_anns_path = DATASET / MODEL / TRAIN / ANNOTATION
    val_anns_path = DATASET / MODEL / VALIDATION / ANNOTATION

    MODEL_FULL_PATH = ""
    models_available = scan_models_in_output()
    # Ask user to choose a model to validate
    for i, model in enumerate(models_available):
        print(f"{i}. {model}")
    model_choice = int(input("Enter the model number to validate: "))
    MODEL_FULL_PATH = models_available[model_choice]

    if not MODEL_FULL_PATH:
        _app_logger.error("No model found. Exiting...")
        exit(1)

    _app_logger.info(f"Loading model from {MODEL_FULL_PATH} for validation.")

    pigs_train_ds = PigsDatasetAdapterEffDet(train_images_path, train_anns_path)
    pigs_val_ds = PigsDatasetAdapterEffDet(val_images_path, val_anns_path)

    dm = EfficientDetDataModule(
        train_dataset_adaptor=pigs_train_ds,
        validation_dataset_adaptor=pigs_val_ds,
        num_workers=4,
        batch_size=MODEL_BATCH_SIZE,
    )

    model = EfficientDetModel(
        num_classes=1,
        img_size=IMG_SIZE[0],
        model_architecture=ARCHITECTURE_D1,
        iou_threshold=0.8,
        prediction_confidence_threshold=CONFIDENCE_THRESHOLD,
        sigma=0.8,
    )

    model.load_state_dict(torch.load(MODEL_FULL_PATH, weights_only=True))
    model.eval()

    all_truths = []
    all_predictions = []

    for i in tqdm.tqdm(range(len(pigs_val_ds)), desc="Validating model", unit="image"):
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
