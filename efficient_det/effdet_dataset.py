# ==================================== IMPORTS ==================================== #

# Libraries
from torch.utils.data import Dataset
import numpy as np
import torch

# Custom files
from utils.functions import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class EfficientDetDataset(Dataset):
    def __init__(
        self,
        dataset_adaptor,
        transforms=get_valid_transforms(target_img_size=img_size[0]),
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        # Convert bboxes to numpy array if they aren't already (they should be)
        pascal_bboxes = np.array(pascal_bboxes, dtype=np.float32)

        # Convert the image to a NumPy array
        image_np = np.array(image, dtype=np.float32)

        # Prepare sample dictionary
        sample = {
            "image": image_np,
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        # Apply transformations
        sample = self.transforms(**sample)

        # Ensure bboxes remain a 2D numpy array
        sample["bboxes"] = np.array(sample["bboxes"], dtype=np.float32)

        # Check if bboxes array is empty
        if sample["bboxes"].size == 0:
            # Handle the case where there are no bounding boxes
            target = {
                "bboxes": torch.zeros((0, 4), dtype=torch.float32),  # No bboxes
                "labels": torch.zeros((0,), dtype=torch.int64),      # No labels
                "image_id": torch.tensor([image_id]),
                "img_size": (image_np.shape[0], image_np.shape[1]),  # Use NumPy shape here
                "img_scale": torch.tensor([1.0]),
            }

            _app_logger.warning(f"Image ID {image_id}: This image is not treated (no bounding boxes).")
            return sample["image"], target, image_id

        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape

        # Convert bounding box format from (x_min, y_min, x_max, y_max) to (y_min, x_min, y_max, x_max)
        pascal_bboxes[:, [0, 1, 2, 3]] = pascal_bboxes[:, [1, 0, 3, 2]]

        target = {
            "bboxes": torch.as_tensor(pascal_bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }


        return image, target, image_id

    def __len__(self):
        return len(self.ds)