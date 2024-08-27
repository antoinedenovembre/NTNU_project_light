# ==================================== IMPORTS ==================================== #

# Libraries
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import json

# Custom files
from utils.functions import *
from utils.logger import _app_logger

# ===================================== CLASS ===================================== #

class PigsDatasetAdapter:
    def __init__(self, images_path, anns_path):
        self.images_path = images_path
        self.anns_path = anns_path

         # Check if annotation directory exists
        if not anns_path.exists():
            raise FileNotFoundError(f"Annotation directory not found: {anns_path}")

        # Check if image directory exists
        if not images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {images_path}")

        # Get all image files
        self.image_paths = [path for path in images_path.iterdir() if path.suffix.lower() in ['.jpg', '.jpeg', '.png']]

        # Load COCO annotations from all JSON files in the directory
        self.coco = COCO()
        for ann_file in self.anns_path.glob("*.json"):
            with open(ann_file, 'r') as f:
                dataset = json.load(f)
            self.coco.dataset.update(dataset)
            # Silencing stdout while creating the index
            import os, contextlib
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.coco.createIndex()

        # Map image filenames to COCO image IDs
        self.image_id_map = {}
        for image_info in self.coco.dataset['images']:
            self.image_id_map[image_info['file_name']] = image_info['id']


        # Collect annotations for each image
        self.annotations = {}
        for img_path in self.image_paths:
            img_name = img_path.name
            if img_name in self.image_id_map:
                img_id = self.image_id_map[img_name]
                # Get annotations for this image
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                self.annotations[img_name] = self.coco.loadAnns(ann_ids)


    def __len__(self):
        return len(self.image_paths)

    def get_annotations(self, image_filename):
        """Get annotations for a specific image by filename."""
        if image_filename in self.annotations:
            return self.annotations[image_filename]
        else:
            _app_logger.warning(f"No annotations found for image {image_filename}")
            return []

    def get_image_and_labels_by_idx(self, index):
        # Open the image file at the given index
        image = Image.open(self.image_paths[index])

        # Get the image file name
        img_name = self.image_paths[index].name

        # Get annotations for this image
        if img_name in self.annotations:
            anns = self.annotations[img_name]
        else:
            _app_logger.warning(f"No annotations found for image {img_name}")
            anns = []

        # Process annotations and filter out any problematic ones
        boxes = np.zeros((len(anns), 4))
        bad_anns = []
        for i, ann in enumerate(anns):
            try:
                boxes[i, :] = get_pascal_bbox(ann["bbox"])
                if np.all(boxes[i, :] == 0):
                    bad_anns.append(i)
                    continue
            except ValueError:
                _app_logger.error(f"Bad bouding box, image ID: {img_name}")
                bad_anns.append(i)
                continue

        # Remove bad annotations
        boxes = np.delete(boxes, bad_anns, 0)
        labels = np.ones(len(anns))
        labels = np.delete(labels, bad_anns, 0)

        return image, boxes, labels, index
    
    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        _app_logger.info(f"Image_id: {image_id}")
        show_image(image, bboxes.tolist())
        _app_logger.info(class_labels)