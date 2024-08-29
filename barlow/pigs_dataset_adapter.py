# ==================================== IMPORTS ==================================== #

# Libraries
from PIL import Image
import os

# ===================================== CLASS ===================================== #

class PigsDatasetAdapterBackbone:
    def __init__(self, images_path):
        self.image_paths = [images_path / path for path in os.listdir(images_path)]

    def __len__(self):
        return len(self.image_paths)

    def get_image_by_idx(self, index):
        image = Image.open(self.image_paths[index])
        return image