# ==================================== IMPORTS ====================================

# Libraries
import sys
import subprocess
from tqdm import tqdm

# ==================================== FUNCTIONS ====================================

# List of required libraries with versions where necessary
libraries = [
    'albumentations',
    'colorama',
    'effdet',
    'ensemble-boxes',
    'fastcore',
    'fvcore',
    'git+https://github.com/alexhock/object-detection-metrics',
    'matplotlib',
    'object_detection_metrics',
    'Pillow',
    'pycocotools',
    'pytorch-lightning',
    'shutil',
    'functools',
    'tensorboard',
    'timm',
    'torch',
    'torchmetrics',
    'torchsummary',
    'torchvision',
    'tqdm',
    'seaborn'
]

def install_libraries():
    for lib in tqdm(libraries, desc="Installing libraries", unit="package"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib],
                              stdout=subprocess.DEVNULL,  # Redirect stdout
                              stderr=subprocess.DEVNULL)  # Redirect stderr

print("Installing required libraries...")
install_libraries()
print("Done!")
