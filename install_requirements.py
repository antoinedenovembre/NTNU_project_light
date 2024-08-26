# ==================================== IMPORTS ====================================

# Libraries
import sys
import subprocess
from tqdm import tqdm

# ==================================== FUNCTIONS ====================================

# List of required libraries with versions where necessary
libraries = [
    'albumentations',             # Use latest unless a specific version is needed
    'colorama',                   # Use latest unless a specific version is needed
    'effdet',                     # Match this with compatible version for torch 2.1.2
    'ensemble-boxes',             # Use latest unless a specific version is needed
    'fastcore',                   # Use latest unless a specific version is needed
    'fvcore',                     # Use latest unless a specific version is needed
    'git+https://github.com/alexhock/object-detection-metrics',
    'matplotlib',                 # Use latest unless a specific version is needed
    'object_detection_metrics',   # Use latest unless a specific version is needed
    'Pillow',                     # Use latest unless a specific version is needed
    'pycocotools',                # Use latest unless a specific version is needed
    'pytorch-lightning',          # Match this with compatible version for torch 2.1.2
    'tensorboard',                # Use latest unless a specific version is needed
    'torch',                      # Specific version of torch
    'torchsummary',               # Use latest unless a specific version is needed
    'torchvision',                # Match this with compatible version for torch 2.1.2
    'tqdm'                        # Use latest unless a specific version is needed
]

def install_libraries():
    for lib in tqdm(libraries, desc="Installing libraries", unit="package"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib],
                              stdout=subprocess.DEVNULL,  # Redirect stdout
                              stderr=subprocess.DEVNULL)  # Redirect stderr

print("Installing required libraries...")
install_libraries()
print("Done!")
