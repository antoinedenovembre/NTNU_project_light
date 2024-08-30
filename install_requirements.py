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
    'rich',
    'tensorboard',
    'timm',
    'torch',
    'torchmetrics',
    'torchsummary',
    'torchvision',
    'tqdm',
    'seaborn'
]

def check_tqdm():
    try:
        import tqdm
        if tqdm.__version__ < "4.61.0":
            print("Updating tqdm...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm>=4.61.0"],
                                  stdout=subprocess.DEVNULL,  # Redirect stdout
                                  stderr=subprocess.DEVNULL)  # Redirect stderr
    except ImportError:
        print("Installing tqdm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"],
                              stdout=subprocess.DEVNULL,  # Redirect stdout
                              stderr=subprocess.DEVNULL)  # Redirect stderr

def install_libraries():
    for lib in tqdm(libraries, desc="Installing libraries", unit="package"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib],
                              stdout=subprocess.DEVNULL,  # Redirect stdout
                              stderr=subprocess.DEVNULL)  # Redirect stderr

print("Installing required libraries...")
check_tqdm()
install_libraries()
print("Done!")
