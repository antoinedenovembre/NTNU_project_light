# ==================================== IMPORTS ==================================== #

# Libraries
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
warnings.filterwarnings("ignore", category=UserWarning) # Suppress UserWarnings

# Custom files
from utils.logger import _app_logger
from scripts.efficient_det import train_efficient_det, validate_efficient_det

# =================================== FUNCTIONS =================================== #

def manage_output_directories():
	from utils.constants import RESULTS_DIR, MODEL_DIR, GRAPH_DIR, IMG_DIR
	for directory in [RESULTS_DIR, MODEL_DIR, GRAPH_DIR, IMG_DIR]:
		if not directory.exists():
			directory.mkdir(parents=True)
			_app_logger.info(f"Created directory: {directory}")

# ===================================== MAIN ===================================== #

def main():
	# Create output directories
	manage_output_directories()

	# Main menu
	while True:
		print("========== Main Menu ==========")
		print("1. Train EfficientDet model")
		print("2. Validate EfficientDet model")
		print("0. Exit")

		choice = input("Enter your choice (0-2): ")

		if choice == "1":
			_app_logger.info("Training EfficientDet model...")
			train_efficient_det(num_sanity_val_steps=0)
		elif choice == "2":
			_app_logger.info("Validating EfficientDet model...")
			validate_efficient_det(num_sanity_val_steps=0)
		elif choice == "0":
			break
		else:
			_app_logger.error("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
