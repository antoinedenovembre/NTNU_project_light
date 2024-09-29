# ==================================== IMPORTS ==================================== #

# Libraries
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
warnings.filterwarnings("ignore", category=UserWarning) # Suppress UserWarnings

# Custom files
from utils.logger import _app_logger
from scripts.efficient_det import train_efficient_det, evaluate_efficient_det, train_efficient_net_backbone
from utils.functions import backup_models, print_title, print_option

# =================================== FUNCTIONS =================================== #

def manage_output_directories():
	from utils.constants import RESULTS_DIR, MODEL_DIR, MODEL_BACKUP_DIR, GRAPH_DIR, IMG_DIR
	for directory in [RESULTS_DIR, MODEL_DIR, MODEL_BACKUP_DIR, GRAPH_DIR, IMG_DIR]:
		if not directory.exists():
			directory.mkdir(parents=True)
			_app_logger.info(f"Created directory: {directory}")

# ===================================== MAIN ===================================== #

def main():
	# Create output directories
	manage_output_directories()

	# Main menu
	while True:
		# Clear terminal before printing menu
		# os.system('cls' if os.name == 'nt' else 'clear')

		print_title("======================================= Main Menu =======================================")
		print()
		print_option("1. Train EfficientDet model")
		print_option("2. Evaluate EfficientDet model")
		print_option("3. Train EfficientNet backbone")
		print_option("4. Backup models in model directory")
		print_option("0. Exit")
		print()

		choice = input("Enter your choice (0-4): ")
		use_backbone = False
		if choice == "1":
			use_backbone = input("Use backbone? (y/n): ")
			backbone_choice = (use_backbone == "y")
			_app_logger.info(f"Using backbone: {backbone_choice}")

		_app_logger.info(f"Choice: {choice}")

		if choice == "1":
			_app_logger.info("Training EfficientDet model...")
			train_efficient_det(num_sanity_val_steps=1, use_backbone=backbone_choice)
		elif choice == "2":
			_app_logger.info("Evaluating EfficientDet model...")
			evaluate_efficient_det(num_sanity_val_steps=1)
		elif choice == "3":
			_app_logger.info("Training EfficientNet backbone...")
			train_efficient_net_backbone(num_sanity_val_steps=1)
		elif choice == "4":
			_app_logger.info("Backing up models in model directory...")
			backup_models()
		elif choice == "0":
			break
		else:
			_app_logger.error("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
