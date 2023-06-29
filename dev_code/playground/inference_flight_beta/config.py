"""
This file contains all the parameters that are used in the training and testing of our Drone Obstacle CNN.
"""

##################################
# Image specific parameters
##################################
NUM_DUMMY_IMAGES = 1
NUM_CHANNELS = 3
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 720
IMAGE_REFRESH_RATE = 0.2

##################################
# User specific parameters
##################################
# Hardware parameters
ACCELERATOR = "gpu"
DEVICES = [0]  # Default: [0] (i.e. use only the first GPU)
NUM_WORKERS = 1  # Determined as optimal from determine_optimal_num_workers.py

# Dataset location
DATA_DIR = "DRONE_OBSTACLES"

# Logging/Saving parameters
TORCH_MODEL_FILENAME = "drone_obstacle_cnn"
TORCH_MODEL_FILENAME_EXT = ".pt"
TORCH_MODEL_FILENAME_LOAD = "drone_obstacle_cnn_acc_0.81.pt"
INFERENCE_MODEL_FILENAME = "drone_obstacle_cnn_acc_0.81.pt"
CHECKPOINT_DIR = "checkpoints"
TEST_AND_SAVE_MODEL_PT = True
TEST_AND_SAVE_MODEL_CKPT = True
LOAD_AND_TEST = False
LOG_EVERY_N_STEPS = 1

# Parameters related to dataset "re-scrambling"
DATASET_FILE_EXT = ".jpg"
SOURCE_FOLDER = "C:\\Users\\bcoburn1\OneDrive - University of Wyoming\\Desktop\\ML4HST_drone\\dev_code\\playground\\reference_CNN_lightning_model\\obstacle_dataset"
DESTINATION_FOLDER = "DRONE_OBSTACLES"  # Uncomment this if you want to augment the 'original' dataset directory
# DESTINATION_FOLDER = "DRONE_OBSTACLES_RESCRAMBLE"
CLASS_A_NAME = "BLOCKED"
CLASS_B_NAME = "UNBLOCKED"
SPLIT_RATIO = (0.8, 0.1, 0.1)

##################################
# Hyperparameters
##################################
# Training parameters
BATCH_SIZE = 64
MAX_EPOCHS = 50
MIN_EPOCHS = 5
LEARNING_RATE = 0.001

# Early Stopping parameters
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.005

# Transformation parameters
SIZE_REDUCTION_FACTOR = 3
