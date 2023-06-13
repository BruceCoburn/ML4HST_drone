"""
This file contains all the parameters that are used in the training and testing of our Drone Obstacle CNN.
"""

# Image specific parameters
NUM_DUMMY_IMAGES = 1
NUM_CHANNELS = 3
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 720
SIZE_REDUCTION_FACTOR = 4

# User specific parameters
DATA_DIR = "DRONE_OBSTACLES"
ACCELERATOR = "gpu"
DEVICES = [0]
TORCH_MODEL_FILENAME = "drone_obstacle_cnn.pt"
ALSO_TEST = True
SAVE_MODEL = True

# Hyperparameters
BATCH_SIZE = 64
MAX_EPOCHS = 10
MIN_EPOCHS = 1
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 3
