import os
import pathlib

# Base directory
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()

# Data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Model parameters
IMG_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 5

# Output paths
MODEL_SAVE_PATH = BASE_DIR / "outputs" / "models" / "best_model.h5"
