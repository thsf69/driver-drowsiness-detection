import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "drowsiness_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Metrics paths
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics", "evaluation_metrics.json")

# Computer Vision Configuration
CAMERA_INDEX = 0
EAR_THRESHOLD = 0.25      # Baseline threshold for EAR (can be updated by model)
MAR_THRESHOLD = 0.6       # Baseline threshold for MAR (yawning)
CONSECUTIVE_FRAMES = 20   # Number of consecutive frames to trigger alert

# Feature Engineering
ROLLING_WINDOW_SIZE = 10  # Window size for smoothing features over time

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
