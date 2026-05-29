"""
Configuration module for the Bengaluru Traffic Demand Prediction project.
Defines paths, feature structures, and model hyperparameters.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they do not exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data paths (with fallbacks to root in case data files are in root)
def get_data_path(filename):
    path_in_data = os.path.join(DATA_DIR, filename)
    path_in_root = os.path.join(BASE_DIR, filename)
    if os.path.exists(path_in_data):
        return path_in_data
    elif os.path.exists(path_in_root):
        return path_in_root
    return path_in_data  # Default fallback

TRAIN_PATH = get_data_path("train.csv")
TEST_PATH = get_data_path("test.csv")
SAMPLE_SUBMISSION_PATH = get_data_path("sample_submission.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")

# Model serialization paths
PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PIPELINE_SAVE_PATH = os.path.join(MODEL_DIR, "full_pipeline.joblib")

# Column classifications
TARGET_COL = "demand"
INDEX_COL = "Index"

CAT_COLS = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
NUM_COLS = ["NumberofLanes", "Temperature"]
GEO_COL = "geohash"
TIME_COL = "timestamp"
DAY_COL = "day"

# Preprocessing configs
WEATHER_CATEGORIES = ["Sunny", "Rainy", "Foggy", "Snowy"]
ROAD_CATEGORIES = ["Residential", "Street", "Highway"]

# Peak hours for Bengaluru Traffic (8 AM - 11 AM, 5 PM - 8 PM)
PEAK_HOURS = [(8, 11), (17, 20)]

# Model hyperparameters and grids
# Fast Grid (for quick local training on MacBook)
FAST_MODEL_GRIDS = {
    "LinearRegression": {
        "fit_intercept": [True]
    },
    "RandomForest": {
        "n_estimators": [10, 20],
        "max_depth": [5, 8],
        "random_state": [42],
        "n_jobs": [-1]
    },
    "XGBoost": {
        "n_estimators": [20, 50],
        "max_depth": [3, 5],
        "learning_rate": [0.1],
        "random_state": [42],
        "n_jobs": [-1]
    },
    "LightGBM": {
        "n_estimators": [20, 50],
        "max_depth": [3, 5],
        "learning_rate": [0.1],
        "random_state": [42],
        "n_jobs": [-1],
        "verbose": [-1]
    }
}

# Full Grid (for teammate running on GPU/heavier machine)
FULL_MODEL_GRIDS = {
    "LinearRegression": {
        "fit_intercept": [True, False]
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, 20],
        "min_samples_split": [2, 5],
        "random_state": [42],
        "n_jobs": [-1]
    },
    "XGBoost": {
        "n_estimators": [100, 300],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "random_state": [42],
        "n_jobs": [-1]
    },
    "LightGBM": {
        "n_estimators": [100, 300],
        "max_depth": [6, 8, 10],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "random_state": [42],
        "n_jobs": [-1],
        "verbose": [-1]
    }
}
