"""
Preprocessing and Feature Engineering pipeline.
Implements a Scikit-Learn compatible transformer class.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import PEAK_HOURS, CAT_COLS, NUM_COLS, GEO_COL, TIME_COL, DAY_COL
from src.utils import decode_geohash, setup_logger

logger = setup_logger("preprocessing")

class TrafficFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that executes the entire preprocessing and feature engineering
    workflow for traffic and travel demand prediction.
    """
    
    def __init__(self):
        self.median_temp = None
        self.mode_road = None
        self.mode_weather = None
        self.geohash_target_map = {}
        self.global_mean_demand = 0.0
        
        # Categorical mappings
        self.large_vehicles_map = {"Allowed": 1, "Not Allowed": 0}
        self.landmarks_map = {"Yes": 1, "No": 0}
        self.road_type_map = {"Residential": 0, "Street": 1, "Highway": 2, "Unknown": 3}
        self.weather_map = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3, "Unknown": 4}
        
    def fit(self, X, y=None):
        """
        Fits the preprocessor on the training data. Learns imputation statistics and target mappings.
        
        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series, optional): Training labels (target demand). Required for target encoding.
            
        Returns:
            self
        """
        logger.info("Fitting TrafficFeatureEngineer...")
        df = X.copy()
        
        # 1. Learn numerical imputations
        self.median_temp = df["Temperature"].median()
        if pd.isna(self.median_temp):
            self.median_temp = 16.4  # Fallback if all values are NaN
            
        # 2. Learn categorical imputations
        self.mode_road = df["RoadType"].mode()[0] if not df["RoadType"].mode().empty else "Residential"
        self.mode_weather = df["Weather"].mode()[0] if not df["Weather"].mode().empty else "Sunny"
        
        # 3. Learn target encoding for geohash
        if y is not None:
            df["target_demand"] = y
            self.global_mean_demand = y.mean()
            # Calculate smoothed target encoding: (group_sum + global_mean * smooth) / (group_count + smooth)
            smooth = 10
            geohash_stats = df.groupby("geohash")["target_demand"].agg(["count", "mean"])
            smoothed_values = (
                (geohash_stats["count"] * geohash_stats["mean"] + self.global_mean_demand * smooth) / 
                (geohash_stats["count"] + smooth)
            )
            self.geohash_target_map = smoothed_values.to_dict()
            logger.info(f"Fitted target encoding for {len(self.geohash_target_map)} geohashes.")
        else:
            self.global_mean_demand = 0.0939  # Default train mean
            
        logger.info("Fitting completed successfully.")
        return self
        
    def transform(self, X):
        """
        Applies feature engineering and imputation to a given DataFrame.
        
        Args:
            X (pd.DataFrame): Input features.
            
        Returns:
            pd.DataFrame: Transformed features.
        """
        df = X.copy()
        
        # 1. Fill missing values
        df["Temperature"] = df["Temperature"].fillna(self.median_temp)
        df["RoadType"] = df["RoadType"].fillna("Unknown")
        df["Weather"] = df["Weather"].fillna("Unknown")
        
        # 2. Extract Temporal Features
        # Parse timestamp: e.g. "13:45"
        hours_col = []
        minutes_col = []
        for ts in df["timestamp"]:
            try:
                parts = ts.split(":")
                hours_col.append(int(parts[0]))
                minutes_col.append(int(parts[1]))
            except Exception:
                # Fallbacks in case of corrupted timestamp
                hours_col.append(12)
                minutes_col.append(0)
                
        df["hour"] = hours_col
        df["minute"] = minutes_col
        df["time_of_day_fraction"] = df["hour"] + df["minute"] / 60.0
        
        # Cyclical encoding
        df["sin_time"] = np.sin(2 * np.pi * df["time_of_day_fraction"] / 24.0)
        df["cos_time"] = np.cos(2 * np.pi * df["time_of_day_fraction"] / 24.0)
        
        # Peak Hour Indicator
        is_peak = np.zeros(len(df), dtype=int)
        for start, end in PEAK_HOURS:
            is_peak = is_peak | ((df["hour"] >= start) & (df["hour"] <= end)).astype(int)
        df["is_rush_hour"] = is_peak
        
        # Day of week / weekend indicator
        # Assuming day is integer, we use day % 7 to get weekday index
        df["day_of_week"] = df["day"] % 7
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)  # 5, 6 as weekend
        
        # 3. Extract Geospatial Features
        # Decode geohash to lat and lon
        latitudes = []
        longitudes = []
        for gh in df["geohash"]:
            lat, lon = decode_geohash(gh)
            if lat is None:
                # Default to central Bengaluru coordinates if invalid
                latitudes.append(12.9716)
                longitudes.append(77.5946)
            else:
                latitudes.append(lat)
                longitudes.append(lon)
        df["latitude"] = latitudes
        df["longitude"] = longitudes
        
        # 4. Encode Categorical Columns
        # LargeVehicles mapping (Allowed / Not Allowed)
        df["LargeVehicles_encoded"] = df["LargeVehicles"].map(self.large_vehicles_map).fillna(0).astype(int)
        
        # Landmarks mapping (Yes / No)
        df["Landmarks_encoded"] = df["Landmarks"].map(self.landmarks_map).fillna(0).astype(int)
        
        # RoadType mapping
        df["RoadType_encoded"] = df["RoadType"].map(self.road_type_map).fillna(3).astype(int)
        
        # Weather mapping
        df["Weather_encoded"] = df["Weather"].map(self.weather_map).fillna(4).astype(int)
        
        # 5. Geohash Target Encoding Feature
        df["geohash_encoded"] = df["geohash"].map(self.geohash_target_map).fillna(self.global_mean_demand)
        
        # 6. Interaction Features
        # Lane intensity: Lanes * vehicle permission
        df["traffic_intensity"] = df["NumberofLanes"] * (1 + df["LargeVehicles_encoded"])
        
        # Temperature deviation
        df["temp_deviation"] = df["Temperature"] - 16.4
        
        # 7. Drop original raw columns to prepare for model input
        cols_to_drop = ["geohash", "timestamp", "RoadType", "LargeVehicles", "Landmarks", "Weather"]
        # Index is kept for tracking or dropped inside the modeling script. We drop it here if it exists.
        if "Index" in df.columns:
            cols_to_drop.append("Index")
            
        df = df.drop(columns=cols_to_drop, errors="ignore")
        
        return df

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for the output of transform.
        """
        return [
            "day", "NumberofLanes", "Temperature", "hour", "minute", 
            "time_of_day_fraction", "sin_time", "cos_time", "is_rush_hour", 
            "day_of_week", "is_weekend", "latitude", "longitude", 
            "LargeVehicles_encoded", "Landmarks_encoded", "RoadType_encoded", 
            "Weather_encoded", "geohash_encoded", "traffic_intensity", "temp_deviation"
        ]
