"""
End-to-end Machine Learning pipeline orchestrator for training and inference.
Saves and loads unified pipeline packages using joblib.
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.config import (
    TRAIN_PATH, TEST_PATH, PIPELINE_SAVE_PATH, TARGET_COL, INDEX_COL, MODEL_DIR
)
from src.preprocessing import TrafficFeatureEngineer
from src.models import train_and_tune_model, plot_feature_importance, evaluate_predictions
from src.utils import setup_logger

logger = setup_logger("pipeline")

class TrainingPipeline:
    """
    Orchestrates the model training, tuning, selection, and serialization process.
    """
    
    def __init__(self, data_path=TRAIN_PATH, fast_mode=True):
        self.data_path = data_path
        self.fast_mode = fast_mode
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None
        self.feature_names = []
        
    def run(self):
        """
        Runs the training pipeline:
        1. Loads train.csv.
        2. Splits data temporally: Train (Day 48) and Validation (Day 49).
        3. Fits feature engineering pipeline.
        4. Trains and tunes multiple regression models.
        5. Selects the best performing model based on validation RMSE.
        6. Generates feature importance plots.
        7. Serializes the final pipeline artifact.
        
        Returns:
            dict: Summary metrics of the best model.
        """
        logger.info(f"Starting training pipeline. Data source: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training data not found at {self.data_path}")
            
        # 1. Load Data
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded training dataset with shape: {df.shape}")
        
        # 2. Split data temporally: Train (Day 48) & Val (Day 49)
        train_df = df[df["day"] == 48].copy()
        val_df = df[df["day"] == 49].copy()
        
        # Fallback if days are differently structured (e.g. customized train.csv)
        if len(val_df) == 0 or len(train_df) == 0:
            logger.warning("Temporal split failed. Performing 80-20 random split.")
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
            
        if self.fast_mode:
            # Subsample for extremely fast runs on MacBook
            train_sample_size = min(10000, len(train_df))
            val_sample_size = min(2000, len(val_df))
            train_df = train_df.sample(n=train_sample_size, random_state=42)
            val_df = val_df.sample(n=val_sample_size, random_state=42)
            logger.info(f"Fast Mode Enabled: Subsampled train to {len(train_df)} rows and val to {len(val_df)} rows.")
            
        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]
        X_val = val_df.drop(columns=[TARGET_COL])
        y_val = val_df[TARGET_COL]
        
        logger.info(f"Training partition size: {X_train.shape}")
        logger.info(f"Validation partition size: {X_val.shape}")
        
        # 3. Fit Preprocessor
        self.preprocessor = TrafficFeatureEngineer()
        X_train_proc = self.preprocessor.fit_transform(X_train, y_train)
        X_val_proc = self.preprocessor.transform(X_val)
        
        self.feature_names = list(X_train_proc.columns)
        logger.info(f"Feature engineering generated {len(self.feature_names)} features: {self.feature_names}")
        
        # 4. Train and Evaluate Models
        models_to_train = ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"]
        results = {}
        
        for name in models_to_train:
            try:
                # Run tuning
                cv_folds = 2 if self.fast_mode else 3
                model, metrics, duration = train_and_tune_model(
                    model_name=name,
                    X_train=X_train_proc,
                    y_train=y_train,
                    X_val=X_val_proc,
                    y_val=y_val,
                    fast_mode=self.fast_mode,
                    cv=cv_folds
                )
                results[name] = {
                    "model": model,
                    "metrics": metrics,
                    "duration": duration
                }
            except Exception as e:
                logger.error(f"Error training {name}: {e}", exc_info=True)
                
        if not results:
            raise RuntimeError("All models failed to train.")
            
        # 5. Automatically Select Best Model based on Validation RMSE
        best_name = min(results.keys(), key=lambda k: results[k]["metrics"]["RMSE"])
        best_package = results[best_name]
        
        self.best_model = best_package["model"]
        self.best_model_name = best_name
        self.best_metrics = best_package["metrics"]
        
        logger.info("*" * 50)
        logger.info(f"AUTOMATIC SELECTION: Best Model is {self.best_model_name}")
        logger.info(f"Best Validation Metrics: RMSE={self.best_metrics['RMSE']:.5f}, MAE={self.best_metrics['MAE']:.5f}, R2={self.best_metrics['R2']:.5f}")
        logger.info("*" * 50)
        
        # 6. Generate and Save Feature Importance Plot
        plot_path = os.path.join(MODEL_DIR, "feature_importance.png")
        plot_feature_importance(self.best_model, self.feature_names, save_path=plot_path)
        
        # 7. Package and Serialize pipeline
        pipeline_package = {
            "preprocessor": self.preprocessor,
            "model": self.best_model,
            "model_name": self.best_model_name,
            "metrics": self.best_metrics,
            "feature_names": self.feature_names,
            "trained_at": pd.Timestamp.now().isoformat(),
            "fast_mode": self.fast_mode
        }
        
        os.makedirs(os.path.dirname(PIPELINE_SAVE_PATH), exist_ok=True)
        joblib.dump(pipeline_package, PIPELINE_SAVE_PATH)
        logger.info(f"Unified pipeline package successfully serialized to {PIPELINE_SAVE_PATH}")
        
        return self.best_metrics


class InferencePipeline:
    """
    Orchestrates the scoring process of unlabelled datasets using a pre-trained pipeline.
    """
    
    def __init__(self, pipeline_path=PIPELINE_SAVE_PATH):
        self.pipeline_path = pipeline_path
        self.preprocessor = None
        self.model = None
        self.model_name = None
        self.feature_names = []
        
    def load_pipeline(self):
        """
        Loads the pre-trained pipeline package.
        """
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline package not found at {self.pipeline_path}. Run training first.")
            
        package = joblib.load(self.pipeline_path)
        self.preprocessor = package["preprocessor"]
        self.model = package["model"]
        self.model_name = package["model_name"]
        self.feature_names = package["feature_names"]
        logger.info(f"Loaded unified pipeline: Model={self.model_name}, Features Count={len(self.feature_names)}")
        return package
        
    def predict(self, test_df):
        """
        Runs feature engineering and batch inference on the test DataFrame.
        
        Args:
            test_df (pd.DataFrame): Input test features.
            
        Returns:
            np.array: Model predictions (clipped to range [0.0, 1.0]).
        """
        if self.preprocessor is None or self.model is None:
            self.load_pipeline()
            
        logger.info("Executing preprocessing and feature extraction on input data...")
        # Preprocess test set using the fitted preprocessor
        test_proc = self.preprocessor.transform(test_df)
        
        # Ensure feature alignment
        missing_feats = set(self.feature_names) - set(test_proc.columns)
        if missing_feats:
            logger.warning(f"Re-creating missing features: {missing_feats}")
            for col in missing_feats:
                test_proc[col] = 0.0
                
        # Reorder columns to match the training feature alignment exactly
        test_proc = test_proc[self.feature_names]
        
        logger.info(f"Running predictions using model: {self.model_name}...")
        y_pred = self.model.predict(test_proc)
        
        # Post-processing: Clip predictions to regression boundaries [0.0, 1.0]
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        return y_pred
        
    def run_file(self, input_csv_path=TEST_PATH, output_csv_path=None):
        """
        Performs inference on a CSV file and saves the submission file.
        
        Args:
            input_csv_path (str): Path to input test CSV.
            output_csv_path (str, optional): Path to output prediction CSV.
            
        Returns:
            pd.DataFrame: Formatted submission DataFrame.
        """
        logger.info(f"Running batch prediction on file: {input_csv_path}")
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input file not found at {input_csv_path}")
            
        test_df = pd.read_csv(input_csv_path)
        logger.info(f"Loaded test file shape: {test_df.shape}")
        
        # Record raw indices
        if INDEX_COL in test_df.columns:
            indices = test_df[INDEX_COL]
        else:
            indices = test_df.index
            
        # Run inference
        y_pred = self.predict(test_df)
        
        # Build submission DataFrame
        submission = pd.DataFrame({
            INDEX_COL: indices,
            "demand": y_pred
        })
        
        if output_csv_path:
            submission.to_csv(output_csv_path, index=False)
            logger.info(f"Predictions saved to: {output_csv_path}")
            
        return submission
