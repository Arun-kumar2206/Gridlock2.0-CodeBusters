"""
Model definition, training, validation, hyperparameter tuning, and selection.
Supports Baseline Linear Regression, Random Forest, XGBoost, and LightGBM.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.config import FAST_MODEL_GRIDS, FULL_MODEL_GRIDS
from src.utils import setup_logger

logger = setup_logger("models")

def get_model_and_grid(model_name, fast_mode=True):
    """
    Returns the raw model object and its corresponding hyperparameter grid.
    
    Args:
        model_name (str): Name of the model ('LinearRegression', 'RandomForest', 'XGBoost', 'LightGBM').
        fast_mode (bool): If True, returns a smaller hyperparameter grid for fast execution.
        
    Returns:
        tuple: (estimator, param_grid)
    """
    grid_source = FAST_MODEL_GRIDS if fast_mode else FULL_MODEL_GRIDS
    
    if model_name == "LinearRegression":
        model = LinearRegression()
        grid = grid_source["LinearRegression"]
    elif model_name == "RandomForest":
        model = RandomForestRegressor()
        grid = grid_source["RandomForest"]
    elif model_name == "XGBoost":
        model = XGBRegressor()
        grid = grid_source["XGBoost"]
    elif model_name == "LightGBM":
        model = LGBMRegressor()
        grid = grid_source["LightGBM"]
    else:
        raise ValueError(f"Unknown model name: {model_name}")
        
    return model, grid

def evaluate_predictions(y_true, y_pred):
    """
    Calculates evaluation metrics: RMSE, MAE, and R^2.
    
    Args:
        y_true (np.array): Ground truth target values.
        y_pred (np.array): Predicted target values.
        
    Returns:
        dict: RMSE, MAE, R2 metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def train_and_tune_model(model_name, X_train, y_train, X_val, y_val, fast_mode=True, cv=3):
    """
    Trains and tunes a single model type using GridSearchCV.
    
    Args:
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        fast_mode (bool): Whether to run in fast mode.
        cv (int): Number of cross-validation folds.
        
    Returns:
        tuple: (best_model, validation_metrics, training_time_seconds)
    """
    logger.info(f"--- Training and Tuning: {model_name} (fast_mode={fast_mode}) ---")
    estimator, param_grid = get_model_and_grid(model_name, fast_mode)
    
    start_time = time.time()
    
    # Grid search cross-validation
    # We use RMSE (neg_mean_squared_error) as the scoring metric
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        verbose=1 if not fast_mode else 0,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    training_time = time.time() - start_time
    logger.info(f"{model_name} best parameters: {grid_search.best_params_}")
    logger.info(f"{model_name} training time: {training_time:.2f} seconds")
    
    # Predict and evaluate on validation set
    y_pred = best_model.predict(X_val)
    # Clip predictions to target range [0.0, 1.0] since demand cannot be negative
    y_pred = np.clip(y_pred, 0.0, 1.0)
    
    metrics = evaluate_predictions(y_val, y_pred)
    logger.info(f"{model_name} validation results: RMSE={metrics['RMSE']:.5f}, MAE={metrics['MAE']:.5f}, R2={metrics['R2']:.5f}")
    
    return best_model, metrics, training_time

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plots and saves feature importance for tree-based models or coefficients for linear models.
    
    Args:
        model: Trained model.
        feature_names (list): List of feature names.
        save_path (str, optional): File path to save the generated plot.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            title = "Feature Importance"
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
            title = "Feature Coefficient Magnitude (Linear Model)"
        else:
            logger.warning("Model does not support feature importance or coefficients.")
            return None
            
        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=feat_imp.head(15), palette="viridis")
        plt.title(f"Top 15 Features - {title}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100)
            logger.info(f"Feature importance plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
        return feat_imp
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        return None
