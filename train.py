"""
Command-line interface to train and tune traffic demand models.
Usage:
  python train.py [--fast] [--data_path DATA_PATH]
"""

import argparse
import sys
from src.config import TRAIN_PATH
from src.pipeline import TrainingPipeline
from src.utils import setup_logger

logger = setup_logger("train_cli")

def main():
    parser = argparse.ArgumentParser(description="Train traffic demand models.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode with subsampled data and smaller hyperparameter grid (ideal for MacBooks)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=TRAIN_PATH,
        help="Path to training CSV file (defaults to config TRAIN_PATH)."
    )
    
    args = parser.parse_args()
    
    logger.info("Initializing Training Pipeline...")
    logger.info(f"Arguments: fast={args.fast}, data_path={args.data_path}")
    
    try:
        pipeline = TrainingPipeline(data_path=args.data_path, fast_mode=args.fast)
        metrics = pipeline.run()
        
        logger.info("Training Completed Successfully!")
        print("\n" + "="*50)
        print("TRAINING STATUS: SUCCESS")
        print(f"Selected Model: {pipeline.best_model_name}")
        print("Validation Set Metrics:")
        for metric_name, val in metrics.items():
            print(f"  - {metric_name}: {val:.6f}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        print(f"\nERROR: Training failed. Details: {e}\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
