"""
Command-line interface to generate demand predictions on test data.
Usage:
  python predict.py [--input INPUT_PATH] [--output OUTPUT_PATH]
"""

import argparse
import sys
import os
import pandas as pd
from src.config import TEST_PATH, SUBMISSION_PATH, PIPELINE_SAVE_PATH
from src.pipeline import InferencePipeline
from src.utils import setup_logger

logger = setup_logger("predict_cli")

def main():
    parser = argparse.ArgumentParser(description="Generate traffic demand predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default=TEST_PATH,
        help="Path to input test CSV (defaults to config TEST_PATH)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=SUBMISSION_PATH,
        help="Path where output submission CSV will be saved (defaults to config SUBMISSION_PATH)."
    )
    
    args = parser.parse_args()
    
    logger.info("Initializing Inference Pipeline...")
    logger.info(f"Arguments: input={args.input}, output={args.output}")
    
    if not os.path.exists(PIPELINE_SAVE_PATH):
        logger.error("No trained model pipeline package found! Run train.py first.")
        print("\nERROR: Model pipeline not found. You must run training (train.py) before generating predictions.\n", file=sys.stderr)
        sys.exit(1)
        
    try:
        pipeline = InferencePipeline(pipeline_path=PIPELINE_SAVE_PATH)
        submission = pipeline.run_file(input_csv_path=args.input, output_csv_path=args.output)
        
        logger.info("Inference Completed Successfully!")
        
        # Verify submission format
        print("\n" + "="*50)
        print("PREDICTION STATUS: SUCCESS")
        print(f"Predictions saved to: {args.output}")
        print(f"Number of rows predicted: {len(submission)}")
        print("First 5 predictions:")
        print(submission.head())
        
        # Sanity check
        if submission.isna().any().any():
            logger.warning("WARNING: Submission contains null values!")
            print("WARNING: There are missing values in the predictions. Please check logs.")
        else:
            print("Sanity Check Passed: No missing values found in submission.csv")
            
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Inference failed with error: {e}", exc_info=True)
        print(f"\nERROR: Inference failed. Details: {e}\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
