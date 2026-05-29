"""
Root configuration import wrapper.
Exposes settings from src/config.py to root execution files.
"""

from src.config import *
import src.config as src_config

print(f"Loaded configuration. Train Path: {TRAIN_PATH}")
