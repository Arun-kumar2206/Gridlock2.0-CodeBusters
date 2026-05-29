"""
Utility functions for geohash decoding, logging, and directory setup.
"""

import os
import logging

def decode_geohash(geohash):
    """
    Decodes a geohash string into latitude and longitude coordinates (center of the bounding box).
    
    Args:
        geohash (str): The geohash string (e.g. 'qp02z1').
        
    Returns:
        tuple: (latitude, longitude) of the center point. Returns (None, None) if invalid.
    """
    if not isinstance(geohash, str) or not geohash:
        return None, None
        
    # Geohash base32 characters
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    char_to_bit = {char: i for i, char in enumerate(BASE32)}
    
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    
    is_even = True
    for char in geohash.lower():
        if char not in char_to_bit:
            return None, None
        val = char_to_bit[char]
        for mask in [16, 8, 4, 2, 1]:
            if is_even:  # Longitude
                mid = (lon_interval[0] + lon_interval[1]) / 2.0
                if val & mask:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:  # Latitude
                mid = (lat_interval[0] + lat_interval[1]) / 2.0
                if val & mask:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
            
    lat = (lat_interval[0] + lat_interval[1]) / 2.0
    lon = (lon_interval[0] + lon_interval[1]) / 2.0
    return lat, lon

def setup_logger(name="traffic_pipeline", log_file="pipeline.log", log_level=logging.INFO):
    """
    Sets up a logger that outputs to both a file and the console.
    
    Args:
        name (str): Name of the logger.
        log_file (str): Filename for the log output inside LOG_DIR.
        log_level (int): Log level (e.g. logging.INFO).
        
    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if logger already initialized
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    try:
        from src.config import LOG_DIR
        os.makedirs(LOG_DIR, exist_ok=True)
        file_path = os.path.join(LOG_DIR, log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file log handler: {e}")
        
    return logger
