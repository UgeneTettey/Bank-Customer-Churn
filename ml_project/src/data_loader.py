# src/data_loader.py

import os
import pandas as pd
import logging
from src.constants import RAW_DATA_DIR, DATA_FILENAME

logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """
    Load data from the raw data directory into a pandas DataFrame.
    The file path is automatically determined from constants.
    """
    file_path = os.path.join(RAW_DATA_DIR, DATA_FILENAME)
    logger.info(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    data = pd.read_csv(file_path)
    logger.info(f"Data loaded successfully with shape {data.shape}")
    return data
