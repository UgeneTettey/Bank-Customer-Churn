# # this module is resonsible for loading data: get raw data from specified path

# import pandas as pd
# import os
# import logging
# from src.constants import RAW_DATA_DIR, DATA_FILENAME

# logger = logging.getLogger(__name__)

# def load_data() -> pd.DataFrame:
#     """
#     Load data from the raw data directory into a pandas DataFrame.

#     Returns:
#     pd.DataFrame: The loaded data as a pandas DataFrame.
#     """
#     file_path = os.path.join(RAW_DATA_DIR, DATA_FILENAME)
#     logger.info(f"Loading data from {file_path}")

#     try:
#         data = pd.read_csv(file_path)
#         return data
#     except FileNotFoundError:
#         logger.error(f"The file at {file_path} was not found.")
#         raise
#     except pd.errors.EmptyDataError:
#         logger.error("The file is empty.")
#         raise
#     except pd.errors.ParserError:
#         logger.error("There was a parsing error while reading the file.")
#         raise
#     logger.info("Data loaded successfully.")
#     return data


# # if __name__ == "__main__":
# #     # for testing purposes
# #     from src.utils import setup_logger
# #     setup_logger()
# #     df = load_data()
# #     print(df.head())




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
