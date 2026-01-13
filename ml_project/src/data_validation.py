# this handles data validation tasks: checking to ensure the data meets expected formats and constraints
import pandas as pd
from src.constants import TARGET_COL
import logging
logger = logging.getLogger(__name__)

required_columns = [
    "CustomerId",
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    TARGET_COL
]

def validate_data(df: pd.DataFrame):
    """
    Validate that the DataFrame contains all required columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to validate."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")
    logger.info("All required columns are present. Validation passed.")

