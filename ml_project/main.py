# main.py

import logging
from src.utils import setup_logger
from src.data_loader import load_data
from src.data_validation import validate_data
from src.data_transformation import transform_data
from src.model_training import train_model
from src.constants import TARGET_COL

logger = logging.getLogger(__name__)

def main():
    setup_logger()
    logger.info("Churn pipeline started")

    # 1. Load data
    df = load_data()
    logger.info(f"Data loaded successfully with shape {df.shape}")

    # 2. Validate data
    logger.info("Starting data validation")
    validate_data(df)
    logger.info("Data validation passed")

    # 3. Transform data
    logger.info("Starting data transformation")
    df = transform_data(df)
    logger.info("Data transformation completed")
    # print(df[TARGET_COL].dtype)  # Ensure target column is integer type

    # 4. Train model
    model, accuracy, classif_report = train_model(df)
    logger.info("Model training completed")

    print("\nPipeline completed successfully")
    print(f"Model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
