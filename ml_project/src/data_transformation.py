# handles essentail data transformation and cleaning tasks to ensure the data is in a suitable format for modeling
# fix missing values, encode categorical variables, scale numerical features, convert data tyypes, etc.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.constants import TARGET_COL
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import set_config
import logging
logger = logging.getLogger(__name__)

# def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
#     """Fill missing values with sensible defaults."""
#     logger.info("Filling missing values...")
#     # if a column has missing values, fill them with the median (for numerical) or mode (for categorical)
#     for col in df.columns:
#         if df[col].dtype == 'object':
#             df[col].fillna(df[col].mode()[0], inplace=True)
#         else:
#             df[col].fillna(df[col].median(), inplace=True)
#     logger.info("Missing values filled")
#     return df


# impute missing values
set_config(transform_output='pandas')
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """fill missing values using simple imputation strategies."""
    # check if there are missing values
    logger.info("Checking for missing values...")
    if not df.isna().values.any():
        logger.info("No missing values found.")
        return df
    
    # if there are missing values, apply imputation
    logger.info("Filling missing values using SimpleImputer...")
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numerical_cols),
            ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols)
        ],
        remainder='passthrough',  # keep other columns unchanged
        verbose_feature_names_out=False  # to avoid prefixing column names. Keep original names
    )

    df = preprocessor.fit_transform(df)
    logger.info("Imputation complete. Missing values filled")
    return df




def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    logger.info("Encoding categorical columns...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    logger.info("Categorical encoding complete")
    return df

def scale_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features."""
    logger.info("Scaling numerical columns...")
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    logger.info("Numerical scaling complete")
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run all transformations in order."""
    logger.info("Starting data transformation...")
    df = fill_missing_values(df)
    df = encode_categorical(df)
    df = scale_numerical(df)
    # Ensure target column is of integer type
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    logger.info("Data transformation complete")
    return df
