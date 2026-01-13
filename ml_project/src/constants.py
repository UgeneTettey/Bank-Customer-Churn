RAW_DATA_DIR = "data/"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

DATA_FILENAME = "customers.csv"
TARGET_COL = "Exited"

TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

IRRELEVANT_COLS = ["RowNumber", "CustomerId", "Surname"]