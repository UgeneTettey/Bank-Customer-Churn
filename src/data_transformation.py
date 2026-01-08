# data preprocessing and transformation functions

def preprocess_data(df):
    """
    Preprocess the input DataFrame by handling missing values and encoding categorical variables.

    Parameters:
    df (pd.DataFrame): The input data to preprocess.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    return df