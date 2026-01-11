from src.constants import TARGET_COL
def clean_data(df):
    df = df.copy()
    df = df.dropna(subset = [TARGET_COL])
    df = df.dropna()
    return df
print("Data cleaned successfully.")
