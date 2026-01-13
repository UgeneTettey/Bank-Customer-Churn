# model prediction script for new data entries

import joblib
import pandas as pd
from src.constants import FEATURE_COLUMNS

model_path = "models/logistic_regression_model.pkl"

# load model
def load_model():
    model = joblib.load(model_path)
    return model

# predict
def predict(input_data: dict):
    model = load_model()

    # ensure all columns are present
    for col in FEATURE_COLUMNS:
        if col not in input_data:
            input_data[col] = 0  # or some default value
    # create DataFrame        
    df = pd.DataFrame([input_data])
    df = df[FEATURE_COLUMNS]

    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    return int(predictions[0]), float(probabilities[0][1])  # return class and probability of positive class