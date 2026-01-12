# src/model_training.py

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import logging
from src.constants import TARGET_COL

logger = logging.getLogger(__name__)

def train_model(df, save_model=True, models_dir="models"):
    """
    Train a logistic regression model on churn data.

    Returns:
        model, accuracy, confusion_matrix
    """

    # define features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # build and train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    logger.info("Model training complete")

    # model evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    classif_report = classification_report(y_test, predictions)

    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{classif_report}")

    # save model
    if save_model:
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "logistic_regression_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved at {model_path}")

    return model, accuracy, conf_matrix
