from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test data and returns various performance metrics.

    Args:
        model: Trained machine learning model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True target values for the test set."""
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# save model to models folder
import os
import joblib
from ml_project.src.constants import MODELS_DIR
def save_model(model, model_name="trained_model.pkl"):
    """
    Saves the trained model to the specified directory.

    Args:
        model: Trained machine learning model.
        model_name (str): Name of the file to save the model as.
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    model_path = os.path.join(MODELS_DIR, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")