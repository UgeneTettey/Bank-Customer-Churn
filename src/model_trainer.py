# model training module
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train(X, y):
    """
    Splits the data into training and testing sets, then trains a Random Forest Classifier.

    Args:
        X (pd.DataFrame): Features dataset.
        y (pd.Series): Target variable.
    Returns:
        RandomForestClassifier: Trained Random Forest model.
        pd.DataFrame: Test features.
        pd.Series: Test target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model(X_train, y_train)
    return model, X_test, y_test



def build_model(X_train, y_train):
    """
    Trains a Random Forest Classifier on the provided training data.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable for training.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model