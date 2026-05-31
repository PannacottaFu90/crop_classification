import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_random_forest(
    X, y, n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
):
    """Train a Random Forest classifier and return the fitted model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        oob_score=True,
    )
    model.fit(X, y)
    return model


def evaluate_classifier(model, X_test, y_test):
    """Return evaluation metrics for a trained classifier."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def predict_stack(model, feature_stack):
    """Predict a classification map from a stacked feature array."""
    if feature_stack.ndim != 3:
        raise ValueError("feature_stack must have shape (bands, rows, cols)")
    rows, cols = feature_stack.shape[1], feature_stack.shape[2]
    feature_array = feature_stack.reshape(feature_stack.shape[0], -1).T
    prediction = model.predict(feature_array)
    return prediction.reshape(rows, cols).astype(int)


def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path):
    return joblib.load(model_path)


def save_checkpoint(data, output_path, compress=3):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(data, output_path, compress=compress)


def load_checkpoint(checkpoint_path):
    return joblib.load(checkpoint_path)
