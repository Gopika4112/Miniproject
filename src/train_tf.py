from typing import Dict, List, Tuple
import numpy as np
import tensorflow as tf

def train_local(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32
) -> None:
    """
    Train the model locally on one hospital dataset.
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def evaluate(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate on a test set and return loss, accuracy, precision, recall, f1.
    """
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    probs = model.predict(X_test, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    y_true = np.array(y_test).astype(int)

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def get_weights(model: tf.keras.Model) -> List[np.ndarray]:
    """
    Federated Learning: Extract weights (Flower-compatible).
    """
    return model.get_weights()

def set_weights(model: tf.keras.Model, weights: List[np.ndarray]) -> None:
    """
    Federated Learning: Set weights received from server.
    """
    model.set_weights(weights)

def predict_risk(model: tf.keras.Model, x_row: np.ndarray) -> Tuple[float, str]:
    """
    Predict probability -> risk percentage + risk category.
    x_row must already be preprocessed/scaled.
    """
    x_row = np.array(x_row, dtype=np.float32).reshape(1, -1)
    prob = float(model.predict(x_row, verbose=0)[0][0])
    risk_pct = prob * 100.0

    if risk_pct < 30:
        level = "Low Risk"
    elif risk_pct < 60:
        level = "Medium Risk"
    else:
        level = "High Risk"

    return risk_pct, level
