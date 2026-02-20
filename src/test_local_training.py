import pandas as pd
import numpy as np

from model_tf import build_model
from train_tf import train_local, evaluate, predict_risk

TARGET_COL = "Outcome"

HOSPITAL_TRAIN_PATH = "dataset/hospital_1.csv"
TEST_PATH = "dataset/test_set.csv"

def load_xy(csv_path: str):
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {csv_path}")

    X = df.drop(TARGET_COL, axis=1).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    cols = df.drop(TARGET_COL, axis=1).columns.tolist()
    return X, y, cols

def main():
    # Load training data for one hospital
    X_train, y_train, train_cols = load_xy(HOSPITAL_TRAIN_PATH)

    # Load test set
    X_test, y_test, test_cols = load_xy(TEST_PATH)

    # Ensure same feature columns
    if train_cols != test_cols:
        print("Column mismatch between hospital train file and test set!")
        print("Train columns:", train_cols)
        print("Test columns :", test_cols)
        return

    input_dim = X_train.shape[1]
    print(f"Input features = {input_dim}")
    print("Features:", train_cols)

    # Build model
    model = build_model(input_dim)

    # Train locally
    '''
    print("\nTraining local model on Hospital 1 data...")
    train_local(model, X_train, y_train, epochs=15, batch_size=32)
    print("Training completed!")
    '''
    hospital_name = HOSPITAL_TRAIN_PATH.split("/")[-1]

    print(f"\nTraining local model on {hospital_name}...")
    train_local(model, X_train, y_train, epochs=15, batch_size=32)
    print("Training completed!")

    # Evaluate
    metrics = evaluate(model, X_test, y_test)
    print("\nEvaluation Metrics on test_set.csv:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Predict risk for one test sample
    risk_pct, level = predict_risk(model, X_test[0])
    print("\nSample Risk Prediction (first test row):")
    print(f"Risk: {risk_pct:.2f}% | Category: {level}")

if __name__ == "__main__":
    main()
