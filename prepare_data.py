import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

import config

# =============================
# LOAD DATA
# =============================
data = pd.read_csv(config.DATA_PATH)

X = data.drop(config.TARGET_COLUMN, axis=1)
y = data[config.TARGET_COLUMN]

# =============================
# STRATIFIED TRAIN-TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y if config.STRATIFY else None
)

# =============================
# FIX PIMA ZERO-AS-MISSING ISSUE
# =============================
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_invalid_cols:
    if col in X_train.columns:
        X_train.loc[X_train[col] == 0, col] = np.nan
        X_test.loc[X_test[col] == 0, col] = np.nan

# =============================
# IMPUTATION (FIT ONLY ON TRAIN)
# =============================
imputer = SimpleImputer(
    strategy=config.IMPUTATION_STRATEGY,
    add_indicator=config.ADD_MISSING_INDICATOR
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Handle column names
if config.ADD_MISSING_INDICATOR:
    columns = imputer.get_feature_names_out()
else:
    columns = X.columns

X_train = pd.DataFrame(X_train, columns=columns)
X_test = pd.DataFrame(X_test, columns=columns)

# =============================
# OPTIONAL FEATURE SELECTION
# =============================
if config.FEATURE_SELECTION:
    selector = SelectKBest(score_func=f_classif, k=config.FEATURE_SELECTION_K)

    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    selected_cols = columns[selector.get_support()]
    X_train = pd.DataFrame(X_train, columns=selected_cols)
    X_test = pd.DataFrame(X_test, columns=selected_cols)

# =============================
# SCALING (FIT ONLY ON TRAIN)
# =============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_df[config.TARGET_COLUMN] = y_train.values

test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_df[config.TARGET_COLUMN] = y_test.values

# =============================
# CREATE DATASET FOLDER
# =============================
os.makedirs("dataset", exist_ok=True)

# Save test set
test_df.to_csv("dataset/test_set.csv", index=False)

# =============================
# STRATIFIED HOSPITAL SPLIT
# =============================
skf = StratifiedKFold(
    n_splits=config.NUM_HOSPITALS,
    shuffle=True,
    random_state=config.RANDOM_STATE
)

for i, (_, idx) in enumerate(skf.split(X_train_scaled, y_train)):
    hospital_data = train_df.iloc[idx]
    hospital_data.to_csv(f"dataset/hospital_{i+1}.csv", index=False)

    print(f"Hospital {i+1} class distribution:")
    print(hospital_data[config.TARGET_COLUMN].value_counts(normalize=True))

# =============================
# SAVE PREPROCESSING OBJECTS
# =============================
joblib.dump(imputer, "dataset/imputer.pkl")
joblib.dump(scaler, "dataset/scaler.pkl")

if config.FEATURE_SELECTION:
    joblib.dump(selector, "dataset/selector.pkl")

print("✅ Fully optimized federated-ready datasets created successfully!")