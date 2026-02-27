import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
# TRAIN TEST SPLIT (STRATIFIED)
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
# Only these columns have invalid zeros
zero_invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in zero_invalid_cols:
    if col in X_train.columns:
        X_train.loc[X_train[col] == 0, col] = np.nan
        X_test.loc[X_test[col] == 0, col] = np.nan

# =============================
# IMPUTATION (FIT ONLY ON TRAIN)
# =============================
imputer = SimpleImputer(
    strategy=config.IMPUTATION_STRATEGY,  # usually "median"
    add_indicator=config.ADD_MISSING_INDICATOR
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Update column names
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
# SAVE TEST SET
# =============================
os.makedirs("dataset", exist_ok=True)
test_df.to_csv("dataset/test_set.csv", index=False)

# =============================
# SPLIT TRAIN DATA FOR HOSPITALS
# =============================
hospital_data = train_df.sample(frac=1, random_state=config.RANDOM_STATE)

split_size = len(hospital_data) // config.NUM_HOSPITALS

for i in range(config.NUM_HOSPITALS):
    start = i * split_size
    end = (i + 1) * split_size if i < config.NUM_HOSPITALS - 1 else len(hospital_data)

    hospital_data.iloc[start:end].to_csv(
        f"dataset/hospital_{i+1}.csv",
        index=False
    )

print("✅ Accuracy-optimized datasets created successfully!")