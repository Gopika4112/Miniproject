import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Load dataset
data = pd.read_csv("diabetes.csv")

print("Original shape:", data.shape)

# Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Combine back into dataframe
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
scaled_df["Outcome"] = y

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Split into 3 hospitals
hospital_1, temp = train_test_split(scaled_df, test_size=0.66, random_state=42)
hospital_2, hospital_3 = train_test_split(temp, test_size=0.5, random_state=42)

hospital_1.to_csv("dataset/hospital_1.csv", index=False)
hospital_2.to_csv("dataset/hospital_2.csv", index=False)
hospital_3.to_csv("dataset/hospital_3.csv", index=False)

print("Hospital datasets created successfully!")
