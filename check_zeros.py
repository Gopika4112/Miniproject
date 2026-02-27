import pandas as pd

paths = [
    "dataset/hospital_1.csv",
    "dataset/hospital_2.csv",
    "dataset/hospital_3.csv"
]

for path in paths:
    print(f"\nChecking: {path}")
    df = pd.read_csv(path)

    for col in df.columns:
        zero_count = (df[col] == 0).sum()
        percent = (zero_count / len(df)) * 100
        print(f"{col}: {zero_count} zeros ({percent:.2f}%)")