from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Downloading UCI Heart Disease dataset...")

# Fetch the UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save dataset
df.to_csv("heart_disease.csv", index=False)

print("\nDataset saved successfully!")
print("Rows and columns:", df.shape)