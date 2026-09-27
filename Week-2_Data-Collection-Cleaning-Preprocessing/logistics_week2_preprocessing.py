import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the logistics dataset
data = pd.read_csv("logistics_data.csv")

print("Original Dataset")
print(data.head())

# Standardize categorical columns
categorical_cols = [
    "Traffic",
    "Weather",
    "Vehicle_Type",
    "Delay_Status"
]

for col in categorical_cols:
    data[col] = (
        data[col]
        .astype("string")
        .str.strip()
        .str.title()
    )

# Convert numerical columns into numeric data types
numeric_cols = [
    "Distance_km",
    "Workload",
    "Delivery_Cost"
]

for col in numeric_cols:
    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )

# Check missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Handle missing numerical values using median
data["Distance_km"] = data["Distance_km"].fillna(
    data["Distance_km"].median()
)

data["Workload"] = data["Workload"].fillna(
    data["Workload"].median()
)

data["Delivery_Cost"] = data["Delivery_Cost"].fillna(
    data["Delivery_Cost"].median()
)

# Handle missing categorical values
data["Weather"] = data["Weather"].fillna("Unknown")

# Remove duplicate records
print("\nDuplicate Records Before Cleaning:")
print(data.duplicated().sum())

data = data.drop_duplicates()

print("\nDuplicate Records After Cleaning:")
print(data.duplicated().sum())

# Outlier detection using IQR
Q1 = data["Distance_km"].quantile(0.25)
Q3 = data["Distance_km"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = data[
    (data["Distance_km"] < lower_limit) |
    (data["Distance_km"] > upper_limit)
]

print("\nDistance Outliers:")
print(outliers)

# Normalize numerical columns using Min-Max Scaling
scaler = MinMaxScaler()

data[numeric_cols] = scaler.fit_transform(
    data[numeric_cols]
)

# Final validation
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

print("\nFinal Dataset Information:")
print(data.info())

print("\nFirst Five Preprocessed Records:")
print(data.head())

# Save the preprocessed dataset
data.to_csv(
    "logistics_preprocessed.csv",
    index=False
)

print("\nPreprocessing completed successfully.")
print("Preprocessed dataset saved as logistics_preprocessed.csv")
