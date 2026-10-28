
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from the same folder
file_path = Path(__file__).with_name(
    "week3_logistics_hypothetical_dataset.csv"
)
df = pd.read_csv(file_path)

print("WEEK 3: LOGISTICS DATA ANALYSIS")
print("=" * 40)

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe(numeric_only=True))

print("\nDelivery status counts:")
print(df["Delivery_Status"].value_counts())

print("\nAverage delivery time by traffic:")
print(
    df.groupby("Traffic_Level")["Delivery_Time_Hours"]
    .mean().round(2)
)

print("\nAverage delivery time by vehicle:")
print(
    df.groupby("Vehicle_Type")["Delivery_Time_Hours"]
    .mean().round(2)
)

# Create output folder for charts
output_dir = Path(__file__).with_name("week3_outputs")
output_dir.mkdir(exist_ok=True)

# Chart 1: Delivery time distribution
plt.figure(figsize=(8, 5))
df["Delivery_Time_Hours"].hist(bins=15)
plt.title("Delivery Time Distribution")
plt.xlabel("Delivery Time (Hours)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig(output_dir / "delivery_time_distribution.png")
plt.close()

# Chart 2: Average delivery time by traffic
traffic_avg = df.groupby("Traffic_Level")[
    "Delivery_Time_Hours"
].mean()

traffic_avg.plot(kind="bar", figsize=(8, 5))
plt.title("Average Delivery Time by Traffic")
plt.xlabel("Traffic Level")
plt.ylabel("Average Hours")
plt.tight_layout()
plt.savefig(output_dir / "delivery_time_by_traffic.png")
plt.close()

# Chart 3: Distance vs delivery time
plt.figure(figsize=(8, 5))
plt.scatter(
    df["Distance_km"],
    df["Delivery_Time_Hours"]
)
plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (Hours)")
plt.tight_layout()
plt.savefig(output_dir / "distance_vs_delivery_time.png")
plt.close()

print("\nAnalysis completed successfully!")
print("Charts saved in:", output_dir)
