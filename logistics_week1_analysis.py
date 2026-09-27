# Week 1 - Strategic Planning and Data Exploration in Logistics

import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

# Replace this filename with the actual dataset name when available
file_path = "logistics_data.csv"

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset file not found.")
    print("Upload the dataset and update the file name before running the program.")

# --------------------------------------------------
# 2. Basic Data Exploration
# --------------------------------------------------

if "df" in locals():
    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nColumn Information:")
    print(df.info())

    # --------------------------------------------------
    # 3. Missing Values
    # --------------------------------------------------

    print("\nMissing Values:")
    print(df.isnull().sum())

    # --------------------------------------------------
    # 4. Duplicate Records
    # --------------------------------------------------

    print("\nNumber of Duplicate Records:")
    print(df.duplicated().sum())

    # --------------------------------------------------
    # 5. Statistical Summary
    # --------------------------------------------------

    print("\nStatistical Summary:")
    print(df.describe(numeric_only=True))

    # --------------------------------------------------
    # 6. KPI Calculation
    # --------------------------------------------------

    if "Delivery_Time_hours" in df.columns:
        average_delivery_time = df["Delivery_Time_hours"].mean()
        print("\nAverage Delivery Time:",
              round(average_delivery_time, 2), "hours")

    if "Delivery_Cost" in df.columns:
        average_delivery_cost = df["Delivery_Cost"].mean()
        print("Average Delivery Cost:",
              round(average_delivery_cost, 2))

    if "Delay" in df.columns:
        delay_rate = (df["Delay"] == 1).mean() * 100
        on_time_rate = (df["Delay"] == 0).mean() * 100

        print("Delay Rate:",
              round(delay_rate, 2), "%")

        print("On-Time Delivery Rate:",
              round(on_time_rate, 2), "%")

    # --------------------------------------------------
    # 7. Cost per Kilometer
    # --------------------------------------------------

    if "Delivery_Cost" in df.columns and "Distance_km" in df.columns:
        total_cost = df["Delivery_Cost"].sum()
        total_distance = df["Distance_km"].sum()

        if total_distance != 0:
            cost_per_km = total_cost / total_distance

            print("Cost per Kilometer:",
                  round(cost_per_km, 2))

    # --------------------------------------------------
    # 8. Data Science Plan
    # --------------------------------------------------

    print("\nPlanned Data Science Techniques:")
    print("1. Descriptive Analytics")
    print("2. Regression")
    print("3. Classification")
    print("4. Clustering")
    print("5. Route Optimization")

    # --------------------------------------------------
    # 9. Project Conclusion
    # --------------------------------------------------

    print("\nWeek 1 Analysis Completed.")
    print("The next stage will focus on detailed EDA,")
    print("predictive modeling and logistics optimization.")
