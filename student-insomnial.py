# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# Load Dataset
DATA_PATH = "Student Insomnia and Educational Outcomes Dataset_version-2.csv"
dataset = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully\n")
print(dataset.head())
print("\nCOLUMNS:")
print(dataset.columns)


# Column Names 
sleep_col = "4. On average, how many hours of sleep do you get on a typical day?"
fatigue_col = (
    "8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?"
)
stress_col = "14. How would you describe your stress levels related to academic workload?"
performance_col = (
    "15. How would you rate your overall academic performance (GPA or grades) in the past semester?"
)

# Ordinal Encoding Maps
frequency_map = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4
}

stress_map = {
    "Very Low": 0,
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Very High": 4
}

performance_map = {
    "Poor": 0,
    "Below Average": 1,
    "Average": 2,
    "Good": 3,
    "Excellent": 4
}


# Encoding
dataset[fatigue_col] = dataset[fatigue_col].map(frequency_map)
dataset[stress_col] = dataset[stress_col].map(stress_map)
dataset[performance_col] = dataset[performance_col].map(performance_map)


# Rename Columns
dataset = dataset.rename(columns={
    sleep_col: "SleepHours",
    fatigue_col: "DaytimeFatigue",
    stress_col: "StressLevel",
    performance_col: "AcademicPerformance"
})


# Drop Missing Values
dataset = dataset.dropna(subset=[
    "SleepHours",
    "DaytimeFatigue",
    "StressLevel",
    "AcademicPerformance"
])

print("\nData types after encoding:")
print(dataset[[
    "SleepHours",
    "DaytimeFatigue",
    "StressLevel",
    "AcademicPerformance"
]].dtypes)


# Feature Selection
X = dataset[["SleepHours", "DaytimeFatigue", "StressLevel"]]
y = dataset["AcademicPerformance"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model (Ordinal Regression via Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed")


# Prediction and Evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nMODEL RESULTS")
print("R² Score:", round(r2, 3))
print("MAE:", round(mae, 3))


# EDA Plots
os.makedirs("eda_plots", exist_ok=True)

# Sleep vs Performance
plt.figure()
plt.scatter(dataset["SleepHours"], dataset["AcademicPerformance"])
plt.xlabel("Sleep Hours")
plt.ylabel("Academic Performance (Ordinal)")
plt.title("Sleep Hours vs Academic Performance")
plt.savefig("eda_plots/sleep_vs_performance.png")
plt.close()

# Stress vs Performance
plt.figure()
plt.scatter(dataset["StressLevel"], dataset["AcademicPerformance"])
plt.xlabel("Stress Level")
plt.ylabel("Academic Performance (Ordinal)")
plt.title("Stress Level vs Academic Performance")
plt.savefig("eda_plots/stress_vs_performance.png")
plt.close()

# Fatigue vs Performance
plt.figure()
plt.scatter(dataset["DaytimeFatigue"], dataset["AcademicPerformance"])
plt.xlabel("Daytime Fatigue")
plt.ylabel("Academic Performance (Ordinal)")
plt.title("Daytime Fatigue vs Academic Performance")
plt.savefig("eda_plots/fatigue_vs_performance.png")
plt.close()

print("\nEDA plots saved to 'eda_plots/'")
