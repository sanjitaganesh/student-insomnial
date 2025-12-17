# importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# Load dataset
dataset = pd.read_csv(
    "Student Insomnia and Educational Outcomes Dataset_version-2.csv"
)

print("Dataset Loaded Successfully")
print(dataset.head())

# Rename important columns
dataset.rename(columns={
    "1. How many hours do you usually sleep per night?": "SleepHours",
    "2. How many hours do you study per day?": "StudyHours",
    "13. On a scale of 1-10, how stressed do you usually feel due to academics?": "StressLevel",
    "15. How would you rate your overall academic performance (GPA or grades) in the past semester?":
        "AcademicPerformance"
}, inplace=True)

# Ordinal encoding
performance_map = {
    "Poor": 0,
    "Below Average": 1,
    "Average": 2,
    "Good": 3,
    "Excellent": 4
}

dataset["AcademicPerformance"] = dataset["AcademicPerformance"].map(performance_map)

print("\nData Types after encoding:")
print(dataset[["SleepHours", "StudyHours", "StressLevel", "AcademicPerformance"]].dtypes)

# Feature selection
X = dataset[["SleepHours", "StudyHours", "StressLevel"]]
y = dataset["AcademicPerformance"]

# Train-test splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model: Linear Regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed")

# Prediction & Evaluation
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nMODEL RESULTS")
print("R² Score:", round(r2, 3))
print("MAE:", round(mae, 3))

# EDA-Plots
os.makedirs("eda_plots", exist_ok=True)

# Sleep vs Performance
plt.figure()
plt.scatter(dataset["SleepHours"], dataset["AcademicPerformance"])
plt.xlabel("Sleep Hours")
plt.ylabel("Academic Performance")
plt.title("Sleep Hours vs Academic Performance")
plt.savefig("eda_plots/sleep_vs_performance.png")
plt.show()

# Stress vs Performance
plt.figure()
plt.scatter(dataset["StressLevel"], dataset["AcademicPerformance"])
plt.xlabel("Stress Level")
plt.ylabel("Academic Performance")
plt.title("Stress Level vs Academic Performance")
plt.savefig("eda_plots/stress_vs_performance.png")
plt.show()

print("\nEDA plots saved to 'eda_plots/'")
