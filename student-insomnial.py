# Imports
import numpy as np
import pandas as pd
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

# Renaming columns
sleep_col = "1. How many hours do you sleep on average per night?"
study_col = "2. How many hours do you study per day?"
stress_col = "3. How stressed do you usually feel?"
performance_col = (
    "15. How would you rate your overall academic performance "
    "(GPA or grades) in the past semester?"
)

# Ordinal Encoding
performance_map = {
    "Poor": 0,
    "Below Average": 1,
    "Average": 2,
    "Good": 3,
    "Excellent": 4
}

dataset[performance_col] = dataset[performance_col].map(performance_map)

# Renaming those Columns
dataset = dataset.rename(columns={
    sleep_col: "SleepHours",
    study_col: "StudyHours",
    stress_col: "StressLevel",
    performance_col: "AcademicPerformance"
})


# Drop rows with missing values
dataset = dataset.dropna(subset=[
    "SleepHours", "StudyHours", "StressLevel", "AcademicPerformance"
])

print("\nData types after encoding:")
print(dataset[["SleepHours", "StudyHours", "StressLevel", "AcademicPerformance"]].dtypes)

# Feature Selection
X = dataset[["SleepHours", "StudyHours", "StressLevel"]]
y = dataset["AcademicPerformance"]

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model(Linear Regression used as Ordinal Regression)
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
plt.ylabel("Academic Performance")
plt.title("Sleep Hours vs Academic Performance")
plt.savefig("eda_plots/sleep_vs_performance.png")
plt.close()

# Stress vs Performance
plt.figure()
plt.scatter(dataset["StressLevel"], dataset["AcademicPerformance"])
plt.xlabel("Stress Level")
plt.ylabel("Academic Performance")
plt.title("Stress Level vs Academic Performance")
plt.savefig("eda_plots/stress_vs_performance.png")
plt.close()

print("\nEDA plots saved to 'eda_plots/'")
