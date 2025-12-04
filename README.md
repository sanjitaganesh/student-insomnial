# Student Insomnia & Academic Performance – ML Analysis

This beginner machine-learning project analyzes how sleep habits, stress levels, caffeine intake, device usage, exercise patterns, and fatigue impact academic performance in students.

The goal is to practice a full end-to-end ML workflow:
- Data cleaning & preprocessing
- Encoding categorical data
- Feature scaling
- Train/test splitting
- Model training
- Evaluation
- Simple EDA visualization

---

## Dataset

Source: Mendeley Data  
**Student Insomnia and Educational Outcomes Dataset (Version 2)**

Each record represents a student survey with features related to sleep patterns and academic outcomes.

---

## Features Used

- SleepHours  
- ConcentrationIssues  
- Fatigue  
- DeviceUsage  
- Caffeine  
- Exercise  
- Stress  

**Target Variable:**

- AcademicPerformance

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## Machine Learning Model

**Algorithm:** Linear Regression

**Preprocessing steps:**

- Encoded all categorical columns using LabelEncoder
- Standardized input features using StandardScaler
- Split dataset into:
  - 80% training
  - 20% testing

---

## Model Evaluation

Metrics used:

- **R² Score**
- **Mean Absolute Error (MAE)**

Example output:
```text
R² Score: 0.173
MAE: 0.508
