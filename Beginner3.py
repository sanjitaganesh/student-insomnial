# importing the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import os

# Reading the dataset
dataset = pd.read_csv("Student Insomnia and Educational Outcomes Dataset_version-2.csv")


print("Dataset Loaded Successfully")
print(dataset.head())

# preprocessing
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

for col in dataset.columns:
    if dataset[col].dtype=='object':
        dataset[col]=le.fit_transform(dataset[col])

print("\nData Types after encoding:")
print(dataset.dtypes)

# Splitting the dataset into training set and testing set
X = dataset.iloc[:, [3, 6, 7, 10, 11, 12, 13]]
y = dataset.iloc[:, 14]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create an object of the algorithm / model
from sklearn.linear_model import LinearRegression
model=LinearRegression()

#Train the model
model.fit(X_train,y_train)
print("\nModel training completed")

# predict the model on testing dataset
y_pred=model.predict(X_test)

#mean absolute error scores
from sklearn.metrics import r2_score,mean_absolute_error
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

print("\n MODEL RESULTS")
print("R² Score:", round(r2,3))
print("MAE:", round(mae,3))

#Exploratory Data Analysis(EDA)
#Visualising the training set data

#create directory for plots
import os
os.makedirs("eda_plots",exist_ok=True)

#sleep vs performance
plot.figure()
plot.scatter(dataset.iloc[:, 3], dataset.iloc[:, 14])
plot.title("Sleep Hours vs Academic Performance")
plot.xlabel("Sleep Hours")
plot.ylabel("Academic Performance")
plot.savefig("eda_plots/sleep_vs_performance.png")
plot.show()

#stress vs performace
plot.figure()
plot.scatter(dataset.iloc[:, 13], dataset.iloc[:, 14])
plot.title("Stress vs Academic Performance")
plot.xlabel("Stress")
plot.ylabel("Academic Performance")
plot.savefig("eda_plots/stress_vs_performance.png")
plot.show()


print("\n EDA plots saved to 'eda_plots/'")



