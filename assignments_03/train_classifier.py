# -----------------------------------------------------------------------
# Mini-Project — Build and Deploy a Weather Classifier
# -----------------------------------------------------------------------

import sys
import json

import requests
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib



""" Task 1: Fetch the Data """
print("\n----- Fetching the Data -----\n ")

# We choose the city Novosibirsk, Russia

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 55.01,
    "longitude": 82.56,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)


print("First 5 rows of weather dataset for Novosibirsk, Russia:\n")
print(df.head())

print("\n Shape of dataset:", df.shape)


""" Task 2:  Engineer the Label """
print("\n-----  Engineer the Label -----\n ")

# We enginering a new column called good_for_running, 
# which is a binary label indicating whether the weather conditions are suitable for running.

# good_for_running column is 1 when:
# - the high is between 7 and 26 degrees, 
# - the low is at least 0, 
# - precipitation is under 3 mm, 
# - the maximum wind is under 30 km/h.

df["is_good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3) &
    (df["wind_speed_10m_max"] < 30)
)


#  the class balance 
print("\nClass balance for is_good_for_running:\n")
print(df["is_good_for_running"].value_counts())

# what fraction of days are good, and does that seem reasonable for your city's climate? 
# If your data is very imbalanced, note which metrics will matter most because of it.
# - We see 258/365 days are not good for running.
# - We see 107/365 days are good for running.   
# - This seems reasonable for Novosibirsk, Russia, which has a cold climate with long snowy winters.


""" Task 3:  Build and Evaluate a KNN Classifier """
print("\n-----  Build and Evaluate a KNN Classifier -----\n ")


# set the dataset for the model

features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["is_good_for_running"]

print("\nFeatures (X):\n", X.head())
print("\nLabel (y):\n", y.head())       

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Pipeline for KNN with scaling
print("\nPipeline for KNN with scaling for different k values:")
      
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

print("\nChoosing k=9 for the final model as it has the highest cross-validation mean with low standard deviation.")

# choose k=9 for the final model
knn_final = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=9)),
])

# fit the model 
knn_final.fit(X_train, y_train)
print("\nModel fitting complete.")

print("\nScaled accuracy:  ", accuracy_score(y_test, knn_final.predict(X_test)))

print("\nClassification report for scaled KNN:")
y_pred = knn_final.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))



""" Task 4:  Build and Evaluate a Logistic Regression Classifier """
print("\n-----  Build and Evaluate a Logistic Regression Classifier -----\n ")


# set the dataset for the model

features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["is_good_for_running"]

print("\nFeatures (X):\n", X.head())
print("\nLabel (y):\n", y.head())       

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# Pipeline for Logistic Regression with scaling
logreg_final = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])


logreg_final.fit(X_train, y_train)
print("\nModel fitting complete.")

y_pred = logreg_final.predict(X_test)
print("\nClassification report for scaled Logistic Regression:")
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))

print("\nCoefficients for Logistic Regression:")
logreg = logreg_final.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# which conditions most strongly push a day toward "skip"?
# - The most negative coefficient is for precipitation_sum (-1.15), indicating that higher precipitation strongly pushes a day toward "skip".
# - The next most negative coefficient is for wind_speed_10m_max (-0.27), indicating that higher wind speed also pushes a day toward "skip".
# - Almost neutral is temperature_2m_max (-0.09), indicating that the maximum temperature has a minimal effect on the classification.
#
# - The only positive coefficient is for temperature_2m_min (+1.56), indicating that higher minimum temperatures push a day toward "good" for running.
# This aligns with the Siberian climate, where cold temperatures and precipitation are significant factors in determining suitable running conditions.


""" Task 5:  Compare and Choose """
print("\n-----  Compare and Choose -----\n ")

print("Compasion is given in a comment.\n")

# Short info about test set:
# - The test set contains 73 days:
# - 21 labeled as "good" 
# - 52 labeled as "skip".
# 
#  Which model had higher accuracy on your data?  
# - 0.92 for the KNN model had
# - 0.73 for the Logistic Regression model
#
# For a running app, is accuracy the metric you care about most, or would you weigh precision or recall more heavily?
# - For a running app, recall (skipping good days) is more important than precision (sending out on bad days). 
# - Especially in Siberia, where there are not so many days suitable for running, we want to make sure we don't miss any good days.
# - Of course, running on a bad day can be dangerous in Siberia, but you can always check the actual weather :) 
#
# If we focus on finding all the good days what we see:
# (recall for "good" class):
# - the KNN model can correctly find 86% of the good days 
# - compared to the Logistic Regression model that can find just 43%!
# (precision for "good" class):
# - the KNN model can correctly identify 86% of the days it predicts as good
# - while the Logistic Regression model can only correctly identify 53% of the days it predicts as good 
# !looks like LR just guesses or flip a coin for the good days, while KNN is much more reliable!
#
# Which model will you deploy, and why? 
# - I will deploy the KNN model because it has higher accuracy, recall, 
# - and precision for the "good" class, which is crucial for a running app.
# - Of course, the KNN model is more complex and require more computational resources, 
# - but the benefits of better performance in identifying good running days outweigh these costs. 
# Probably, we can try to tune the Logistic Regression model further (add more features in order to involve nonlianear relationships)
# but for now, the KNN model is the better choice for deployment.



""" Task 6:  Save the Model """
print("\n-----  Save the Model -----\n ")

# Save the trained KNN model to a file using joblib
joblib.dump(knn_final, "models/weather_classifier.pkl")
print("Model saved.")

# Save the metadata to a JSON file

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "features": features,
    "label": "good_for_running",
    "city": "Novosibirsk, Russia",
    "test_accuracy": 0.92,
    "test_f1": 0.86,
    "label_rule": {
        "temperature_2m_max": "7 to 26 C",
        "temperature_2m_min": ">= 0 C",
        "precipitation_sum": "< 3 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

with open("models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved.")