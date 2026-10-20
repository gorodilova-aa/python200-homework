import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("First 5 rows of the dataset:")
print(df.head())

# ------------------------------------------------------------------------------------
# --- Preprocessing --- 
print("\n ---  Preprocessing ---\n")
# ------------------------------------------------------------------------------------

""" Preprocessing Question 1 """

print("\tPreprocessing Question 1\n")
# Fit the StandardScaler on the training data and transform both the training and test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean and std from training data only
X_test_scaled  = scaler.transform(X_test)        # apply the same scaling to the test data

# Print the mean of each column of the scaled training data
print("Scaled training data column means:", X_train_scaled.mean(axis=0))
print("We see that the means are very close to 0, which is expected after standardization.\n")

# why you fit the scaler on X_train only?
# - We fit the scaler on X_train only to prevent data leakage from the test set into the training process, 
# - ensuring X_test acts as genuinely unseen data.


""" Preprocessing Question 2 """
print("\tPreprocessing Question 2\n")

# Fit the OneHotEncoder on the 'season' column and transform it
encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["season"]])
print(encoder.categories_[0])
print(encoded)

# Print array shape and category names
print("Encoded shape:", encoded.shape)
print("Category names:", encoder.get_feature_names_out())

# how many columns did the single season column become, and why is that number what it is?
# - The single 'season' column became 4 columns because there are 4 unique categories (spring, summer, fall, winter),
# - and OneHotEncoder creates one binary column for each distinct category.


""" Preprocessing Question 3 """
print("\tPreprocessing Question 3\n")


numeric_features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
categorical_features = ["season"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# preparing X and y for the pipeline (include season in your feature columns)
features_with_season = numeric_features + categorical_features
X_season = df[features_with_season]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_season, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train_s, y_train_s)      # fits the scaler, the encoder, AND the classifier
predictions = model.predict(X_test_s)   # applies the same preprocessing, then predicts

# print its test accuracy
print("Test Accuracy:", model.score(X_test_s, y_test_s))

# what does the pipeline do automatically that you would otherwise have to do by hand?
# - The pipeline automatically applies the same preprocessing steps to both the training and test data,
# - ensuring consistency in the data transformation process and preventing data leakage.


# ------------------------------------------------------------------------------------
# --- KNN --- 
print("\n ---  KNN ---\n")
# ------------------------------------------------------------------------------------

# setting the dataset for KNN
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["good_for_running"]

print(X.shape)
print(y.value_counts())

""" KNN Question 1 """
print("\tKNN Question 1\n")

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)

# Pipleine for KNN with scaling
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
print("\nScaled accuracy:  ", accuracy_score(y_test, knn_scaled.predict(X_test)))

print("Classification report for scaled KNN:")
y_pred = knn_scaled.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))


""" KNN Question 2 """
print("\tKNN Question 2\n")

for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

# Add a comment naming the k you would choose and why:
# - we would choose k=11:  mean=0.942  std=0.021
# - This k value has the highest mean accuracy and a relatively low standard deviation, 
# - indicating good performance and stability across different folds of the cross-validation.


""" KNN Question 3 """
print("\tKNN Question 3\n")

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
print("Unscaled accuracy:", accuracy_score(y_test, knn_unscaled.predict(X_test)))

knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
print("Scaled accuracy:  ", accuracy_score(y_test, knn_scaled.predict(X_test)))

# why would scaling matter for a distance-based model?
# - Scaling matters for distance-based models like KNN because these models rely on distance calculations 
# - between data points. If features are on different scales, those with larger ranges can dominate the distance metric,
# - like temperature_2m_max, which has a much larger range than precipitation_sum. 


# ------------------------------------------------------------------------------------
# --- Logistic Regression --- 
print("\n ---  Logistic Regression ---\n")
# ------------------------------------------------------------------------------------

# setting the dataset for Logistic Regression
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


""" Logistic Regression Question 1 """
print("\tLogistic Regression Question 1\n")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))


""" Logistic Regression Question 2 """
print("\tLogistic Regression Question 2\n")

logreg = model.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")

# which features push most strongly toward "skip," and does that match your intuition about running weather?
# - The features that push most strongly toward "skip" are precipitation_sum (-1.89) and wind_speed_10m_max (-2.44),
# -  as they have relativly large negative coefficients. 
# - This matches intuition about running weather, as higher precipitation and stronger winds are generally unfavorable conditions for running.


""" Logistic Regression Question 3 """
print("\tLogistic Regression Question 3\n")

print("The probability of 'good' for the first five test days:")
probabilities = model.predict_proba(X_test)[:, 1]   # probability of "good"
print(probabilities[:5].round(2))

# What does a probability near 0.5 tell you about the model's confidence for that day?
# - A probability near 0.5 indicates that the model is uncertain about the classification for that day.

# ------------------------------------------------------------------------------------
# --- Evaluation --- 
print("\n ---  Evaluation ---\n")
# ------------------------------------------------------------------------------------

""" Evaluation Question 1 """
print("\tEvaluation Question 1\n")

# Data
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot(cmap="Blues")

# Save the figure to outputs/logreg_confusion_matrix.png

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png", bbox_inches="tight")
plt.show()

print("Confusion matrix saved to outputs/logreg_confusion_matrix.png\n")

# how many false positives and how many false negatives did the model make, and which kind of error would matter more for a running app?
# - false positives: 7
# - false negatives: 9
# - In the context of a running app, false negatives would matter more 
# - because users might miss out on good running days due to incorrect predictions. 



# # ------------------------------------------------------------------------------------
# --- joblib --- 
print("\n ---  joblib ---\n")
# ------------------------------------------------------------------------------------

""" joblib Question 1 """
print("\tjoblib Question 1\n")


# Save the trained model to a file using joblib
joblib.dump(model, "models/weather_classifier.pkl")
print("Model saved.")

# Load the model back from the file and print its type
clf = joblib.load("models/weather_classifier.pkl")
print(f"Type of loaded model: {type(clf)}")

# comparing the loaded model's accuracy with the original model's accuracy
assert np.array_equal(model.predict(X_test), clf.predict(X_test))
print("Loaded model matches the original.")

# what would go wrong if you had saved only the LogisticRegression step without the scaler, then called .predict() on raw data?
# - If you saved only the LogisticRegression step without the scaler, the model would incorrectly interpret the raw data, leading to inaccurate predictions.

