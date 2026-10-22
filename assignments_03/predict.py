import joblib
import json
import pandas as pd


""" Task 7:  Load and Predict """
print("\n-----  Load and Predict -----\n ")

# Load the model 
model = joblib.load("models/weather_classifier.pkl")

# load the metadata
metadata = json.load(open("models/weather_classifier_metadata.json"))

print("Loaded model metadata:")
print("Features:", metadata["features"])
print("City:", metadata["city"])
print("Test accuracy:", metadata["test_accuracy"])
print("Label:", metadata["label"])
print("Label rule:", metadata["label_rule"])


# Creating a sample input for prediction

# 
new_data = {
    "temperature_2m_max": [15,  30,   7,  25], 
    "temperature_2m_min": [5,   20,   0,  15],
    "precipitation_sum":  [0.1, 1.0,  2,  2.5],
    "wind_speed_10m_max": [5,   12,   25, 29]
}

new_df = pd.DataFrame(new_data)
print("\nNew data for prediction:\n")
print(new_df)


# 2. Get predictions and class probabilities
# (Assuming `model` is your fitted Pipeline containing StandardScaler + Classifier)
predictions = model.predict(new_df)
probabilities = model.predict_proba(new_df)

print("\nPredictions for new data:\n")

predictions_labels = ["skip" if p == 0 else "good" for p in predictions]

for i, (pred, prob) in enumerate(zip(predictions_labels, probabilities)):
    print(f"Day {i+1}: Predicted={pred}, Probabilities: {prob[0]:.3f} (skip), {prob[1]:.3f} (good)")

""" Task 8:  Reflect """
print("\n-----  Reflect -----\n ")

print("Reflection is given in a comment")


# 1. Borderline day:
#   - Selected Borderline Day: Day 3 (T_max=7, T_min=0, Precip=2.0, Wind=25).
#   - Probability: 0.444 (44.4%) for "good" and 0.556 (55.6%) for "skip".
#   - Confidence: The model is highly uncertain here close to a coin toss. 
# How would you handle a day where it predicts 0.52?
#   - I would treat it as a borderline case and consider additional context
#   - such as weather forecasts, user preferences, etc.
#   - We cound also implement a threshold (e.g., 0.6) for making a confident "good" prediction.
#   
#       
# 2. What would break if someone ran predict.py before train_classifier.py had ever been run? 
#   -  `predict.py` attempts to load the model via `joblib.load()`.
#   -  If `train_classifier.py` has not run yet, the file (e.g., `model.joblib`) does not exist on disk, triggering a `FileNotFoundError`.
#  How could you make the error message more helpful?
#  - Improving the error message: Wrap the loading logic in a `try...except FileNotFoundError` block
#  - (or use `os.path.exists()`), and raise a clear message such as:
#  - "FileNotFoundError: Model artifact not found. 
#  - Please run `python train_classifier.py` first to train and serialize the model before running predictions."

# 3. Cloud Pipeline Adaptation (Week 10):
#   - Current limitation: `predict.py` hardcodes hypothetical inputs directly in memory 
#     inside the script and prints human-readable strings to stdout.
#   - Required change: The script must dynamically fetch real-time input data from an external 
#    source (such as an API endpoint, message queue, or cloud database/S3 bucket) and 
#    serialize its output (e.g., return a JSON payload or write records to a database/storage) 
#     rather than relying on hardcoded dictionaries and CLI print statements.