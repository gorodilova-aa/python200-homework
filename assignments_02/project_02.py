# -----------------------------------------------------------------------
# Mini-Project — Predicting the Daily High
# -----------------------------------------------------------------------

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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


""" Task 2: Look at the Features """
print("\n----- Choosing Features -----\n ")

# 1. Print df.describe()
print("Basic summary for the dataset:\n")
print(df.describe())

# Result: From the summary we see there are no missing values

# 2. Print the correlation of each numeric column with temperature_2m_max, sorted
correlation = (
    df.corr(numeric_only=True)["temperature_2m_max"]
    .sort_values(ascending=False)
)
print("\nCorrelation of each numeric column with High temperature:\n")
print(correlation)

# Which feature relates most strongly to the daily high?
# - low temperature (temperature_2m_min) strogly correlated: 0.958553
# - precipitation and wind_speed have low correlation (~0.07 and ~-0.05 respectively)

# 3. Create at least one plot that helps you understand the data 

plt.figure(figsize=(10, 7))
scatter = plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"], edgecolors="k", alpha=0.8)
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.savefig("outputs/Novosibirsk_high_vs_low.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n Scatter plot saved to outputs/Novosibirsk_high_vs_low.png\n")

# add a comment describing what you see
# - The scatter plot high vs low reveals a tight, linear relationship between daily low and daily high temperatures.
# - As the minimum temperature rises, the maximum temperature increases proportionally 


""" Task 3: Task 3: Engineer a Feature """
print("\n----- Engineering a Feature -----\n ")

# Add a binary is_summer column:
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

# Print the number of summer and non-summer days
print("summer and non-summer days counts:\n")
print(df["is_summer"].value_counts())

# do you expect is_summer to help predict the daily high, and why?
# - probably yes. Summer months typicaly have higher temperatures due to longer daytime.


""" Task 4: Baseline Model """
print("\n----- Baseline Model -----\n ")

# gettinng X and y for a model
X = df[["temperature_2m_min"]].values
y = df[["temperature_2m_max"]].values

# Split the data into training and test sets, an 80/20 split, and random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split complete.\n")

# Fit a LinearRegression model to training data 
model_baseline = LinearRegression()
model_baseline.fit(X_train, y_train)
print("Model fitting complete.\n")

# predict on the test set 
y_pred = model_baseline.predict(X_test)
print("\nPredictions made.\n")

# Compute and print RMSE and R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model_baseline.score(X_test, y_test)

print("\tSlope:", model_baseline.coef_[0])
print("\tTest RMSE:", rmse)
print("\tTest R²:", r2)
print("Tests complete.\n")

# the RMSE is in degrees Celsius. In plain English, how far off is a typical prediction?
# - The RMSE represents the typical standard deviation of the residuals. 
# - This means the baseline model's predicted high temperature is typically off from the actual high
# - by roughly the value of the RMSE (around 4.20).


""" Task 5: Full Model """
print("\n----- Full Model -----\n ")

# gettinng X and y for a model
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]

X_full = df[feature_cols].values
y = df["temperature_2m_max"].values

# Split the data into training and test sets, an 80/20 split, and random_state=42
X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
print("Data split complete.\n")

# Fit a LinearRegression model to training data 
model_full = LinearRegression()
model_full.fit(X_full_train, y_train)
print("Model fitting complete.\n")

# test and predict on the test set 
y_pred_train = model_full.predict(X_full_train)
y_pred_test = model_full.predict(X_full_test)
print("\nPredictions made.\n")

# Compute and print RMSE and R²

r2_train = model_full.score(X_full_train, y_train)
r2_test = model_full.score(X_full_test, y_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\tSlope:", model_full.coef_[0])
print("\tTest RMSE:", rmse_test)
print("\tTest R² for train:", r2_train)
print("\tTest R² for test:", r2_test)
print("Tests complete.\n")

# Compare the test R² to your baseline from Task 4. How much does adding features help?
# - R2 slightly increased from 0.9298728258246624 to 0.9301586969120015

print("Each feature name with its coefficient:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# interpreting the coefficients:
# - temperature_2m_min (+1.012): Raises the predicted high. For each 1C low temp increase, high also rises by ~1C
# - precipitation_sum (-0.239):  Lowers the predicted high. Rainy days dropping the high by ~0.24 C per unit of rain.
# - wind_speed_10m_max (+0.044): Slightly raises the predicted high, though the magnitude is very small 
#   and close to zero, suggesting wind speed has a minor effect on peak daily temperatures.
# - is_summer (+2.087): Raises the predicted high. Being in June, July, or August adds ~2.09°C to the 
#   expected daily high due to prolonged daylight.

# Are any signs surprising?
# The positive sign on wind_speed_10m_max is somewhat surprising, as wind is often intuitively associated 
# with cooling breezes or storm fronts; however, its near-zero value shows it plays a negligible role.
#
# Train R² vs Test R²:
# - Train R²: ~0.916
# - Test R²:  ~0.930
# The train and test R² scores are very close (with test even slightly higher), showing virtually no gap. 
# This confirms the model generalizes well to unseen data and is not overfitting.

""" Task 6: Evaluate and Summarize """
print("\n----- Evaluate and Summarize -----\n ")


# Predicted on the x-axis, actual on the y-axis
plt.figure(figsize=(10, 7))
plt.scatter(y_pred_test, y_test, color="steelblue", edgecolors="k", alpha=0.8)

# Diagonal reference line (where predicted == actual)
min_val = min(y_pred_test.min(), y_test.min())
max_val = max(y_pred_test.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="diagonal reference line (predicted = actual)")

plt.title("Predicted vs Actual for Novosibirsk")
plt.xlabel("Predicted Daily High (°C)")
plt.ylabel("Actual Daily High (°C)")
plt.legend()
plt.savefig("outputs/predicted_vs_actual_high.png", dpi=150, bbox_inches="tight")
plt.close()

print("Scatter plot Predicted vs Actual saved to outputs/predicted_vs_actual.png\n")

# Does the model struggle more at the high end, the low end, or is the error roughly even across the range? 
# - The points are distributed reasonably evenly along the diagonal reference line across the full temperature range. 
# - The model does not appear to struggle at the extreme low or extreme high end, its even more close to real values at the ends.

# -----------------------------------------------------------------------
# ---------- Summary ----------------------------------------------------
# -----------------------------------------------------------------------

# 1. The size of the dataset and the test set:
#    - Dataset contains 365 daily observations (2023 full calendar year),
#    - with an 80/20 train/test split resulting in 292 training days and 73 test days.
#
# 2. The RMSE and R² of your full model, and what a typical error of that size means on a temperature scale:
#    - the full model achieved a test R² of ~0.930 and a test RMSE of ~4.199 °C.
#    - On a temperature scale, an RMSE of ~4.199 °C means that the model's predicted
#    - daily maximum temperature is typically off from the actual recorded high by roughly 4.2 °C.
#
# 3. Which feature has the largest effect on the predicted high, and in which direction?
#    - 'is_summer' (+2.087) contributes the largest impact lifting predicted highs by ~2.1 °C during summer months.
#    
# 4. Surprising result:
#    - 'wind_speed_10m_max' had an almost negligible positive coefficient (+0.044).
#    - binary feature "is_summer" has the most impact.