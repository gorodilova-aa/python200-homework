import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------------------------
# --- The scikit-learn API --- 
print("\n ---  The scikit-learn API ---\n")
# ------------------------------------------------------------------------------------

""" scikit-learn Question 1 """

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

# Create model
model = LinearRegression()

# Fit model to the training data
model.fit(temp_min, temp_max)

# Predict the high for new low temperatures
new_temp_min = np.array([6, 18]).reshape(-1, 1)
temp_max_predictions = model.predict(new_temp_min)

# Print slope, intercept, and predictions with labels
print("\tscikit-learn Question 1\n")
print("Slope (coef_):", model.coef_[0])
print("Intercept:", model.intercept_)
print(f"Predicted high for low of {new_temp_min[0][0]}°: {temp_max_predictions[0]}")
print(f"Predicted high for low of {new_temp_min[1][0]}°: {temp_max_predictions[1]}")


""" scikit-learn Question 2 """
print("\tscikit-learn Question 2\n")

# Start with this 1D array
x = np.array([10, 20, 30, 40, 50])
print(f"x array: {x}")
print(f"shape of x array: {x.shape}")

# Convert it to a 2D array and print the new shape.
x = x.reshape(-1, 1)
print(f"reshaped x array: {x}")
print(f"shape of reshaped x array: {x.shape}")

""" 
    Why scikit-learn needs X to be 2D?
    - scikit-learn expects the input features as 2D arrays, 
      where each row represents a sample and each column represents a feature.
"""


# ------------------------------------------------------------------------------------
# --- Linear Regression --- 
print("\n ---  Linear Regression ---\n")
# ------------------------------------------------------------------------------------

# data creation

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)
print("Data creation is complete.\n")

""" Linear Regression Question 1 """
print("\tLinear Regression Question 1\n")

# looking at the data via a scatter plot

plt.figure(figsize=(10, 7))
scatter = plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm", edgecolors="k", alpha=0.8)
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.legend(scatter.legend_elements()[0], ["Dry", "Rainy"], title="Rain Status")
plt.savefig("outputs/high_vs_low.png", dpi=150, bbox_inches="tight")
plt.close()
print("Scatter plot saved to outputs/high_vs_low.png\n")


# Are there two visible groups? What does that suggest about the is_rainy variable?
#    - Yes, there are two visible groups whre one group placed higher than the other.
#    - This suggests that the is_rainy variable has a significant impact on the relationship between daily low and high temperatures.
#    - The plot shows a clear positive linear relationship between the variables within each group.
    

""" Linear Regression Question 2 """
print("\tLinear Regression Question 2\n")

# Reshape temp_min to a 2D array
X = temp_min.reshape(-1, 1)
y = temp_max

# Split the data into training and test sets using temp_min as the only feature, an 80/20 split, and random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print the shapes of all four arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("Data split complete.\n")

""" Linear Regression Question 3 """
print("\tLinear Regression Question 3\n")

# Fit a LinearRegression model to your training data from Question 2
model = LinearRegression()
model.fit(X_train, y_train)
print("Model fitting complete.\n")

# Print the slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# predict on the test set 
y_pred = model.predict(X_test)
print("Predictions made.\n")

# Compute and print RMSE and R²
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("\tTest RMSE:", rmse)
print("\tTest R²:", r2)
print("Tests complete.\n")

# a comment interpreting the slope; what does it mean for the daily high?
# - The slope is approximately 0.94 (close to 1.0).
# - This means that for every 1°C increase in the daily low temperature,
# - the daily high temperature is predicted to increase by roughly 1°C.


""" Linear Regression Question 4 """
print("\tLinear Regression Question 4\n")

# adding two features
X_full = np.column_stack([temp_min, is_rainy])

# split
X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
print("Data split complete, two features.\n")

# Fit a LinearRegression model 
model_full = LinearRegression()
model_full.fit(X_full_train, y_full_train)
print("Model fitting complete.\n")

# predict on the test set 
y_pred_full = model_full.predict(X_full_test)
print("Predictions made.\n") 

# Compute and print R²
r2 = model_full.score(X_full_test, y_full_test)
print("\tTest R²:", r2)
print("Test complete.\n")

# Does adding the rain flag help? 
# - For 1 feature we have Test R2 0.8944151516634623
# - For 2 features we get Test R2 0.945061816962249
# - So, yes, adding the rain flag helps significantly because it accounts for the temperature drop observed on overcast/rainy days.

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# What does is_rainy coefficient represent in practical terms?
# - s_rainy coefficient: -2.9190219583163612
# - Holding the low temperature constant, a rainy day (is_rainy = 1) is predicted to have a high temperature 
# - that is roughly 3°C cooler than a dry day (is_rainy = 0).


""" Linear Regression Question 5 """
print("\n\tLinear Regression Question 5\n")

# Predicted on the x-axis, actual on the y-axis
plt.figure(figsize=(10, 7))
plt.scatter(y_pred_full, y_full_test, color="steelblue", edgecolors="k", alpha=0.8)

# Diagonal reference line (where predicted == actual)
min_val = min(y_pred_full.min(), y_full_test.min())
max_val = max(y_pred_full.max(), y_full_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="diagonal reference line (predicted = actual)")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Daily High (°C)")
plt.ylabel("Actual Daily High (°C)")
plt.legend()
plt.savefig("outputs/predicted_vs_actual_warmup.png", dpi=150, bbox_inches="tight")
plt.close()

print("Scatter plot Predicted vs Actual saved to outputs/predicted_vs_actual_warmup.png\n")
 
# what does it mean when a point falls above the diagonal? What about below?
# - points above the line have actual > predicted. This means the model underpredicted the high (the real day was hotter).
# - points below the line have actual < predicted. This means the model overpredicted the high (the real day was cooler than expected).