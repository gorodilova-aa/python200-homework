import numpy as np
from sklearn.linear_model import LinearRegression

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
