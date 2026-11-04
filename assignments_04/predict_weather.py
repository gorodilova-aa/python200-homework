# predict_weather.py
"""Classify a few sample days using the saved weather model."""

from weather_model import WeatherClassifier

SAMPLE_DAYS = [
    {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12}, # clearly good,
    {"temperature_2m_max": 22, "temperature_2m_min": 13, "precipitation_sum": 9.0, "wind_speed_10m_max": 14}, # clearly bad
    {"temperature_2m_max": 25, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 33}, # borderline, high wind
    {"temperature_2m_max": 6,  "temperature_2m_min": 1,  "precipitation_sum": 0.1, "wind_speed_10m_max": 5},  # borderline, low temp
]


def main() -> None:
    """Load the component and print a prediction for each sample day."""
    
    print("\n-----  Predicting Running Weather Conditions for each sample day:-----\n")
    classifier = WeatherClassifier("models/weather_classifier.pkl")
    for day, result in zip(SAMPLE_DAYS, classifier.predict(SAMPLE_DAYS)):
        print(f"Max_t {day['temperature_2m_max']:.0f}C, Min_t {day['temperature_2m_min']:.0f}C, {day['precipitation_sum']:.0f}mm rain, "
              f"{day['wind_speed_10m_max']:.0f} km/h wind -> {result.label} ({result.probability:.2f})")

# Without this guard, any other script importing `predict_weather.py` (e.g., to reuse
# a helper function or sample data) would immediately execute `main()` as a side effect,
# redundantly loading the model, generating predictions, and printing to stdout upon import.
if __name__ == "__main__":
    main()


# Task 6: Reflection
"""
1. Feature order bug:
   Scikit-learn models rely entirely on positional array indices, not dictionary keys. 
   If a user passes a dictionary in a different order (e.g., precipitation first), 
   the model would treat rainfall (20.0 mm) as maximum temperature (20.0°C), silently 
   returning garbage predictions without raising an error.

2. Fixture vs. real model file:
   Building a tiny model in a fixture makes the tests self-contained, fast, and 
   independent of external files. Tests verify the component's interface and logic, 
   not whether a specific pre-trained .pkl file exists or changes.

3. Week 10 readiness & production:
   - Ready: It cleanly accepts a batch of raw dictionaries and returns structured 
     Prediction dataclass objects, completely hiding scikit-learn details.
   - To add for production: Schema/data validation (e.g., rejecting negative rain 
     or missing/None values) and logging for latency and prediction distribution.
"""