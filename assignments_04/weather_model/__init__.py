"""Weather Model Package.

Exposes the WeatherClassifier and Prediction dataclass 
for classifying running weather conditions.
"""

from weather_model.classifier import WeatherClassifier, Prediction

__all__ = ["WeatherClassifier", "Prediction"]