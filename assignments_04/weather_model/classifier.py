from dataclasses import dataclass
from pathlib import Path
from turtle import pd
from typing import Any
import joblib
import pandas as pd

@dataclass
class Prediction:
    """Represents the running prediction result based on weather conditions.

    Attributes:
        label (str):    Classification label, either "good" or "skip".
        probability (float):    Model confidence score from 0.0 to 1.0 indicating
                                the likelihood of conditions being good for running.
    """

    label: str
    probability: float



class WeatherClassifier:

    REQUIRED_FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
    
    def __init__(self, model_path: str | Path) -> None:
        """Loads the trained model pipeline from the given path.

        Args:
            model_path (str | Path): Path to the saved .pkl model pipeline file.

        Raises:
            FileNotFoundError: If no model file exists at model_path.
        """
        path = Path(model_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = joblib.load(path)

    def predict(self, days: list[dict[str, Any]]) -> list[Prediction]:
        """Predicts running weather conditions for a batch of days.

        Args:
            days (list[dict[str, Any]]): A list of dictionaries, each containing
                weather observations for a day.

        Returns:
            list[Prediction]: A list of Prediction objects corresponding to each
                day in the input list. Returns an empty list if days is empty.

        Raises:
            ValueError: If any required feature is missing from any of the day dicts.
        """
        # Return an empty list if days is empty.
        if not days:
            return []

        # Validate that all required features are present in every day dictionary
        for day in days:
            missing_features = [
                feature for feature in self.REQUIRED_FEATURES if feature not in day
            ]
            if missing_features:
                raise ValueError(
                    f"Day data is missing required features: {missing_features}"
                )

        # Select the four required features in a fixed order
        #features_ordered = [
        #    [day[feature] for feature in self.REQUIRED_FEATURES] for day in days
        #]

        features_ordered = pd.DataFrame(
            [[day[feat] for feat in self.REQUIRED_FEATURES] for day in days],
            columns=self.REQUIRED_FEATURES,
        )

        # Predict labels (0 or 1) and take probabilities of class 1 ("good")
        raw_predictions = self.model.predict(features_ordered)
        probabilities = self.model.predict_proba(features_ordered)[:, 1]

        results = []
        for pred, prob in zip(raw_predictions, probabilities):
            label = "good" if pred == 1 else "skip"
            results.append(Prediction(label=label, probability=float(prob)))

        return results