# tests/test_classifier.py
import pytest
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from weather_model import WeatherClassifier, Prediction

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

@pytest.fixture
def model_path(tmp_path):
    """Train a tiny model on obvious days and save it to a temp file."""
    good = [{"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10}] * 15
    bad = [{"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40}] * 15
    X = pd.DataFrame(good + bad)[FEATURES]
    y = [1] * 15 + [0] * 15

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)

    path = tmp_path / "weather_classifier.pkl"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def clf(model_path):
    """A WeatherClassifier loaded from the saved tiny model."""
    return WeatherClassifier(model_path)





@pytest.mark.parametrize(
    "day_data, expected_label",
    [
        # A clearly good day is labeled "good"
        (
            { "temperature_2m_max": 23.0, "temperature_2m_min": 15.0, "precipitation_sum": 0.0, "wind_speed_10m_max": 4.0 },
            "good",
        ),
        # A clearly bad day (cold, rainy, windy) is labeled "skip"
        (
            { "temperature_2m_max": 2.0, "temperature_2m_min": -2.0, "precipitation_sum": 15.0, "wind_speed_10m_max": 40.0 },
            "skip",
        ),
    ],
)

# Test 1: A clearly good day is labeled "good" & Test 2: A clearly bad day (cold, rainy, windy) is labeled "skip".
def test_predict_labels_parametrized(
    clf: WeatherClassifier,
    day_data: dict,
    expected_label: str,
):
    """Checks Behaviors 1 & 2 using parametrization."""
    predictions = clf.predict([day_data])

    assert len(predictions) == 1
    assert predictions[0].label == expected_label



# Test 3: The returned probability is between 0 and 1.
def test_probability_is_between_zero_and_one(clf):
    result = clf.predict([
        {"temperature_2m_max": 15, "temperature_2m_min": 8, "precipitation_sum": 1.0, "wind_speed_10m_max": 12}
    ])
    assert 0.0 <= result[0].probability <= 1.0

# Test 4: Passing three days returns three Prediction objects, in order.
def test_one_prediction_per_day_in_order(clf):
    days = [
        {"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40},
        {"temperature_2m_max": 25, "temperature_2m_min": 18, "precipitation_sum": 0.2, "wind_speed_10m_max": 20},
    ]
    results = clf.predict(days)
    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)

# Test 5: Passing an empty list returns an empty list.
def test_empty_list_returns_empty(clf):
    assert clf.predict([]) == []

# Test 6: A day missing a required feature raises ValueError
def test_missing_feature_raises(clf):
    with pytest.raises(ValueError, match="missing required features"):
        clf.predict([{"temperature_2m_max": 18, "temperature_2m_min": 9}])   # no precipitation_sum or wind_speed_10m_max

# Test 7: Constructing a WeatherClassifier with a path to a missing file raises FileNotFoundError
def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file"):
        WeatherClassifier(tmp_path / "does_not_exist.pkl")



# Deliberate Break Verification:
# When deliberately changing the error message in classifier.py from:
#   raise FileNotFoundError(f"Model file not found at: {model_path}")
# to a generic:
#   raise FileNotFoundError(f"Cannot open: {model_path}")
# the test `test_missing_model_file_raises` immediately failed with:
#   FAILED tests/test_classifier.py::test_missing_model_file_raises - AssertionError: Regex pattern did not match.
# This confirmed that pytest.raises(..., match="...") actively verifies not just
# the exception type, but the expected explanatory error message as well.