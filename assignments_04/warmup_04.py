
import json
from pathlib import Path
from dataclasses import dataclass
import pytest

# ------------------------------------------------------------------------------------
# --- Warmup Question 1 --- 
print("\n ---  Warmup Question 1 ---\n")
# ------------------------------------------------------------------------------------


# a class ModelInfo that reads a JSON file once in __init__ and stores its content
class ModelInfo:
    def __init__(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        return len(self.data["features"])


# Create a small JSON file with a "features" list 
sample_data = {
    "model_name": "weather_classifier",
    "features": [
        "temperature_high",
        "temperature_low",
        "precipitation",
        "wind_speed",
        "humidity"
    ],
}

sample_path = Path("sample_model_info.json")
with open(sample_path, "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=2)

# Build ModelInfo and display the feature count
model_info = ModelInfo(sample_path)
print(f"Feature count: {model_info.feature_count()}")

# why is loading the file in __init__ better than loading it inside feature_count() if the method is called many times?
# - Loading the file in __init__ reads and parses the JSON from disk exactly once,
# - while loading it inside feature_count() would read and parse the file every time the method is called, which is inefficient.


# ------------------------------------------------------------------------------------
# --- Warmup Question 2 --- 
print("\n ---  Warmup Question 2 ---\n")
# -----------------------------------------------------------------------------------

@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain_str = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c:.1f}C, {rain_str}"


# Create two DayForecast objects
forecast_1 = DayForecast(date="2026-10-27", high_c=25.2, will_rain=False)
forecast_2 = DayForecast(date="2026-10-28", high_c=20.5, will_rain=True)

# Print each summary
print(forecast_1.summary())
print(forecast_2.summary())

# name one thing the @dataclass decorator wrote for you that you did not have to write by hand.
# - We did not have to write `def __init__(self, date, high_c, will_rain):` 
# - and manually assign  `self.date = date`, `self.high_c = high_c`, and `self.will_rain = will_rain`.
# (It also automatically generated `__repr__` and `__eq__`).


# ------------------------------------------------------------------------------------
# --- Warmup Question 3 --- 
print("\n ---  Warmup Question 3 ---\n")
# ------------------------------------------------------------------------------------

def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Invalid temperature value: '{text}'")

# Pytest tests
def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)

def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")

# after running pytest, we get:
# warmup_04.py::test_parse_temperature_valid PASSED                                                                                                                                 [ 50%]
# warmup_04.py::test_parse_temperature_bad_input_raises PASSED  



# what would pytest.raises(ValueError) without match fail to catch?
# - It would fail to catch cases where a ValueError is raised for reasons other than an invalid temperature value

  