# -----------------------------------------------------
# Tests for weatherkit.schemas.
# -----------------------------------------------------

import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from weatherkit.schemas import WeatherResponse, HourlyBlock

# Note in a comment which test caught the change.
# - once we change  "if not (len_time == len_temp == len_precip):" to  "if not (len_time == len_temp):"
#   the test `test_mismatched_list_lengths_raises` failed.


def test_valid_response_loads():
    # why a plain relative path is unreliable?
    # - path like "weather_raw.json" depends on the working directory from which pytest is invoked. 
    # - Using Path(__file__).parent.parent anchors the path relative to this test file, making it robust no matter where pytest is executed.

    raw_path = Path(__file__).parent.parent / "weather_raw.json"
    with open(raw_path, "r") as f:
        data = json.load(f)

    response = WeatherResponse.model_validate(data)
    assert len(response.hourly.time) == 168

def test_latitude_out_of_bounds_raises():
    incorrect_data = {
        "latitude": 200.0,
        "longitude": 0.0,
        "timezone": "UTC",
        "elevation": 10.0,
        "hourly": {
            "time": ["2026-10-08T14:00"],
            "temperature_2m": [25.0],
            "precipitation": [0.0],
        },
    }
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(incorrect_data)

def test_mismatched_list_lengths_raises():
    incorrect_data = {
        "time": ["2026-10-08T12:00", "2026-10-08T13:00", "2026-10-08T14:00"],
        "temperature_2m": [25.0, 26.0, 27.0],  
        "precipitation": [0.0, 0.1], # missing one element
    }
    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(incorrect_data)


def test_null_in_temperature_raises():
    incorrect_data = {
        "time": ["2026-10-08T14:00"],
        "temperature_2m": [None],
        "precipitation": [0.5],
    }
    with pytest.raises(ValidationError):
        HourlyBlock.model_validate(incorrect_data)


