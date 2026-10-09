# -----------------------------------------------------------------------
# Internal domain models and transformations for weather records.
# -----------------------------------------------------------------------

from dataclasses import dataclass
from weatherkit import WeatherResponse


# Why is HourlyReading a dataclass rather than a Pydantic model, when WeatherResponse is a Pydantic model?
# - WeatherResponse is at external boundary, where raw, untrusted data enters the application.
# - Pydantic applies runtime validation, coercion, and structural integrity (e.g., array lengths and bounds).
# - After this validation at the boundary, HourlyReading operates internally as domain data. 
# - At this stage the data is guaranteed to be clean, so a lightweight standard-library dataclass avoids unnecessary validation and runs more efficiently.


@dataclass
class HourlyReading:
    """Single hourly weather measurement.

    Args:
        timestamp: ISO-8601 string representation of observation time.
        temperature_c: Temperature measured in degrees Celsius (°C).
        precipitation_mm: Total precipitation measured in millimeters (mm).
    Returns:
        An instance of HourlyReading representing the weather observation for the specified hour.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a columnar API hourly data into a sequential list of HourlyReading objects.

    Iterates through the parallel arrays within the response's hourly block,
    preserving the original chronological sequence.

    Args:
        response: A validated WeatherResponse instance containing an HourlyBlock.

    Returns:
        A list of HourlyReading instances ordered chronologically.
    """
    hourly = response.hourly
    return [
        HourlyReading(
            timestamp=t,
            temperature_c=temp,
            precipitation_mm=precip,
        )
        for t, temp, precip in zip(
            hourly.time,
            hourly.temperature_2m,
            hourly.precipitation,
            strict=True,
        )
    ]