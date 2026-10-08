# -----------------------------------------------------------------------
# Aggregation and summarization for hourly weather readings.
# -----------------------------------------------------------------------

from collections import defaultdict
from dataclasses import dataclass
import json
from weatherkit.schemas import WeatherResponse
from weatherkit.records import HourlyReading, to_readings

@dataclass
class DailySummary:
    """Aggregated weather summary for a single day."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the temperature range for the day."""
        return self.temp_max - self.temp_min

class DailyAggregator:
    """Utility class to aggregate hourly readings into daily summaries."""

    def __init__(self, min_hours: int = 24) -> None:
        """Initialize the aggregator with a minimum hours threshold."""
        self.min_hours = min_hours

    def _group_by_date(self, readings: list[HourlyReading]) -> dict[str, list[HourlyReading]]:
        """Helper method to group hourly readings by the first 10 characters of the timestamp."""
       
        grouped_by_date_HourlyReading: dict[str, list[HourlyReading]] = defaultdict(list)
        
        for reading in readings:
            date_key = reading.timestamp[:10]
            grouped_by_date_HourlyReading[date_key].append(reading)
        return grouped_by_date_HourlyReading

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Aggregate hourly readings into daily summaries."""

        grouped_by_date_HourlyReading = self._group_by_date(readings)
        summaries: list[DailySummary] = []

        for date_key in sorted(grouped_by_date_HourlyReading.keys()):
            day_readings = grouped_by_date_HourlyReading[date_key]
            if len(day_readings) < self.min_hours:
                continue

            temps = [r.temperature_c for r in day_readings]
            precip = sum(r.precipitation_mm for r in day_readings)

            summary = DailySummary(
                date=date_key,
                temp_max=max(temps),
                temp_min=min(temps),
                precipitation_sum=round(precip, 4),
                hours_observed=len(day_readings),
            )
            summaries.append(summary)

        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return a list of dates that have fewer than the minimum required hours of observations."""
        grouped_by_date_HourlyReading = self._group_by_date(readings)
        incomplete: list[str] = []
        for date_key, day_readings in grouped_by_date_HourlyReading.items():
            if len(day_readings) < self.min_hours:
                incomplete.append(date_key)
        return incomplete


