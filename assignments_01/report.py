# -------------------------------------------------------------------------------------------------
# Script to load weather_raw.json file and to validate it with WeatherResponse.model_validate(...).
# -------------------------------------------------------------------------------------------------

import json
from pathlib import Path
from weatherkit import DailyAggregator, to_readings, HourlyReading, WeatherResponse


def main() -> None:
    # 1. Load weather_raw.json
    raw_path = Path(__file__).parent / "weather_raw.json"
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 2. Validate it into a WeatherResponse
    response = WeatherResponse.model_validate(raw_data)

    # 3. Convert it to HourlyReading objects
    readings = to_readings(response)

    # 4. Aggregates to DailySummary objects
    aggregator = DailyAggregator(min_hours=24)
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    # 5. Prints a readable table
    print(f"Weather Summary for {response.timezone}")
    print(f"Latitude: {response.latitude}, Longitude: {response.longitude}\n")

    header = f"{'Date':<12} {'High (°C)':>10} {'Low (°C)':>10} {'Precip (mm)':>12} {'Range (°C)':>11}"
    print(header)
    print("-" * len(header))

    for day in summaries:
        print(
            f"{day.date:<12} {day.temp_max:>10.1f} {day.temp_min:>10.1f} "
            f"{day.precipitation_sum:>12.1f} {day.temp_range():>11.1f}"
        )

    # 6. Prints a warning line listing dropped incomplete days
    print()
    if incomplete:
        print(f"Warning: The following incomplete days were dropped: {', '.join(incomplete)}")
    else:
        print("Notice: No incomplete days were dropped.")


#  what would happen if you omitted the guard and someone imported report.py to reuse one of its helper functions?
# - If we omitted the `if __name__ == "__main__":` guard, any code outside of functions
# - (or direct calls to main()) would execute immediately when another module imports report.py.
#-  For example, if another script simply wanted to import a helper function from this file,
# - running that import would unintentionally load the JSON file, print tables, and run the whole report.

if __name__ == "__main__":
    main()


# --------------------------------------
# Task 7: Reflection
# --------------------------------------
"""
1. Rejecting null vs tolerating gaps:
   - When to reject: In safety-critical systems (like aviation alerts), where
     missing data is dangerous and failing early prevents corrupt reports.
   - When to tolerate: In long-term climate analysis, where discarding a full week
     of 168 hours over a single missing sensor reading is wasteful.
   - Schema change to tolerate it:
     Allow None values in HourlyBlock:
         temperature_2m: list[float | None]

2. min_hours=24 running at noon:
   - What goes wrong: At noon, only 12 hours exist, so summarize() silently drops
     today because it does not reach the 24-hour threshold.
   - How incomplete_days() helps: It lists the dropped dates so the pipeline can
     warn the user instead of letting current data disappear without notice.

3. Package vs single file in Week 10:
   - Cleaner imports: A cloud pipeline can import just the needed piece
     (e.g., `from weatherkit.schemas import WeatherResponse`) without running
     report script code or test suites.
"""



