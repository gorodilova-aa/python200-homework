# -------------------------------------------------------------------------------------------------
# Script to load weather_raw.json file and to validate it with WeatherResponse.model_validate(...).
# -------------------------------------------------------------------------------------------------

import json
from weatherkit import DailyAggregator, to_readings, HourlyReading, WeatherResponse


with open("weather_raw.json", "r") as f:
    data = json.load(f)

resp = WeatherResponse.model_validate(data)
print(f"Latitude: {resp.latitude}, Timezone: {resp.timezone}")
print(f"Observations count: {len(resp.hourly.time)}")


# ----------- self-check after task 4 ---------------

with open("weather_raw.json") as f:
    raw = json.load(f)
data = WeatherResponse.model_validate(raw)


readings = to_readings(data)


aggregator = DailyAggregator(min_hours=24)
summaries = aggregator.summarize(readings)


print(f"Number of full days: {len(summaries)}")
for s in summaries[:2]:
    print(f"Date: {s.date}, Max Temp: {s.temp_max}, Min Temp: {s.temp_min}, Precipitation: {s.precipitation_sum}, Hours Observed: {s.hours_observed}")