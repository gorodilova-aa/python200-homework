# -----------------------------------------------------
# Tests for weatherkit.records.
# -----------------------------------------------------

from weatherkit.schemas import WeatherResponse, HourlyBlock
from weatherkit.records import HourlyReading, to_readings


def _make_an_example_response() -> WeatherResponse:
    return WeatherResponse(
        latitude=55.0302,  
        longitude=82.9204,
        timezone="Novosibirsk",
        elevation=153.0,
        hourly=HourlyBlock(
            time=["2026-10-08T12:00", "2026-10-08T13:00", "2026-10-08T14:00"],
            temperature_2m=[10.0, 11.7, 13.5],
            precipitation=[0.0, 0.5, 0.0],
        ),
    )


def test_to_readings_preserves_order_and_bounds():
    response = _make_an_example_response()
    readings = to_readings(response)

    assert len(readings) == 3
    assert readings[0].timestamp == "2026-10-08T12:00"
    assert readings[-1].timestamp == "2026-10-08T14:00"


def test_to_readings_pairs_index_correctly():
    response = _make_an_example_response()
    readings = to_readings(response)

    for i, r in enumerate(readings):
        assert r.timestamp == response.hourly.time[i]
        assert r.temperature_c == response.hourly.temperature_2m[i]
        assert r.precipitation_mm == response.hourly.precipitation[i]


def test_hourly_reading_equality():
    r1 = HourlyReading("2026-10-08T14:00", 25.0, 0.1)
    r2 = HourlyReading("2026-10-08T14:00", 25.0, 0.1)
    assert r1 == r2