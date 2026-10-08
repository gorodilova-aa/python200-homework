# -----------------------------------------------------
# Tests for weatherkit.summarize.
# -----------------------------------------------------

import pytest
from weatherkit.records import HourlyReading
from weatherkit.summarize import DailyAggregator, DailySummary


# Note in a comment which test caught the change.
# - If we break DailySummary.temp_range (e.g. change temp_max - temp_min to temp_max + temp_min),
# - test_temp_range caught the change:
#FAILED tests/test_summarize.py::test_temp_range[25.0-15.0-10.0] - assert 40.0 == 10.0 ± 1.0e-05
#FAILED tests/test_summarize.py::test_temp_range[5.0--5.0-10.0] - assert 0.0 == 10.0 ± 1.0e-05
#FAILED tests/test_summarize.py::test_temp_range[19.3-3.7-15.6] - assert 23.0 == 15.6 ± 1.6e-05


@pytest.fixture
def two_day_readings() -> list[HourlyReading]:
    readings = []
    # Day 1: 24 observations
    for h in range(24):
        readings.append(
            HourlyReading(
                timestamp=f"2026-10-08T{h:02d}:00",
                temperature_c=10.0 + h * 0.5,
                precipitation_mm=0.1,
            )
        )
    # Day 2: 24 observations
    for h in range(24):
        readings.append(
            HourlyReading(
                timestamp=f"2026-10-09T{h:02d}:00",
                temperature_c=5.0 + h * 0.2,
                precipitation_mm=0.2,
            )
        )
    return readings


def test_grouping_two_full_days(two_day_readings):
    aggregator = DailyAggregator(min_hours=24)
    summaries = aggregator.summarize(two_day_readings)

    assert len(summaries) == 2
    


def test_temp_min_and_max(two_day_readings):
    aggregator = DailyAggregator(min_hours=24)
    summaries = aggregator.summarize(two_day_readings)

    day1 = summaries[0]
    assert day1.temp_min == 10.0
    assert day1.temp_max == 10.0 + 23 * 0.5


def test_precipitation_sum(two_day_readings):
    aggregator = DailyAggregator(min_hours=24)
    summaries = aggregator.summarize(two_day_readings)

    # 24 * 0.1 = 2.4
    assert summaries[0].precipitation_sum == pytest.approx(2.4)


def test_incomplete_day_dropped():
    # Only 5 observations for 2026-10-10
    short_day = [
        HourlyReading(f"2026-10-08T{h:02d}:00", 15.0, 0.0) for h in range(5)
    ]
    aggregator = DailyAggregator(min_hours=24)

    summaries = aggregator.summarize(short_day)
    dropped = aggregator.incomplete_days(short_day)

    assert len(summaries) == 0
    assert dropped == ["2026-10-08"]


def test_lowering_min_hours_keeps_day():
    short_day = [
        HourlyReading(f"2026-10-08T{h:02d}:00", 15.0, 0.0) for h in range(5)
    ]
    aggregator = DailyAggregator(min_hours=5)

    summaries = aggregator.summarize(short_day)
    dropped = aggregator.incomplete_days(short_day)

    assert len(summaries) == 1
    assert summaries[0].date == "2026-10-08"
    assert dropped == []


@pytest.mark.parametrize(
    "t_max, t_min, expected_range",
    [
        (25.0, 15.0, 10.0),
        (5.0, -5.0, 10.0),
        (0.0, 0.0, 0.0),
        (19.3, 3.7, 15.6),
    ],
)
def test_temp_range(t_max, t_min, expected_range):
    summary = DailySummary(
        date="2026-10-08",
        temp_max=t_max,
        temp_min=t_min,
        precipitation_sum=0.0,
        hours_observed=24,
    )
    assert summary.temp_range() == pytest.approx(expected_range)