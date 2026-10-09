# -----------------------------------------------------------------------
# Pydantic schemas validating raw external Open-Meteo API payloads.
# -----------------------------------------------------------------------

from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Columnar weather observation data over time."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def verify_equal_lengths(self):
        """Ensure all observation lists have the same length. Raise an error if the three lists are not all the same length."""
        len_time = len(self.time)
        len_temp = len(self.temperature_2m)
        len_precip = len(self.precipitation)

        if not (len_time == len_temp == len_precip):
            raise ValueError(
                f"Columnar arrays must be of equal length. "
                f"Observed lengths: time={len_time}, temperature_2m={len_temp}, precipitation={len_precip}."
            )
        return self


class WeatherResponse(BaseModel):
    """Top level of Open-Meteo forecast API response"""

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone: str
    elevation: float
    hourly: HourlyBlock

