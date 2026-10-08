from dataclasses import dataclass, FrozenInstanceError, field
from pydantic import BaseModel, Field, ValidationError, model_validator
import pytest


# ------------------------------------------------------------------------------------
# --- Classes --- 
print("\n ---  Classes ---\n")
# ------------------------------------------------------------------------------------

# # Classes Q1-Q2 ---

# Write a class Thermometer that stores a list of temperature readings in Celsius.
class Thermometer:
    def __init__(self, location, readings = None):
        if readings is None:
            readings = []
        self.location = str(location)
        self.readings = list(readings)

    def __repr__(self):
        return f"Thermometer(location={self.location!r}, n_readings={len(self.readings)}, average={self.average()})"

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

# --- Testing the Thermometer class for Question 1 ---
# Create a Thermometer for a location of your choice, add at least four readings, and print the average and the hottest.
thermometer_MOR = Thermometer("Morrisville")

# check empty readings
print("\nLocation:", thermometer_MOR.location) 
print("Readings:", thermometer_MOR.readings)
print("Average:", thermometer_MOR.average())
print("Hottest:", thermometer_MOR.hottest())

# add at least four readings
thermometer_MOR.add(20)
thermometer_MOR.add(22)
thermometer_MOR.add(19)
thermometer_MOR.add(21)

# check readings after adding temperatures
print("\nLocation:", thermometer_MOR.location) 
print("Readings:", thermometer_MOR.readings)
print("Average:", thermometer_MOR.average())
print("Hottest:", thermometer_MOR.hottest())

# why does average() need to handle the empty case?
# - if there are no readings, calculating the average would result in a division by zero error. 
# - Handling the empty case prevents this error and allows the program to return a meaningful value (None) instead.

# --- Testing the Thermometer class for Question 2 ---
# Cheking the __repr__ method of the Thermometer class
print("\nTesting __repr__ method:")
print(thermometer_MOR)

# adding another one instance
thermometer_CARY = Thermometer("Cary")
thermometer_CARY.add(12)
thermometer_CARY.add(16)
thermometer_CARY.add(20)

# Checking the __repr__ method for the list of Thermometer class
print([thermometer_MOR, thermometer_CARY])



# what Python displays when a class has no __repr__, and why that is unhelpful when debugging?
# - Without __repr__, Python displays the default '<ClassName object at 0x...>', which only reveals the type and memory address. 
# - This is unhelpful during debugging because it hides the object's internal state, making it impossible to distinguish instances in logs or tracebacks.


# # Classes Q3 ---

class TemperatureAlert:
    def __init__(self, threshold = 30.0):
        self.threshold = float(threshold)

    def breaches(self, thermometer):
        return list(temperature for temperature in thermometer.readings if temperature > self.threshold)

# --- Testing the TemperatureAlert class for Question 3 ---
alert1 = TemperatureAlert(20)
alert2 = TemperatureAlert()

# check if alerts are triggered for the thermometer
print("\nAlert 1 breaches (threshold {}):".format(alert1.threshold), alert1.breaches(thermometer_MOR))
print("Alert 2 breaches (threshold {}):".format(alert2.threshold), alert2.breaches(thermometer_MOR))

# why is the threshold stored on TemperatureAlert rather than passed as an argument to breaches()?
# What advantage does that give you if you have twenty thermometers to check?
# - Storing the threshold on TemperatureAlert configures the rule once on the object.
# - With 20 thermometers, you can reuse that single alert across all of them without re-passing the threshold 20 times.


# ------------------------------------------------------------------------------------
# --- Dataclasses, Type Hints, and Docstrings ---
print("\n ---  Dataclasses, Type Hints, and Docstrings ---\n")
# ------------------------------------------------------------------------------------

# # Dataclass Q1-Q2

# rewrite class as a dataclass
#class Station:
#    def __init__(self, station_id, name, latitude, longitude, elevation):
#        self.station_id = station_id
#        self.name = name
#        self.latitude = latitude
#        self.longitude = longitude
#        self.elevation = elevation



@dataclass(frozen=True)
class Station:
    """
    A class representing a weather station.

    Attributes:
        station_id (str): The unique identifier for the station.
        name (str): The name of the station.
        latitude (float): The latitude of the station.
        longitude (float): The longitude of the station.
        elevation (float): The elevation of the station in meters.
    """
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float

# Create two Station objects with identical field values and print station_a == station_b
station_a = Station("NSK", "Novosibirsk", 55.0302, 82.9204, 153.0)
station_b = Station("NSK", "Novosibirsk", 55.0302, 82.9204, 153.0)
print(station_a)
print(station_b)
print(f"(station_a == station_b): {station_a == station_b}")

# why the result is what it is, and what it would have been with the original hand-written class?
# - The result is True because @dataclass automatically generates an __eq__ method.
# - With the original hand-written class, the result would have been False because standard classes inherit the default __eq__ from object, 
# which checks identity (whether they are the same memory location, i.e., station_a is station_b).


# try to modify one of the fields of station_a
print("\nThe message for attempting to modify station_a:")
try:
    station_a.latitude = 55.0
except FrozenInstanceError as e:
    print("Cannot modify:", e)


# Build a set containing three Station objects where two are identical, and print the length.
station_c = Station("MSK", "Moscow", 55.7558, 37.6173, 156.0)
print("\nAdding these three stations to station_set:")
print(station_a)
print(station_b)
print(station_c)
station_set = {station_a, station_b, station_c}
print(f"\nLength of station_set: {len(station_set)}")

# what does frozen=True give you besides immutability, and why is that useful here?
# - Besides immutability, frozen=True also makes the dataclass hashable if all its fields are hashable.
# - This is useful because it allows instances of the Station class to be added to sets and used as dictionary keys, as demonstrated above.

# # Dataclass Q3
@dataclass
class StationBatch:
    """
    A class representing a batch of weather stations.

    Attributes:
        region (str): The region to which the batch of stations belongs.
        stations (list[Station]): A list of Station objects.
    """
    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """
        Add a Station object to the batch.

        Args:
            station (Station): The Station object to add.
        """
        self.stations.append(station)
        
    def highest(self) -> Station | None:
        """
        Get the Station object with the highest elevation in the batch.

        Returns:
            Station | None: The Station object with the highest elevation, or None if the batch is empty.
        """
        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)



# Attemp to create a dataclass with stations: list[Station] = [] rised the following error:
# "ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory"
# we fixed this using the following: list[Station] = field(default_factory=list) that ensures a fresh, independent list is created for each instance.


# Trying to use the class
print("\n Example of using ")
batch = StationBatch(region="Triangle")
print(f"Created batch: {batch}")

s1 = Station("RTP1", "Morrisville", 35.82, -78.82, 110.0)
s2 = Station("RTP2", "Cary", 35.79, -78.78, 150.0)

batch.add(s1)
batch.add(s2)

print(f"Updated batch: {batch}")

print(f"Highest station in {batch.region}: {batch.highest().name}")


# ------------------------------------------------------------------------------------
# --- Pydantic ---
print("\n ---  Pydantic ---\n")
# ------------------------------------------------------------------------------------

# # Pydantic Q1 & Q4

class Reading(BaseModel):
    """
    A class representing a weather reading.

    Attributes:
        station_id (str): at list 3 characters
        timestamp (str): required
        temperature_c (float): between -90 and 60
        humidity (float): between 0 and 100
    """
    station_id: str = Field(...,min_length=3)
    timestamp: str = Field(..., description="Required timestamp")
    temperature_c: float = Field(..., ge=-90, le=60)
    humidity: float = Field(..., ge=0, le=100)

    # part for Q4
    @model_validator(mode="after")
    def check_sensor(self):
        """A failed sensor error: humidity is exactly 0.0 AND temperature_c is below -40"""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                f"Failed sensor: humidity is {self.humidity} and temperature_c is {self.temperature_c}"
            )
        return self

valid_reading = Reading(
    station_id="RTP1",
    timestamp="2026-10-06T15:28",
    temperature_c=21.0,
    humidity=42.0
)
print(f"Valid reading: {valid_reading}\n")

# # Pydantic Q2

# A missing required field
try:
    invalid_reading = Reading(
        station_id="RTP1",
        timestamp=None,
        temperature_c=21.0,
        humidity="42.0"
    )
except ValidationError as e:
    print(f"Validation error 1: {e}\n")

# A temperature_c of 150.0
try:
    invalid_reading = Reading(
        station_id="RTP1",
        timestamp="2026-10-06T15:28",
        temperature_c=150.0,
        humidity="42.0"
    )
except ValidationError as e:
    print(f"Validation error 2: {e}\n")

# A humidity of "very humid"
try:
    invalid_reading = Reading(
        station_id="RTP1",
        timestamp="2026-10-06T15:28",
        temperature_c=21.0,
        humidity="very humid"
    )
except ValidationError as e:
    print(f"Validation error 3: {e}\n")


# Then construct a Reading where temperature_c is passed as the string "21.5" and humidity is passed as the integer 40.
try:
    reading = Reading(
        station_id="RTP1",
        timestamp="2026-10-06T15:28",
        temperature_c="21.5",
        humidity=40
    )
    print(f"Reading with input str temperature and int humidity: {reading}")
    print(f"type of temperature_c: {type(reading.temperature_c)}")
    print(f"type of humidity: {type(reading.humidity)}")    
except ValidationError as e:
    print(f"Validation error: {e}")

# When possibble Pydantic will try to coerce types to the correct type.
# In the previous example, Pydantic successfully converted the string "21.5" to a float and the integer 40 to a float for humidity,
# but obviously it cannot convert "very humid" to a float.


# # Pydantic Q3

try:
    reading = Reading(
        station_id="RT",
        timestamp=None,
        temperature_c="cold",
        humidity=40.4
    )
except ValidationError as e:
    print(f"\n{e.error_count()} problems found:")
    for err in e.errors():
        print(f"  field={err['loc']}  type={err['type']}  msg={err['msg']}")

# # Pydantic Q4

# we added check_sensor with mode="after" above 

# valid sensor reading
valid_sensor_reading = Reading(
    station_id="RTP1",
    timestamp="2026-10-06T15:28",
    temperature_c=21.0,
    humidity=42.0
)
print(f"\nValid sensor reading: {valid_sensor_reading}\n")

# invalid sensor reading
try:
    invalid_sensor_reading = Reading(
        station_id="RTP1",
        timestamp="2026-10-06T15:28",
        temperature_c=-54.0,
        humidity=0.0
    )
except ValidationError as e:
    print(f"Validation error for invalid sensor reading: {e}\n")

# why this rule cannot be expressed with Field constraints alone?
# - Field constraints alone cannot check multiple fields in relation to each other.


# ------------------------------------------------------------------------------------
# --- Pytest ---
print("\n ---  Pytest ---\n")
# ------------------------------------------------------------------------------------

# # Pytest Q1 

def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius (float) to Fahrenheit (float)."""
    return (celsius * 9/5) + 32

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# we add pytest.approx for floating point comparison

# # Pytest Q2 

def mean(values: list[float]) -> float:
    """Calculate the mean of a non-empty list of float values."""

    if not values:
        raise ValueError("The list of values cannot be empty.")
    return sum(values) / len(values)

def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# without match="empty" the test would pass for function that does not check the empty list condition specifically.


# # Pytest Q3

@pytest.mark.parametrize(
    "values, expected_mean",
    [
        ([1.0], 1.0),
        ([-14.0, 5.0, 9.0], 0.0),
        ([10.0, 20.0, 30.0], 20.0),
        ([0.0, -100.0, 40.0], -20.0),
    ]
)
def test_mean_values(values, expected_mean):
    assert mean(values) == expected_mean

#warmup_01.py::test_mean_values[values0-1.0] PASSED
#warmup_01.py::test_mean_values[values1-0.0] PASSED
#warmup_01.py::test_mean_values[values2-20.0] PASSED
#warmup_01.py::test_mean_values[values3--20.0] PASSED

# why is one parametrized test with four cases better than four nearly identical test functions?
# - It reduces code duplication.
# - It makes it easier to add more test cases.
# - It provides a clear overview of all test scenarios.
# - It ensures consistency in test logic.


# # Pytest Q4

# after changing the conversion formula from 9/5 to 9/4, we get this info:
#======================================================================================= FAILURES =======================================================================================
#______________________________________________________________________________ test_celsius_to_fahrenheit ______________________________________________________________________________
#
#    def test_celsius_to_fahrenheit():
#       assert celsius_to_fahrenheit(0) == 32
#       assert celsius_to_fahrenheit(100) == 212
#       assert 257.0 == 212
#        +  where 257.0 = celsius_to_fahrenheit(100)
#warmup_01.py:370: AssertionError
#=============================================================================== short test summary info ================================================================================
#FAILED warmup_01.py::test_celsius_to_fahrenheit - assert 257.0 == 212

# what specific values did pytest show you in the failure report, and why is that more useful than a bare "assertion failed"?
# - Pytest showed that celsius_to_fahrenheit(100) returned 257.0 instead of the expected 212, which helps identify the incorrect conversion formula.

