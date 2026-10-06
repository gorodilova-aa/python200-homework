# ------------------------------------------------------------------------------------
# --- Classes --- 
print("\n ---  Classes ---\n")
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
# # Dataclass Q1-Q2

# rewrite class as a dataclass
#class Station:
#    def __init__(self, station_id, name, latitude, longitude, elevation):
#        self.station_id = station_id
#        self.name = name
#        self.latitude = latitude
#        self.longitude = longitude
#        self.elevation = elevation

from dataclasses import dataclass, FrozenInstanceError, field

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
print("\n Example of using StationBatch")
batch = StationBatch(region="Triangle")
print(f"Created batch: {batch}")

s1 = Station("RTP1", "Morrisville", 35.82, -78.82, 110.0)
s2 = Station("RTP2", "Cary", 35.79, -78.78, 150.0)

batch.add(s1)
batch.add(s2)

print(f"Updated batch: {batch}")

print(f"Highest station in {batch.region}: {batch.highest().name}")