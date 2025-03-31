from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, NamedTuple


class Scenario(NamedTuple):
    # TODO: Merging(adding) scenarios
    name: str
    is_tod: bool
    is_weather: bool
    conditions: dict[str, List[str]]

    def __mul__(self: Scenario, other: Scenario) -> Scenario:
        if isinstance(other, self.__class__):
            if self.is_tod and other.is_tod:
                if not (self.conditions["tod"] == other.conditions["tod"]):
                    raise NotImplementedError
            if self.is_weather and other.is_weather:
                cl = Counter(self.conditions["weather"])
                cr = Counter(other.conditions["weather"])
                if not cl == cr:
                    raise NotImplementedError
            new_name = self.name + other.name
            new_is_tod = self.is_tod or other.is_tod
            new_is_weather = self.is_weather or other.is_weather
            new_conditions = {**self.conditions, **other.conditions}
            return Scenario(new_name, new_is_tod, new_is_weather, new_conditions)

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, self.__class__):
            new_name = self.name + other.name
            new_is_tod = self.is_tod or other.is_tod
            new_is_weather = self.is_weather or other.is_weather
            new_conditions = defaultdict(list)
            {
                new_conditions[key].extend(value)
                for d in [self.conditions, other.conditions]
                for key, value in d.items()
            }
            return Scenario(new_name, new_is_tod, new_is_weather, new_conditions)

    __radd__ = __add__


daydict = {"tod": ["Day", "day"]}
nightdict = {"tod": ["Night", "night"]}
twilightdict = {"tod": ["Dawn/Dusk", "Dawn", "Dusk", "dusk", "dawn"]}
raindict = {
    "weather": [
        "rain",
        "Light Drizzle",
        "Moderate Drizzle",
        "Dense Drizzle",
        "Light Rain",
        "Moderate Rain",
        "Heavy Rain",
    ]
}
sundict = {"weather": ["sunny", "Clear Sky", "Mainly Clear"]}
clouddict = {"weather": ["Partly Cloudy", "Overcast"]}
snowdict = {"weather": ["Light Snow", "Moderate Snow"]}

ALL = Scenario("all", False, False, {})
NIGHT = Scenario("night", True, False, nightdict)
DAY = Scenario("day", True, False, daydict)
TWILIGHT = Scenario("twilight", True, False, twilightdict)

RAIN = Scenario("rain", False, True, raindict)
SUN = Scenario("sun", False, True, sundict)
CLOUD = Scenario("cloud", False, True, clouddict)
SNOW = Scenario("snow", False, True, snowdict)

NIGHTRAIN = NIGHT * RAIN
DAYRAIN = DAY * RAIN
DAYSUN = DAY * SUN
DAYRAINSUN = DAY * (RAIN + SUN)
NIGHTSUN = NIGHT * SUN
TWILIGHTSUN = TWILIGHT * SUN
TWILIGHTRAIN = TWILIGHT * RAIN
NIGHTCLOUD = NIGHT * CLOUD
DAYCLOUD = DAY * CLOUD
TWILIGHTCLOUD = TWILIGHT * CLOUD
NIGHTSNOW = NIGHT * SNOW
DAYSNOW = DAY * SNOW
TWILIGHTSNOW = TWILIGHT * SNOW

scenarios = [
    DAY,
    NIGHT,
    TWILIGHT,
    SUN,
    RAIN,
    CLOUD,
    SNOW,
    DAYSUN,
    DAYRAIN,
    DAYRAINSUN,
    DAYCLOUD,
    DAYSNOW,
    NIGHTSUN,
    NIGHTRAIN,
    NIGHTCLOUD,
    NIGHTSNOW,
    TWILIGHTSUN,
    TWILIGHTRAIN,
    TWILIGHTCLOUD,
    TWILIGHTSNOW,
    ALL,
]
adwersbad_scenarios = {}
for s in scenarios:
    adwersbad_scenarios[s.name] = s


if __name__ == "__main__":
    print(DAY)
    print(RAIN)
    print(DAY * RAIN)
    print(DAY + SUN)
    print(DAY + NIGHT)
    print(DAY * (RAIN + SUN))
    print(DAY + RAIN + SUN)
