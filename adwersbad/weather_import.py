import datetime

import openmeteo_requests
import requests_cache
from retry_requests import retry
from suncalc import get_times

cache_session = requests_cache.CachedSession(".cache", expire_after=-1)

retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://archive-api.open-meteo.com/v1/archive"
weather_wmo_codes = {
    0: "Clear Sky",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    56: "Light Freezing Drizzle",
    57: "Dense Freezing Drizzle",
    61: "Light Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    66: "Light Freezing Rain",
    67: "Heavy Freezing Rain",
    71: "Light Snow",
    73: "Moderate Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Slight Rain Showers",
    81: "Moderate Rain Showers",
    82: "Heavy Rain Showers",
    85: "Slight Snow Showers",
    86: "Heavy Snow Showers",
    95: "Slight or Moderate Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def request_data(lat, lon, date_start, date_end):
    """Request weather data from open-meteo.com API

    Args:
    lat (float): latitude
    lon (float): longitude
    date_start (datetime.date): start date
    date_end (datetime.date): end date
    
    Returns:
    json: weather data

    String format:
    https://archive-api.open-meteo.com/v1/archive? \
        latitude=52.52&longitude=13.41&start_date=2023-05-11&end_date=2023-05-25& \
        hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,surface_pressure,precipitation,rain,snowfall,weathercode,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,windspeed_10m,winddirection_10m,windgusts_10m \ 
        &15min=temperature_2m \
        &models=best_match&timeformat=unixtime
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_start,
        "end_date": date_end,
        # "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "rain", "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm", "is_day"]
        "hourly": ["weather_code", "is_day"],
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_wmo = hourly.Variables(0).ValuesAsNumpy()
    hourly_isday = hourly.Variables(1).ValuesAsNumpy()
    # hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    # hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    # hourly_dew_point_2m = hourly.Variables(2).ValuesAsNumpy()
    # hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
    # hourly_rain = hourly.Variables(4).ValuesAsNumpy()
    # hourly_snowfall = hourly.Variables(5).ValuesAsNumpy()
    # hourly_snow_depth = hourly.Variables(6).ValuesAsNumpy()
    # hourly_weather_code = hourly.Variables(7).ValuesAsNumpy()
    # hourly_pressure_msl = hourly.Variables(8).ValuesAsNumpy()
    # hourly_surface_pressure = hourly.Variables(9).ValuesAsNumpy()
    # hourly_cloud_cover = hourly.Variables(10).ValuesAsNumpy()
    # hourly_cloud_cover_low = hourly.Variables(11).ValuesAsNumpy()
    # hourly_cloud_cover_mid = hourly.Variables(12).ValuesAsNumpy()
    # hourly_cloud_cover_high = hourly.Variables(13).ValuesAsNumpy()
    # hourly_wind_speed_10m = hourly.Variables(14).ValuesAsNumpy()
    # hourly_wind_direction_10m = hourly.Variables(15).ValuesAsNumpy()
    # hourly_soil_temperature_0_to_7cm = hourly.Variables(16).ValuesAsNumpy()
    # hourly_soil_moisture_0_to_7cm = hourly.Variables(17).ValuesAsNumpy()
    # hourly_is_day = hourly.Variables(18).ValuesAsNumpy()
    #
    # hourly_data = {"date": pd.date_range(
    #     start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
    #     end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
    #     freq = pd.Timedelta(seconds = hourly.Interval()),
    #     inclusive = "left"
    # )}
    # hourly_data["temperature_2m"] = hourly_temperature_2m
    # hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    # hourly_data["dew_point_2m"] = hourly_dew_point_2m
    # hourly_data["precipitation"] = hourly_precipitation
    # hourly_data["rain"] = hourly_rain
    # hourly_data["snowfall"] = hourly_snowfall
    # hourly_data["snow_depth"] = hourly_snow_depth
    # hourly_data["weather_code"] = hourly_weather_code
    # hourly_data["pressure_msl"] = hourly_pressure_msl
    # hourly_data["surface_pressure"] = hourly_surface_pressure
    # hourly_data["cloud_cover"] = hourly_cloud_cover
    # hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    # hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
    # hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    # hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    # hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    # hourly_data["soil_temperature_0_to_7cm"] = hourly_soil_temperature_0_to_7cm
    # hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
    # hourly_data["is_day"] = hourly_is_day
    #
    # hourly_dataframe = pd.DataFrame(data = hourly_data)
    return hourly_wmo, hourly_isday


def get_weather_from_timestamp(ts, lat, lon, delta=0):
    if not isinstance(ts, datetime.datetime):
        dt = datetime.datetime.fromtimestamp(ts / 1000000)
    else:
        dt = ts
    date = dt.date()
    hour = dt.hour
    weather, isday = request_data(lat, lon, date, date + datetime.timedelta(days=delta))
    weather = weather[hour]
    isday = isday[hour]
    weather_string = weather_wmo_codes[weather]
    return weather, weather_string, isday


def check_if_twilight(ts, lat, lon):
    times = get_times(ts, lon, lat)
    dawn = times["nautical_dawn"]
    sunrise = times["sunrise"]
    dusk = times["nautical_dusk"]
    sunset = times["sunset"]
    is_twilight = False
    tod = ""
    if ts > dawn and ts < sunrise:
        tod = "dawn"
        is_twilight = True
    if ts > sunset and ts < dusk:
        tod = "dusk"
        is_twilight = True
    return is_twilight, tod
