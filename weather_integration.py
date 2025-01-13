import math
import requests
import requests_cache
import openmeteo_requests
from retry_requests import retry

# Initialize cache session and retry session
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)

# Openmeteo client
openmeteo = openmeteo_requests.Client(session=retry_session)

# Constants
TEMP_UNIT_KEY = "temperature_unit"
DEFAULT_TEMP_UNIT = "fahrenheit"


def get_lat_long(city: str):
    """
    Fetch the latitude and longitude for a given city using Open Meteo's geocoding API.
    """
    geo_params = {
        "name": city,
        "language": "en",
        "format": "json"
    }

    response = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=geo_params)

    if response.status_code == 200:
        try:
            data = response.json()
            first_object = data['results'][0]
            latitude = first_object['latitude']
            longitude = first_object['longitude']
            return latitude, longitude
        except ValueError:
            print("Error: Failed to parse response as JSON.")
    else:
        print(f"Error: Unable to fetch data for city {city}. Status code: {response.status_code}")
    return None, None


def get_weather_params(latitude, longitude, temperature_unit):
    """
    Return the weather parameters for a given latitude, longitude, and temperature unit.
    """
    return {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m",
        "hourly": "temperature_2m",
        "daily": "weather_code",
        "temperature_unit": temperature_unit,
        "timezone": "America/Chicago",
        "forecast_days": 1
    }


def fetch_weather_data(weather_params):
    """
    Fetch weather data from Open Meteo API using provided weather parameters.
    """
    return openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=weather_params)


def get_weather(city: str):
    """
    Retrieve and format the weather for a given city based on a JSON object containing city name,
    latitude, longitude, and temperature unit.
    """

    # Get latitude and longitude, either from the json_obj or by fetching from the API
    latitude, longitude = get_lat_long(city)

    if not latitude or not longitude:
        return f"Error: Unable to retrieve location data for {city}."

    # Get the weather parameters and fetch the weather data
    weather_params = get_weather_params(latitude, longitude, DEFAULT_TEMP_UNIT)
    responses = fetch_weather_data(weather_params)

    if responses:
        response = responses[0]
        current = response.Current()
        current_temperature = math.ceil(current.Variables(0).Value())

        return f"The weather in {city} right now is {current_temperature} degrees Fahrenheit."
    return f"Error: Unable to fetch weather data for {city}."