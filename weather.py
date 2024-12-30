import json
import math

import openmeteo_requests
import requests

import requests_cache

from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

temp_unit_key = "temperature_unit"

def get_lat_long(city: str):
	print("No latitude or longitude provided")
	geo_params = {
		"name": city,
		"language": "en",
		"format": "json"
	}
	print("geo_params {}".format(geo_params))
	response = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=geo_params)
	if response.status_code == 200:
		try:
			data = response.json()
			# Extract the first object from the 'results' list
			first_object = data['results'][0]
			print("Parsed JSON response:", first_object)
			latitude = first_object['latitude']
			longitude = first_object['longitude']
			return latitude, longitude
		except ValueError:
			print("Failed to parse response as JSON.")


def get_weather_params(latitude, longitude, temperature_unit):
	weather_params = {
		"latitude": latitude,
		"longitude": longitude,
		"current": "temperature_2m",
		"hourly": "temperature_2m",
		"daily": "weather_code",
		"temperature_unit": temperature_unit,
		"timezone": "America/Chicago",
		"forecast_days": 1
	}
	return weather_params


def get_weather(json_obj: json):
	city = json_obj["city"]
	if "latitude" in json_obj and "longitude" in json_obj:
		latitude = json_obj["latitude"]
		longitude = json_obj["longitude"]
	else:
		latitude, longitude = get_lat_long(city)

	if temp_unit_key in json_obj:
		temperature_unit = json_obj[temp_unit_key]
	else:
		temperature_unit = "fahrenheit"

	weather_params = get_weather_params(latitude, longitude, temperature_unit)


	responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=weather_params)
	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	current = response.Current()
	current_temperature = math.ceil(current.Variables(0).Value())

	result_text = "The weather in {} right now is {} degrees {}".format(city, current_temperature, temperature_unit)
	return result_text