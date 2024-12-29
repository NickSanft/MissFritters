from tts_advanced import AdvancedStuffSayer
from config import Config
from stt import StuffHearer
from langchain_community.llms import Ollama
import json
import re
import weather

config = Config()
stuff_sayer = AdvancedStuffSayer()
#stuff_sayer = SimpleStuffSayer()
stuff_hearer = StuffHearer()

fritters_key = "fritters_action"
weather_action = "get_weather"
weather_prompt_addition = """
Please respond as you normally would unless the user is asking specifically about the weather in a city. If they are, please do the following: 
    1) Please return a JSON key called {} with a value of {} and a JSON key for the city.
    2) If your model knows the definite latitude and longitude of the city, please also provide JSON keys called latitude and longitude.
    3) If the user specified fahrenheit or celsius, please also provide a JSON key called temperature_unit with a value of fahrenheit or celsius respectively.
""".format(fritters_key, weather_action)

config_llama_model = "llama_model"
ollama_instance = Ollama(model=config.get_config(config_llama_model))

def ask_stuff(prompt_to_ask: str):
    full_request = "The user is asking: \"{}\". {}".format(prompt_to_ask, weather_prompt_addition)
    print("Full request to ask: " + full_request)
    ollama_response = ollama_instance.invoke(full_request)
    print("Response from model: {}".format(ollama_response))

    match = re.search(r'{.*?}', ollama_response, re.DOTALL)
    if not match:
        print("No JSON here, buddy!")
        return ollama_response
    else:
        json_str = match.group(0)
        print("JSON Object: {}".format(json_str))
        json_obj = json.loads(json_str)
        return perform_action(json_obj)

def perform_action(json_obj: json):
    if fritters_key in json_obj:
        action_to_perform = json_obj[fritters_key]
        if action_to_perform == weather_action:
            return weather.get_weather(json_obj)

def hear_mode():
    stuff_sayer.say_stuff("Hey, I am Miss Fritters! What would you like to ask?")
    while True:
        prompt = None
        while prompt is None:
            prompt = stuff_hearer.hear_stuff()
        response = ask_stuff(prompt)
        stuff_sayer.say_stuff(response)

if __name__ == '__main__':
    hear_mode()
    #stuff_sayer.say_stuff("Hey")
    #prompt = "What the weather is like in Chicago today?"
    #print(ask_stuff(prompt))
