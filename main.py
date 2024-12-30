from tts_advanced import AdvancedStuffSayer
from config import Config
from stt import StuffHearer
from langchain_community.llms import Ollama
import json
import weather

config = Config()
stuff_sayer = AdvancedStuffSayer()
stuff_hearer = StuffHearer()

text_only = False
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

    json_str = find_first_json_object(ollama_response)
    if json_str is None:
        print("No JSON here, buddy!")
        return ollama_response
    else:
        print("JSON Object to parse, good luck: {}".format(json_str))
        try:
            json_obj = json.loads(json_str)
            return perform_action(json_obj)
        except json.JSONDecodeError:
            return "There's some json in this response, but it's gross {}".format(ollama_response)


def find_first_json_object(input_str):
    # Variable to keep track of where the JSON object starts
    start_index = input_str.find('{')

    if start_index == -1:
        return None  # No opening brace found, so no JSON object

    # Start scanning for the end of the first complete JSON object
    brace_count = 0
    for i in range(start_index, len(input_str)):
        char = input_str[i]

        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        # If brace count is 0, we found the end of a complete JSON object
        if brace_count == 0:
            end_index = i + 1
            json_str = input_str[start_index:end_index]
            return json_str
    # If no complete JSON object is found
    return None

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
        if text_only:
            print("Text only set, response is {}".format(response))
        else:
            stuff_sayer.say_stuff(response)

if __name__ == '__main__':
    #hear_mode()
    #stuff_sayer.say_stuff("Hey")
    prompt = "What the weather is like in Chicago today?"
    print(ask_stuff(prompt))
