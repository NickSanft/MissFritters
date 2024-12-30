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

# Constants
FRITTERS_KEY = "fritters_action"
WEATHER_ACTION = "get_weather"
CONFIG_LLAMA_MODEL  = "llama_model"

WEATHER_PROMPT_ADDITION  = """
Please respond as you normally would unless the user is asking specifically about the weather in a city. If they are, please do the following: 
    1) Please return a JSON key called {} with a value of {} and a JSON key for the city.
    2) If your model knows the definite latitude and longitude of the city, please also provide JSON keys called latitude and longitude.
    3) If the user specified fahrenheit or celsius, please also provide a JSON key called temperature_unit with a value of fahrenheit or celsius respectively.
""".format(FRITTERS_KEY, WEATHER_ACTION)

# Initialize Ollama model
ollama_instance = Ollama(model=config.get_config(CONFIG_LLAMA_MODEL))


def ask_stuff(prompt: str) -> str:
    """Ask the Ollama model and handle response."""
    full_request = f'The user is asking: "{prompt}". {WEATHER_PROMPT_ADDITION}'
    print(f"Full request to ask: {full_request}")

    ollama_response = ollama_instance.invoke(full_request)
    print(f"Response from model: {ollama_response}")

    json_str = find_first_json_object(ollama_response)
    if not json_str:
        print("No JSON object found!")
        return ollama_response

    print(f"JSON Object to parse: {json_str}")
    try:
        json_obj = json.loads(json_str)
        return perform_action(json_obj)
    except json.JSONDecodeError:
        return f"There's some malformed JSON in this response: {ollama_response}"


def find_first_json_object(input_str: str) -> str:
    """Extract and return the first valid JSON object from the string."""
    start_index = input_str.find('{')
    if start_index == -1:
        return None  # No opening brace found

    brace_count = 0
    for i in range(start_index, len(input_str)):
        char = input_str[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if brace_count == 0:
            return input_str[start_index:i + 1]  # Return the complete JSON object

    return None  # No complete JSON object found

def perform_action(json_obj: dict) -> str:
    """Perform an action based on the parsed JSON object."""
    if FRITTERS_KEY in json_obj and json_obj[FRITTERS_KEY] == WEATHER_ACTION:
        return weather.get_weather(json_obj)
    return "Action not recognized or incomplete JSON."


def hear_mode():
    """Activate the hear mode to interact with the user."""
    stuff_sayer.say_stuff("Hey, I am Miss Fritters! What would you like to ask?")

    while True:
        prompt = None
        while prompt is None:
            prompt = stuff_hearer.hear_stuff()  # Wait for a valid prompt

        response = ask_stuff(prompt)
        if text_only:
            print(f"Text only mode, response: {response}")
        else:
            stuff_sayer.say_stuff(response)

if __name__ == '__main__':
    hear_mode()
    #stuff_sayer.say_stuff("Hey")
    #prompt = "What the weather is like in Chicago today?"
    #print(ask_stuff(prompt))
