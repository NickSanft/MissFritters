import json
import requests

from simple_tts import SimpleStuffSayer
from advanced_tts import AdvancedStuffSayer
from config import Config
from stt import StuffHearer

config = Config()
stuff_sayer = AdvancedStuffSayer()
#stuff_sayer = SimpleStuffSayer()
stuff_hearer = StuffHearer()

config_llama_model = "llama_model"
config_llama_url = "llama_url"


def ask_stuff(prompt_to_ask: str):
    json_obj = {'model': config.get_config(config_llama_model), 'prompt': prompt_to_ask}
    print("Prompt to ask: " + prompt_to_ask)
    x = requests.post(config.get_config(config_llama_url), json=json_obj)
    result = "Response from model: "

    for line in x.text.splitlines():
        j = json.loads(line)
        result = result + j["response"]
    print(result)
    return result


def hear_mode():
    stuff_sayer.say_stuff("Hey, I am Miss Fritters! What would you like to ask?")
    while True:
        prompt = stuff_hearer.hear_stuff()
        response = ask_stuff(prompt)
        stuff_sayer.say_stuff(response)


if __name__ == '__main__':
    hear_mode()
    #stuff_sayer.say_stuff("Hello!")
