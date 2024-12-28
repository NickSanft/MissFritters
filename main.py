from playsound import playsound

from tts_simple import SimpleStuffSayer
from tts_advanced import AdvancedStuffSayer
from config import Config
from stt import StuffHearer
from langchain_community.llms import Ollama

config = Config()
stuff_sayer = AdvancedStuffSayer()
#stuff_sayer = SimpleStuffSayer()
stuff_hearer = StuffHearer()


config_llama_model = "llama_model"
ollama_instance = Ollama(model=config.get_config(config_llama_model))


def ask_stuff(prompt_to_ask: str):
    print("Prompt to ask: " + prompt_to_ask)
    result = ollama_instance.invoke(prompt_to_ask)
    print("Response from model: {}".format(result).format(result))
    return result


def hear_mode():
    stuff_sayer.say_stuff("Hey, I am Miss Fritters! What would you like to ask?")
    while True:
        prompt = None
        while prompt is None:
            prompt = stuff_hearer.hear_stuff()
        response = ask_stuff(prompt)
        stuff_sayer.say_stuff(response)

if __name__ == '__main__':
    #hear_mode()
    stuff_sayer.say_stuff("Hello, you are a good dog!")
    #stuff_sayer.say_stuff(ask_stuff("What is the weather like in Chicago?"))
