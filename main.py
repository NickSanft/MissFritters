import json
import requests
import speech_recognition as sr

from simple_tts import SimpleStuffSayer
from advanced_tts import AdvancedStuffSayer
from config import Config

config = Config()
thing_sayer = AdvancedStuffSayer()
#thing_sayer = SimpleStuffSayer()

r = sr.Recognizer()


def hear_stuff():
    with sr.Microphone(device_index=config.get_config("microphone_device_no")) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("SPEAK")
        audio = r.listen(source)
        try:
            result = r.recognize_google(audio)
            print("Here's what you said: {}".format(result))
        except:
            print("Couldn't hear crap, captain")
    return result


def ask_stuff(prompt_to_ask: str):
    json_obj = {'model': config.get_config("llama_model"), 'prompt': prompt_to_ask}
    print("Prompt to ask: " + prompt_to_ask)
    x = requests.post(config.get_config("llama_url"), json=json_obj)
    result = "Response from model: "

    for line in x.text.splitlines():
        j = json.loads(line)
        result = result + j["response"]
    print(result)
    return result


def hear_mode():
    thing_sayer.say_stuff("Hey, I am Miss Fritters! What would you like to ask?")

    while True:
        prompt = hear_stuff()
        response = ask_stuff(prompt)
        thing_sayer.say_stuff(response)


if __name__ == '__main__':
    #hear_mode()
    thing_sayer.say_stuff("Hello there!")
