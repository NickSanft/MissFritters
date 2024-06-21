import json
import pyttsx3
import requests
import speech_recognition as sr

from config import Config

config = Config()

engine = pyttsx3.init()
engine.setProperty('voice', "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0")

r = sr.Recognizer()

def say_stuff(message: str):
    print("Message to say: {}".format(message))
    engine.say(message)
    engine.runAndWait()


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


def print_config():
    print(sr.Microphone.list_microphone_names())
    print(sr.Microphone.list_microphone_names()[2])

    voices = engine.getProperty('voices')
    for voice in voices:
        # to get the info. about various voices in our PC
        print("Voice:")
        print("ID: %s" % voice.id)
        print("Name: %s" % voice.name)
        print("Age: %s" % voice.age)
        print("Gender: %s" % voice.gender)
        print("Languages Known: %s" % voice.languages)


if __name__ == '__main__':

    say_stuff("Hey, I am Miss Fritters! What would you like to ask?")

    while True:
        prompt = hear_stuff()
        response = ask_stuff(prompt)
        say_stuff(response)
