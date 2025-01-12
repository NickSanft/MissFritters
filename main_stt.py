from miss_fritters import ask_stuff
from message_source import MessageSource
from stt import StuffHearer
from tts_advanced import AdvancedStuffSayer

stuff_sayer = AdvancedStuffSayer()
stuff_hearer = StuffHearer()
text_only = False

def hear_mode():
    """Activate the hear mode to interact with the user."""
    stuff_sayer.say_stuff("Hey, I am Miss Fritters! What would you like to ask?")

    while True:
        prompt = None
        while prompt is None:
            prompt = stuff_hearer.hear_stuff()  # Wait for a valid prompt

        response = ask_stuff(prompt, MessageSource.LOCAL, "0")
        print("Final response: {}".format(response))
        if text_only:
            print(f"Text only mode, response: {response}")
        else:
            stuff_sayer.say_stuff(response)


if __name__ == '__main__':
    hear_mode()
    #stuff_sayer.say_stuff("Hey")
    #prompt = "Why is the sky blue?"
    #print(ask_stuff(prompt))