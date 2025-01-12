from message_source import MessageSource
from miss_fritters import ask_stuff

user_id = "Terrence"

if __name__ == '__main__':
    thing_to_ask = input("What would you like to ask Miss Fritters?\r\n")
    while True:
        response = ask_stuff(thing_to_ask, MessageSource.LOCAL, user_id)
        thing_to_ask = input("\r\n\r\n\r\nRESPONSE FROM MODEL: " + response + "\r\n")