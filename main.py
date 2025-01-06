import random

from langchain_ollama import ChatOllama

from message_source import MessageSource
from tts_advanced import AdvancedStuffSayer
from config import Config
from stt import StuffHearer
from langchain_core.tools import tool
import weather_integration
import fritters_constants
import discord_integration


config = Config()
stuff_sayer = AdvancedStuffSayer()
stuff_hearer = StuffHearer()

text_only = True

ROLE_DESCRIPTION = ("Your name is Miss Fritters, and you are a helpful chatbot that has her own personal opinions. "
                    "For users that are nice, you speak formally and try to answer as objectively as possible. "
                    "For users that are mean, you are obnoxiously nice to them, use zoomer slang, and are very opinionated.")

def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    match source:
        case MessageSource.DISCORD:
            prompt = "The following is coming from a user on Discord with the name \"{}\" - {}".format(user_id, base_prompt)
        case _:
            prompt = "A user through a cli named {} says the following: {}".format(user_id, base_prompt)
    print(f"Prompt to ask: {prompt}")
    full_request = [
        ("system", ROLE_DESCRIPTION),
        ("human",  prompt)
    ]
    ollama_response = ollama_instance.invoke(full_request)
    print(f"Original Response from model: {ollama_response}")
    print(f"Tool calls: {ollama_response.tool_calls}")
    first_result = ollama_response.tool_calls[0]
    first_result_args = first_result['args']
    what_to_call = first_result['name']

    print(f"What to call: {what_to_call}")
    match what_to_call:
        case "respond_to_user":
            return respond_to_user(first_result_args['content'])
        case "roll_dice":
            print("Rollin dice...")
            a = str(first_result_args['num_dice'])
            b = str(first_result_args['num_sides'])
            return roll_dice(a, b)
        case "get_weather":
            return get_weather(first_result_args['city'])
        case _:
            response_back = "Unknown tool_call: {} with args: {}".format(what_to_call, first_result_args)
            print(response_back)
            return response_back

@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """Receives a city and gets the current weather from an API.

    Args:
        city (str): The name of the city.
    """
    return weather_integration.get_weather(city)

@tool(parse_docstring=True)
def roll_dice(num_dice: str, num_sides: str) -> str:
    """Rolls a number of dice.

    Args:
        num_dice (str): The number of dice to roll in str format.
        num_sides (str): The number of dice to roll in str format.
    """
    print("Preparing to roll {} {}-sided dice...".format(num_dice, num_sides))
    results = []
    for _ in range(int(num_dice)):
        results.append(random.randint(1, int(num_sides)))
    return "Here are the results: of rolling {} {}-sided dice\r\n {}".format(num_dice, num_sides, results)

@tool(parse_docstring=True)
def respond_to_user(content: str) -> str:
    """This tool call should always be used as default response when a specific tool is not determined.

    Args:
        content (str): The response to return to the user.
    """
    print(f"Response to user: {content}")
    return content

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
    if config.has_config(fritters_constants.DISCORD_KEY):
        discord_integration.__init__(config)
    else:
        hear_mode()
    #stuff_sayer.say_stuff("Hey")
    #prompt = "Why is the sky blue?"
    #print(ask_stuff(prompt))

ollama_instance = ChatOllama(model=config.get_config(fritters_constants.CONFIG_LLAMA_MODEL), format="json").bind_tools([roll_dice, get_weather, respond_to_user])