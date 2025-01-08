import random
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from message_source import MessageSource
from tts_advanced import AdvancedStuffSayer
from config import Config
from stt import StuffHearer
from langchain_core.tools import tool
import weather_integration
import fritters_constants
import discord_integration

config = Config()
stuff_sayer = None
stuff_hearer = None

text_only = True


def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    role_description = ("Your name is Miss Fritters, and you are a helpful chatbot that has her own personal opinions. "
     "For prompts that are mean, you use zoomer slang and are very opinionated. "
                        "Otherwise, you speak formally and try to answer as objectively as possible. ")
    match source:
        case MessageSource.DISCORD:
            role_description = role_description + "This prompt is coming from a user on Discord with the name \"{}\" - {}".format(user_id, base_prompt)
        case _:
            role_description = role_description + "A user through a cli named {} says the following: {}".format(user_id, base_prompt)
    print(f"Prompt to ask: {base_prompt}")
    default_config = {"configurable": {"user_id": user_id, "conversation_id": "1"}}
    print("Current store: {}".format(store))
    ollama_response = chain_with_message_history.invoke({"role_description": role_description, "prompt": base_prompt}, config=default_config)

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
        stuff_sayer = AdvancedStuffSayer()
        stuff_hearer = StuffHearer()
    #stuff_sayer.say_stuff("Hey")
    #prompt = "Why is the sky blue?"
    #print(ask_stuff(prompt))

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{role_description}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{prompt}"),
])

store = {}  # memory is maintained outside the chain

ollama_instance = (ChatOllama(model=config.get_config(fritters_constants.CONFIG_LLAMA_MODEL)).bind_tools([get_weather, respond_to_user]))

chain = prompt_template | ollama_instance

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryHistory()
    return store[(user_id, conversation_id)]

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="prompt",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)
