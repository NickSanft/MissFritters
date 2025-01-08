import random
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from message_source import MessageSource
from config import Config
from langchain_core.tools import tool
import weather_integration
import fritters_constants
import discord_integration

config = Config()


def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    base_system_description = """
    Your name is MissFritters, and you are a helpful chatbot that has her own personal opinions.
    For prompts that are mean, you use zoomer slang and are very opinionated.
    Otherwise, you speak formally and try to answer as objectively as possible.
    {source_info}
    """

    match source:
        case MessageSource.DISCORD:
            role_description = base_system_description.format(source_info="This prompt is coming from a user on Discord with the name \"{}\" - {}".format(user_id, base_prompt))
        case _:
            role_description = base_system_description.format(source_info="A user through a cli named {} says the following: {}".format(user_id, base_prompt))

    print(f"Role description: {role_description}")
    print(f"Prompt to ask: {base_prompt}")
    default_config = {"configurable": {"user_id": user_id, "conversation_id": "1"}}
    print("Current store: {}".format(store))
    ollama_response = chain_with_message_history.invoke({"role_description": role_description, "prompt": base_prompt}, config=default_config)

    print(f"Original Response from model: {ollama_response}")
    print(f"Tool calls: {ollama_response.tool_calls}")

    if not ollama_response.tool_calls:
        return ollama_response.content
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
        num_dice (str): The number of dice to roll in string format.
        num_sides (str): The number of dice to roll in string format.
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

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{role_description}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{prompt}"),
])

store = {}  # memory is maintained outside the chain

ollama_instance = (ChatOllama(model=config.get_config(fritters_constants.CONFIG_LLAMA_MODEL))
                   .bind_tools([get_weather, roll_dice, respond_to_user]))


chain = prompt_template | ollama_instance

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        for message in messages:
            if type(message) is AIMessage and message.tool_calls:
                first_result = message.tool_calls[0]
                if first_result:
                    if first_result['name'] == "respond_to_user":
                        message.content = first_result['args']['content']
            print("Adding message of type {} to the store: {}".format(type(message), message))
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

if __name__ == '__main__':
    if config.has_config(fritters_constants.DISCORD_KEY):
        discord_integration.__init__(config)