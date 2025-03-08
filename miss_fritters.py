# ===== IMPORTS =====
import json
import random
import re
import uuid
from contextlib import ExitStack
from datetime import datetime
from typing import Literal

import pytz
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent

import deck_of_cards_integration
import fritters_utils
from kasa_integration import turn_off_lights, turn_on_lights, change_light_color
# ===== LOCAL MODULES =====
from message_source import MessageSource
from sqlite_store import SQLiteStore
from wordle_integration import play_wordle_internal

# ===== CONFIGURATION =====
LLAMA_MODEL = "llama3.2"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
DB_NAME = "chat_history.db"

# Constants for the routing decisions
CONVERSATION_NODE = "conversation"
CODING_NODE = "help_with_coding"
STORY_NODE = "tell_a_story"
HOME_NODE = "home_management"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"


def get_conversation_tools_description():
    """
    Returns a dictionary of available tools and their descriptions.
    """
    conversation_tool_dict = {
        "get_current_time": (get_current_time, "Fetch the current time (US / Central Standard Time)."),
        "search_web": (search_web, "Use only to search the internet if you are unsure about something."),
        "roll_dice": (roll_dice, "Roll different types of dice."),
        "deck_draw_cards": (deck_draw_cards, "Draw cards from a deck."),
        "deck_cards_left": (deck_cards_left, "Check remaining cards in a deck."),
        "deck_reload": (deck_reload, "Shuffle or reload the current deck."),
        "search_memories": (search_memories, "Returns a JSON payload of stored memories you have had with a user."),
        "play_wordle": (play_wordle, "Takes in a word and game number and tries to solve the Wordle.")
    }
    return conversation_tool_dict


def get_home_management_tools_description():
    """
    Returns a dictionary of available tools and their descriptions.
    """
    home_tool_dict = {
        "turn_off_lights": (turn_off_lights, "Turns off the lights."),
        "turn_on_lights": (turn_on_lights, "Turns on the lights."),
        "turn_off_bedroom_lights": (turn_off_lights, "Turns off the bedroom lights."),
        "turn_on_bedroom_lights": (turn_on_lights, "Turns on the bedroom lights."),
        "change_light_color": (change_light_color, "Changes light color. Accepts a valid hue in degrees.")
    }

    return home_tool_dict


# ===== UTILITY FUNCTIONS =====
def get_system_description(tools: dict[str, tuple[BaseTool, str]]):
    """
    Format the chatbot's system role description dynamically by including tools from the list.
    """
    # Set tools list dynamically
    tool_descriptions = "".join(
        [f"    {tool_name}: {tup[1]}\n" for tool_name, tup in tools.items()]
    )

    return f"""
Role:
    You are an AI conversationalist named Miss Fritters, you respond to the user's messages with witty, sassy, upbeat dialog.
    You do retain memories per user, and can use the search_memories tool to retrieve them.
    When responding to the user, keep your response to a paragraph or less.

Tools:
{tool_descriptions}
    """


def get_source_info(source: MessageSource, user_id: str) -> str:
    """Generate source information based on the messaging platform."""
    if source == MessageSource.DISCORD_TEXT:
        return f"User is texting from Discord (User ID: {user_id})"
    elif source == MessageSource.DISCORD_VOICE:
        return f"User is speaking from Discord (User ID: {user_id}). Answer in 10 words or less."
    return f"User is interacting via CLI (User ID: {user_id})"


def format_prompt(prompt: str, source: MessageSource, user_id: str) -> str:
    """Format the final prompt for the chatbot."""
    return f"""
    Context:
        {get_source_info(source, user_id)}
    Question:
        {prompt}
    """


def search_memories_internal(config: RunnableConfig):
    user_id = config.get("metadata").get("user_id")
    search_result = store.search((user_id, "memories"), 30)
    summaries = {}
    for _, summary_dict in search_result:
        for key, summary in summary_dict.items():
            summaries[key] = summary
    json_summaries = json.dumps(summaries)
    print(json_summaries)
    return json_summaries


@tool(parse_docstring=True, return_direct=True)
def play_wordle(word: str, game_number: int):
    """
    Attempts to play a game of world with a provided word and game number.

    Args:
    word: The word to guess.
    game_number: The game number.
    """
    return play_wordle_internal(word, game_number)


@tool(parse_docstring=True)
def get_current_time():
    """
    Returns the current time as a string in RFC3339 (YYYY-MM-DDTHH:MM:SS) format.

    Example - 2025-01-13T23:11:56.337644-06:00
    """
    return get_current_time_internal()


def get_current_time_internal():
    # Get the current time in UTC
    utc_now = datetime.now(pytz.utc)

    # Convert to CST (Central Standard Time)
    cst = pytz.timezone('US/Central')
    cst_now = utc_now.astimezone(cst)

    # Format the timestamp in RFC3339 format
    rfc3339_timestamp = cst_now.isoformat()

    print(rfc3339_timestamp)
    return rfc3339_timestamp


@tool(parse_docstring=True)
def search_web(text_to_search: str):
    """
    Takes in a string and returns results from the internet.

    Args:
    text_to_search: The text to search the internet for information.

    Returns:
    list: A list of dictionaries, each containing string keys and string values representing the search results.
    """
    results = DDGS().text(text_to_search, max_results=5)
    print(results)
    return results


@tool(parse_docstring=True)
def roll_dice(num_dice: int, num_sides: int, config: RunnableConfig):
    """
    Rolls a specified number of dice, each with a specified number of sides.

    Args:
    num_dice: The number of dice to roll.
    num_sides: The number of sides on each die.
    config: The RunnableConfig.

    Returns:
    list: A list containing the result of each die roll.
    """
    user_id = config.get("metadata").get("user_id")
    if num_dice <= 0 or num_sides <= 0:
        raise ValueError("Both number of dice and number of sides must be positive integers.")

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return (f"Here are the results: {user_id}."
            f" {rolls}")


@tool(parse_docstring=True, return_direct=True)
def deck_cards_left(config: RunnableConfig) -> str:
    """If the user asks how many cards are left, this will return the number of cards left in their deck.

    Args:
        config: The RunnableConfig.
    """
    user_id = config.get("metadata").get("user_id")
    return deck_of_cards_integration.get_remaining_card_number(user_id)


@tool(parse_docstring=True, return_direct=True)
def deck_reload(config: RunnableConfig) -> str:
    """If a user asks to reload their deck, this should be called.

    Args:
        config: The RunnableConfig.
    """
    user_id = config.get("metadata").get("user_id")
    return deck_of_cards_integration.reload_deck(user_id)


@tool(parse_docstring=True, return_direct=True)
def deck_draw_cards(number_of_cards: int, config: RunnableConfig) -> str:
    """If someone asks to draw cards, this should be called.

    Args:
        number_of_cards: The number of cards to draw.
        config: The RunnableConfig.
    """
    user_id = config.get("metadata").get("user_id")
    return deck_of_cards_integration.draw_cards(number_of_cards, user_id)


@tool(parse_docstring=True)
def search_memories(config: RunnableConfig):
    """ This function returns memories in JSON format.

    Args:
        config: The RunnableConfig.
    """
    print("TOOL CALLED")
    return search_memories_internal(config)


def add_memory(user_id: str, memory_key: str, memory_to_store: str):
    """ This function stores a memory. Only use this if the user has asked you to.

    Args:
        user_id (str): The id of the user to store the memory.
        memory_key (str): A unique identifier for the memory.
        memory_to_store (str): The memory you wish to store.
    """
    memory_dict = {memory_key: memory_to_store}
    store.put((user_id, "memories"), str(uuid.uuid4()), memory_dict)
    return "Added memory for {}: {}".format(memory_key, memory_to_store)


# ===== MAIN FUNCTION =====
def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    """Process user input and return the chatbot's response."""
    user_id_clean = re.sub(r'[^a-zA-Z0-9]', '', user_id)  # Clean special characters
    full_prompt = format_prompt(base_prompt, source, user_id_clean)

    system_prompt = get_system_description(get_conversation_tools_description())
    print(f"Role description: {system_prompt}")
    print(f"Prompt to ask: {full_prompt}")

    config = {"configurable": {"user_id": user_id_clean, "thread_id": user_id_clean}}
    inputs = {"messages": [("user", full_prompt)]}

    return print_stream(app.stream(inputs, config=config, stream_mode="values"))


def print_stream(stream):
    """Process and print streamed messages."""
    message = ""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    return message.content


# ===== SETUP & INITIALIZATION =====
conversation_tools = [tool_info[0] for tool_info in get_conversation_tools_description().values()]
print(conversation_tools)

home_tools = [tool_info[0] for tool_info in get_home_management_tools_description().values()]
print(home_tools)

store = SQLiteStore(DB_NAME)
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(DB_NAME))
llama_instance = ChatOllama(model=LLAMA_MODEL)

MISTRAL_MODEL = "mistral"
mistral_instance = ChatOllama(model=MISTRAL_MODEL)

CODE_MODEL = "codellama"
code_instance = ChatOllama(model=CODE_MODEL)

MISTRAL_ORCA_MODEL = "mistral-openorca"
orca_instance = ChatOllama(model=MISTRAL_ORCA_MODEL)

HERMES_MODEL = "hermes3"
hermes_instance = ChatOllama(model=HERMES_MODEL)

conversation_react_agent = create_react_agent(llama_instance, tools=conversation_tools)
home_management_react_agent = create_react_agent(llama_instance, tools=home_tools)


def supervisor_routing(state: MessagesState, config: RunnableConfig):
    """Handles general conversation, calling appropriate helpers for specific tasks."""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    user_id = config.get("metadata").get("user_id")

    home_management_desc = ""
    home_management_example = ""
    if fritters_utils.check_root_user(user_id):
        home_management_desc = f"\"{HOME_NODE}\" - use if the user is requesting you to manage lights in a home."
        home_management_example = f"- \"Can you turn off the lights?\" -> \"{HOME_NODE}\""

    supervisor_prompt = f"""
    Your response must always be one of the following options:
    "{CONVERSATION_NODE}" - used by default.
    "{CODING_NODE}" - use if the user is asking for something code-related.
    "{STORY_NODE}" - use if the user is asking you tell a story.
    {home_management_desc}

    Do NOT generate any additional text or explanations.
    Only return one of the above values as the complete response.
    Example inputs and expected outputs:
    - "Can you help me with a Python script to list all values in a dict" → "{CODING_NODE}"
    - "Can you tell me a story about frogs?" → "{STORY_NODE}"
    - "How are you doing?" → "{CONVERSATION_NODE}"
    {home_management_example}
    """
    print(f"Supervisor prompt: {supervisor_prompt}")
    inputs = [("system", supervisor_prompt), ("user", latest_message)]
    original_response = hermes_instance.invoke(inputs)
    route = original_response.content.lower()
    print(f"ROUTE DETERMINED: {route}")
    if route not in [CODING_NODE, STORY_NODE, CONVERSATION_NODE, HOME_NODE]:
        print("This bot went a little crazy, defaulting to conversation.")
        route = CONVERSATION_NODE
    return route


def should_continue(state: MessagesState) -> Literal["summarize_conversation", END]:
    """Decide whether to summarize or end the conversation."""
    return SUMMARIZE_CONVERSATION_NODE if len(state["messages"]) > 15 else END


def tell_a_story(state: MessagesState, config: RunnableConfig):
    """Handles requests to tell a story."""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    inputs = [
        ("system", "You are a ChatBot that receives a prompt and tells a story based off of it."),
        ("user", latest_message)]
    resp = orca_instance.invoke(inputs, config=get_config_values(config))
    return {'messages': [resp]}


def help_with_coding(state: MessagesState, config: RunnableConfig):
    """Handles requests for coding help."""
    print("In: help_with_coding")
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    inputs = [
        ("system", "You are a ChatBot that assists with writing or explaining code."),
        ("user", latest_message)]
    code_resp = code_instance.invoke(inputs, config=get_config_values(config))
    return {'messages': [code_resp]}


def summarize_conversation(state: MessagesState, config: RunnableConfig):
    print("In: summarize_conversation")
    user_id = config.get("metadata").get("user_id")
    summary_message_prompt = "Please summarize the conversation above:"
    messages = state["messages"]
    # messages[-1].content = messages[-1].content + "\r\n I am wrapping up this conversation and starting a new one :)"
    messages = messages + [HumanMessage(content=summary_message_prompt)]
    summary_response = llama_instance.invoke(messages)
    timestamp = get_current_time_internal()
    summary = f"Summary made at {timestamp} \r\n {summary_response.content}"
    print(f"Summary: {summary}")
    response_key_inputs = [
        ("system",
         "Please provide a short sentence describing this memory starting with the word \"memory\". Example - memory_of_pie"),
        ("user", summary)]
    summary_response_key = llama_instance.invoke(response_key_inputs, config=get_config_values(config))
    print(f"Summary Key: {summary_response_key.content}")
    add_memory(user_id, summary_response_key.content, summary)
    # Remove all but the last message
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-1]]

    return {"messages": delete_messages}


def conversation(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    print(f"Latest messsage: {latest_message}")
    inputs = {"messages": [("system", get_system_description(get_conversation_tools_description())),
                           ("user", latest_message)]}
    resp = print_stream(conversation_react_agent.stream(inputs, config=get_config_values(config), stream_mode="values"))
    return {'messages': [resp]}


def home_management(state: MessagesState, config: RunnableConfig):
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    print(f"Latest messsage: {latest_message}")
    inputs = {"messages": [("system", get_system_description(get_home_management_tools_description())),
                           ("user", latest_message)]}
    resp = print_stream(
        home_management_react_agent.stream(inputs, config=get_config_values(config), stream_mode="values"))
    return {'messages': [resp]}


def get_config_values(config: RunnableConfig):
    config_values = {
        "configurable": {
            "user_id": config.get("metadata").get("user_id"),
            "thread_id": config.get("metadata").get("thread_id"),
        }
    }
    return config_values


# ===== GRAPH WORKFLOW =====
workflow = StateGraph(MessagesState)

# Define nodes
workflow.add_node(CONVERSATION_NODE, conversation)
workflow.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation)
workflow.add_node(CODING_NODE, help_with_coding)
workflow.add_node(STORY_NODE, tell_a_story)
workflow.add_node(HOME_NODE, home_management)

# Set workflow edges
workflow.add_conditional_edges(START, supervisor_routing,
                               {CONVERSATION_NODE: CONVERSATION_NODE, CODING_NODE: CODING_NODE, STORY_NODE: STORY_NODE,
                                HOME_NODE: HOME_NODE})
workflow.add_conditional_edges(CONVERSATION_NODE, should_continue)
workflow.add_conditional_edges(CODING_NODE, should_continue)
workflow.add_conditional_edges(STORY_NODE, should_continue)
workflow.add_conditional_edges(HOME_NODE, should_continue)
workflow.add_edge(SUMMARIZE_CONVERSATION_NODE, END)

# Compile graph
app = workflow.compile(checkpointer=checkpointer, store=store)


# with open("mermaid_diagram.png", "wb") as binary_file:
#     binary_file.write(app.get_graph().draw_mermaid_png())


def test_asking_stuff():
    ask_stuff("Hi there!", MessageSource.DISCORD_TEXT, "hello")
    ask_stuff("Apple pie is my favorite, what is your favorite pie?", MessageSource.DISCORD_TEXT, "hello")
    ask_stuff("What other desserts are similar to pie?", MessageSource.DISCORD_TEXT, "hello")
    ask_stuff("What pie is the most famous in New York?", MessageSource.DISCORD_TEXT, "hello")
    ask_stuff("I am tired", MessageSource.DISCORD_TEXT, "hello")
    ask_stuff("Thanks", MessageSource.DISCORD_TEXT, "hello")
    ask_stuff("Wow", MessageSource.DISCORD_TEXT, "hello")
