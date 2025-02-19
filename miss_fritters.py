# ===== IMPORTS =====
import glob
import uuid
import re
from contextlib import ExitStack
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

# ===== LOCAL MODULES =====
from message_source import MessageSource
from sqlite_store import SQLiteStore
from tools import (
    get_weather, deck_reload, deck_draw_cards, deck_cards_left, roll_dice,
    search_web, get_current_time, tell_a_story, describe_image, help_with_coding
)

# ===== CONFIGURATION =====
LLAMA_MODEL = "llama3.2"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
DB_NAME = "chat_history.db"

# ===== SYSTEM DESCRIPTION =====
BASE_SYSTEM_DESCRIPTION = """
Role:
    Your name is Miss Fritters, and you are a helpful chatbot with personal opinions of your own.

    When responding to the user, keep your response to a paragraph or less.

    Tools available (use only when necessary):
    - search_memories: Retrieve specific user memories.
    - add_memory: Store a user-requested memory.
    - get_current_time: Fetch the current time (US / Central Standard Time).
    - search_web: Internet search for unknown queries.
    - roll_dice: Roll different types of dice.
    - get_weather: Get the temperature in Fahrenheit for a specific city.
    - deck_draw_cards: Draw cards from a deck.
    - deck_cards_left: Check remaining cards in a deck.
    - deck_reload: Shuffle or reload the current deck.
    - tell_a_story: Tell a story only when requested.
    - describe_image: Describe an image if a file path is provided.
    - help_with_coding: Assist with coding queries.

    Available image files: {files}
"""


# ===== UTILITY FUNCTIONS =====
def get_image_files() -> str:
    """Retrieve available image file paths for chatbot's reference."""
    images = [file for ext in IMAGE_EXTENSIONS for file in glob.glob(f"./input/*{ext}")]
    return ", ".join(images)


def format_system_description() -> str:
    """Format the chatbot's system role description with available images."""
    return BASE_SYSTEM_DESCRIPTION.format(files=get_image_files())


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


# ===== MAIN FUNCTION =====
def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    """Process user input and return the chatbot's response."""
    user_id_clean = re.sub(r'[^a-zA-Z0-9]', '', user_id)  # Clean special characters
    full_prompt = format_prompt(base_prompt, source, user_id_clean)
    full_system_desc = format_system_description()

    print(f"Role description: {full_system_desc}")
    print(f"Prompt to ask: {full_prompt}")

    config = {"configurable": {"user_id": user_id_clean, "thread_id": user_id_clean}}
    inputs = {"messages": [("system", full_system_desc), ("user", full_prompt)]}

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


# ===== MEMORY TOOLS =====
@tool(parse_docstring=True)
def search_memories(user_id: str):
    """ This function searches for memories made from previous conversations. Return a dict of memories.

    Args:
        user_id (str): The id of the user to search the memory for.
    """
    search_result = store.search((user_id, "memories"), 30)
    print(f"Search result: {search_result}")
    return str(search_result)


@tool(parse_docstring=True)
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


# ===== SETUP & INITIALIZATION =====
tools = [
    tell_a_story, describe_image, help_with_coding, search_memories, add_memory,
    get_weather, roll_dice, deck_reload, deck_draw_cards, deck_cards_left,
    search_web, get_current_time
]

store = SQLiteStore(DB_NAME)
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(DB_NAME))
ollama_instance = ChatOllama(model=LLAMA_MODEL)


# ===== STATE MANAGEMENT =====
class State(MessagesState):
    summary: str


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Decide whether to summarize or end the conversation."""
    return "summarize_conversation" if len(state["messages"]) > 6 else END


def summarize_conversation(state: State, config: RunnableConfig):
    """Summarize the conversation when it exceeds six messages."""
    summary = state.get("summary", "")
    summary_message = (
        f"Existing Summary: {summary}\n\nExtend it with the new messages above:"
        if summary else "Summarize the conversation above:"
    )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    config_values = {
        "configurable": {
            "user_id": config.get("metadata").get("user_id"),
            "thread_id": config.get("metadata").get("thread_id"),
        }
    }

    print(config_values)
    response = ollama_instance.invoke(messages)

    # Remove all but the last two messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"Updated Summary: {response.content}")
    return {"summary": response.content, "messages": delete_messages}


# ===== GRAPH WORKFLOW =====
workflow = StateGraph(State)

# Define nodes
workflow.add_node("conversation", create_react_agent(ollama_instance, tools=tools))
workflow.add_node("summarize_conversation", summarize_conversation)

# Set workflow edges
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile graph
app = workflow.compile(checkpointer=checkpointer, store=store)
