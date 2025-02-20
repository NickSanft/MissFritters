# ===== IMPORTS =====
import glob
import re
from contextlib import ExitStack
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
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
    search_web, get_current_time
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
    - get_current_time: Fetch the current time (US / Central Standard Time).
    - search_web: Use only to search the internet if you are unsure about something.
    - roll_dice: Roll different types of dice.
    - get_weather: Get the temperature in Fahrenheit for a specific city.
    - deck_draw_cards: Draw cards from a deck.
    - deck_cards_left: Check remaining cards in a deck.
    - deck_reload: Shuffle or reload the current deck.
"""

# - search_memories: Retrieve specific user memories.
# - add_memory: Store a user-requested memory.

# Constants for the routing decisions
CONVERSATION_NODE = "conversation"
CODING_NODE = "help_with_coding"
STORY_NODE = "tell_a_story"
SUMMARIZE_CONVERSATION_NODE = "summarize_conversation"


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


# ===== SETUP & INITIALIZATION =====
tools = [
    get_weather, roll_dice, deck_reload, deck_draw_cards, deck_cards_left,
    search_web, get_current_time
]

store = SQLiteStore(DB_NAME)
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(DB_NAME))
ollama_instance = ChatOllama(model=LLAMA_MODEL)

MISTRAL_MODEL = "mistral"
mistral_instance = ChatOllama(model=MISTRAL_MODEL)

CODE_MODEL = "codellama"
code_instance = ChatOllama(model=CODE_MODEL)


# ===== STATE MANAGEMENT =====
class State(MessagesState):
    summary: str


def supervisor_routing(state: State, config: RunnableConfig):
    """Handles general conversation, calling appropriate helpers for specific tasks."""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""

    supervisor_prompt = """
    Your response must always be one of the following options:
    "conversation" - used by default.
    "help_with_coding" - use if the user is asking for something code-related.
    "tell_a_story" - use if the user is asking you tell a story.

    Do NOT generate any additional text or explanations.
    Only return one of the above values as the complete response.
    Example inputs and expected outputs:
    - "Can you help me with a Python script to list all values in a dict" → "HELP_WITH_CODING"
    - "Can you tell me a story about frogs?" → "TELL_A_STORY"
    - "How are you doing?" → "OTHER"
    """
    inputs = [("system", supervisor_prompt), ("user", latest_message)]
    original_response = ollama_instance.invoke(inputs, config=get_config_values(config))
    print("ROUTE DETERMINED: " + original_response.content)

    return original_response.content.lower()


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Decide whether to summarize or end the conversation."""
    return "summarize_conversation" if len(state["messages"]) > 6 else END


def tell_a_story(state: State, config: RunnableConfig):
    """Handles requests to tell a story."""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    inputs = [
        ("system", "You are a ChatBot that receives a prompt and tells a story based off of it."),
        ("user", latest_message)]
    resp = mistral_instance.invoke(inputs, config=get_config_values(config))
    return {'messages': [resp]}


def help_with_coding(state: State, config: RunnableConfig):
    """Handles requests for coding help."""
    messages = state["messages"]
    latest_message = messages[-1].content if messages else ""
    inputs = [
        ("system", "You are a ChatBot that assists with writing or explaining code."),
        ("user", latest_message)]
    code_resp = code_instance.invoke(inputs, config=get_config_values(config))
    return {'messages': [code_resp]}


def summarize_conversation(state: State, config: RunnableConfig):
    """Summarize the conversation when it exceeds six messages."""
    summary = state.get("summary", "")
    summary_message = (
        f"Existing Summary: {summary}\n\nExtend it with the new messages above:"
        if summary else "Summarize the conversation above:"
    )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = ollama_instance.invoke(messages, config=get_config_values(config))

    # Remove all but the last two messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    print(f"Updated Summary: {response.content}")
    return {"summary": response.content, "messages": delete_messages}


def get_config_values(config: RunnableConfig):
    config_values = {
        "configurable": {
            "user_id": config.get("metadata").get("user_id"),
            "thread_id": config.get("metadata").get("thread_id"),
        }
    }
    return config_values


# ===== GRAPH WORKFLOW =====
workflow = StateGraph(State)

# Define nodes
workflow.add_node(CONVERSATION_NODE, create_react_agent(ollama_instance, tools=tools))
workflow.add_node(SUMMARIZE_CONVERSATION_NODE, summarize_conversation)
workflow.add_node(CODING_NODE, help_with_coding)
workflow.add_node(STORY_NODE, tell_a_story)

# Set workflow edges
workflow.add_conditional_edges(START, supervisor_routing,
                               {CONVERSATION_NODE: CONVERSATION_NODE, CODING_NODE: CODING_NODE, STORY_NODE: STORY_NODE})
workflow.add_conditional_edges(CONVERSATION_NODE, should_continue)
workflow.add_conditional_edges(CODING_NODE, should_continue)
workflow.add_conditional_edges(STORY_NODE, should_continue)
workflow.add_edge(SUMMARIZE_CONVERSATION_NODE, END)

# Compile graph
app = workflow.compile(checkpointer=checkpointer, store=store)

# ask_stuff("Can you write a Python script that prints the numbers 1-20?", MessageSource.DISCORD_TEXT, "hello")
# ask_stuff("apple pie is my favorite", MessageSource.DISCORD_TEXT, "hello")
# ask_stuff("Can you tell me a story about pandas?", MessageSource.DISCORD_TEXT, "hello")
#
# with open("test.png", "wb") as binary_file:
#     binary_file.write(app.get_graph().draw_mermaid_png())
