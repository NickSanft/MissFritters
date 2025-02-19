import glob
import uuid
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import create_react_agent
import re
from contextlib import ExitStack
from langgraph.checkpoint.sqlite import SqliteSaver

from message_source import MessageSource
from sqlite_store import SQLiteStore
from tools import get_weather, deck_reload, deck_draw_cards, deck_cards_left, roll_dice, search_web, get_current_time, \
    tell_a_story, describe_image, help_with_coding

LLAMA_MODEL = "llama3.2"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

base_system_description = """
Role:
    Your name is Miss Fritters, and you are a helpful chatbot with personal opinions of your own.
    
    When responding back to the user, please try to keep the response to a paragraph or lower.

    Here are the tools available to you, do not call these tools unless you need to.

    search_memories: Use only to search for specific memories for a given user.
    
    add_memory: can be used to add a memory if requested by the user.
    
    get_current_time: returns the current time as an RFC3339 timestamp in US / Central Standard Time.

    search_web: Searches the internet for a term and returns the results. Use this if you are unsure what the user is asking.

    roll_dice: used to roll different types of dice.

    get_weather: used to get the temperature from a specific city in Fahrenheit. Please do not use it unless the user gives you a specific city.'

    deck_draw_cards: Used to draw cards from a deck of cards.

    deck_cards_left: Used to find the remaining cards in a deck of cards.

    deck_reload: Shuffles or reloads the deck of cards that is currently active for the user.
    
    tell_a_story: Only use this if the user asks you to tell a story.  
    
    describe_image: Only use this if a file path is provided. Used to describe an image. The filepaths you have are: {files}
    
    help_with_coding: Only use this to help a user with coding.
"""


def format_system_description() -> str:
    # List to store all image files
    images = []

    # Loop through each image pattern and append matching files
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(f"./input/*{ext}"))

    # Now `images` contains all matching image files
    print(images)
    return base_system_description.format(files=images)


# Helper function for role description formatting
def format_prompt(prompt: str, source: MessageSource, user_id: str) -> str:
    prompt_template = """ 
    Context:
        {source_info}
    Question:
        {prompt}
    """
    return prompt_template.format(prompt=prompt, source_info=get_source_info(source, user_id))


def get_source_info(source: MessageSource, user_id: str) -> str:
    if source == MessageSource.DISCORD_TEXT:
        return f"This human is texting from Discord with the user_id \"{user_id}\""
    elif source == MessageSource.DISCORD_VOICE:
        return f"This human is coming from Discord with the user_id \"{user_id}\". Please answer in 10 words or less if possible."
    return f"A user through a CLI with the user_id {user_id}"


# Main function to ask questions with specific tools
def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    # Remove special characters because people love to have underscores in their usernames.
    user_id_clean = re.sub(r'[^a-zA-Z0-9]', '', user_id)
    full_prompt = format_prompt(base_prompt, source, user_id_clean)
    full_system_desc = format_system_description()
    print(f"Role description: {full_system_desc}")
    print(f"Prompt to ask: {full_prompt}")

    config = {"configurable": {"user_id": user_id_clean, "thread_id": user_id_clean}}
    inputs = {"messages": [("system", full_system_desc), ("user", full_prompt)]}
    ollama_response = print_stream(app.stream(inputs, config=config, stream_mode="values"))

    print(f"Original Response from model: {ollama_response}")
    return ollama_response


def print_stream(stream):
    message = ""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
    return message.content


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


tools = [tell_a_story, describe_image, help_with_coding, search_memories, add_memory, get_weather, roll_dice, deck_reload,
         deck_draw_cards, deck_cards_left, search_web, get_current_time]

db_name = "chat_history.db"
store = SQLiteStore(db_name)

exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(db_name))
ollama_instance = ChatOllama(model=LLAMA_MODEL)

# We will add a `summary` attribute (in addition to `messages` key,
# which MessagesState already has)
class State(MessagesState):
    summary: str

# We now define the logic for determining whether to end or summarize the conversation
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Return the next node to execute."""
    messages = state["messages"]
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    # Otherwise we can just end
    return END

def summarize_conversation(state: State, config: RunnableConfig):
    # First, we summarize the conversation
    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, we use a different system prompt
        # to summarize it than if one didn't
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    config_values = {"configurable": {"user_id": config.get("metadata").get("user_id"), "thread_id": config.get("metadata").get("thread_id")}}
    print(config_values)
    response = ollama_instance.invoke(messages)
    # We now need to delete messages that we no longer want to show up
    # I will delete all but the last two messages, but you can change this
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    print(f"Summary of conversation earlier: {summary_message}")
    return {"summary": response.content, "messages": delete_messages}

# Define a new graph
workflow = StateGraph(State)

# Define the conversation node and the summarize node
workflow.add_node("conversation", create_react_agent(ollama_instance, tools=tools))
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `conversation`.
    # This means these are the edges taken after the `conversation` node is called.
    "conversation",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `summarize_conversation` to END.
# This means that after `summarize_conversation` is called, we end.
workflow.add_edge("summarize_conversation", END)

# Finally, we compile it!
app = workflow.compile(checkpointer=checkpointer, store=store)
