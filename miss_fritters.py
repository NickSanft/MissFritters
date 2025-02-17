import glob
import uuid

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
import re
from contextlib import ExitStack
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

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
    ollama_response = print_stream(graph.stream(inputs, config=config, stream_mode="values"))

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
graph = create_react_agent(ollama_instance, tools=tools, checkpointer=checkpointer, store=store)

# with open("test.png", "wb") as binary_file:
#     binary_file.write(graph.get_graph().draw_mermaid_png())
