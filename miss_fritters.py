import glob
import uuid

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import re
from contextlib import ExitStack
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

from message_source import MessageSource
from tools import get_weather, deck_reload, deck_draw_cards, deck_cards_left, roll_dice, search_web, get_current_time, \
    tell_a_story, describe_image

PERSISTENT = True
LLAMA_MODEL = "llama3.2"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

memory = MemorySaver()

base_system_description = """
Role:
    Your name is Miss Fritters, and you are a helpful chatbot with personal opinions of your own.
    
    When responding back to the user, please try to keep the response to a paragraph or lower.

    Here are the tools available to you, do not call these tools unless you need to.

    search_memories: can be provided a query to print a list of memories for a given user.
    
    add_memory: can be used to add a memory if requested by the user.
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
def search_memories(user_id: str, search_query: str):
    """ This function searches for memories made from previous conversations. Return a dict of memories.

    Args:
        user_id (str): The id of the user to search the memory for.
        search_query (str): The query to search memories of previous conversations with the user.
    """
    search_result = store.search(
        (user_id, "memories"),
        query=search_query,
        limit=3  # Return top 3 matches
    )
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


# tools = [tell_a_story, get_weather, roll_dice, deck_reload, deck_draw_cards, deck_cards_left, search_web, get_current_time, describe_image]
tools = [search_memories, add_memory]

if PERSISTENT:
    db_name = "chat_history.db"
else:
    db_name = ":memory:"

store = InMemoryStore()
exit_stack = ExitStack()
checkpointer = exit_stack.enter_context(SqliteSaver.from_conn_string(db_name))
ollama_instance = ChatOllama(model=LLAMA_MODEL)
graph = create_react_agent(ollama_instance, tools=tools, checkpointer=checkpointer, store=store)

# with open("test.png", "wb") as binary_file:
#     binary_file.write(graph.get_graph().draw_mermaid_png())
