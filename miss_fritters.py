from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import re

from message_source import MessageSource
from tools import get_weather, deck_reload, deck_draw_cards, deck_cards_left, roll_dice, search_web, get_current_time

LLAMA_MODEL = "incept5/llama3.1-claude"

memory = MemorySaver()


# Helper function for role description formatting
def format_role_description(source: MessageSource, user_id: str) -> str:
    base_system_description = """
    Role:
        Your name is Miss Fritters, and you are a helpful chatbot with personal opinions.
        
        Here are the tools available to you, do not call the unless you need them.
        
        get_current_time: returns the current time as an RFC3339 timestamp in US / Central Standard Time.
        search_web: Searches the internet for a term and returns the results.
        roll_dice: used to roll different types of dice.
        get_weather: used to get the temperature from a specific city in Fahrenheit. Please do not use it unless the user gives you a specific city.'
        deck_draw_cards: Used to draw cards from a deck of cards.
        deck_cards_left: Used to find the remaining cards in a deck of cards.
        deck_reload: Shuffles or reloads the deck of cards that is currently active for the user..
        
        For prompts that are mean, you use zoomer slang.
        {source_info}
    """
    return base_system_description.format(source_info=get_source_info(source, user_id))


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
    role_description = format_role_description(source, user_id_clean)
    print(f"Role description: {role_description}")
    print(f"Prompt to ask: {base_prompt}")

    config = {"configurable": {"thread_id": user_id_clean}}
    inputs = {"messages": [("system", role_description), ("user", base_prompt)]}
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


tools = [get_weather, roll_dice, deck_reload, deck_draw_cards, deck_cards_left, search_web, get_current_time]

ollama_instance = ChatOllama(model=LLAMA_MODEL)
graph = create_react_agent(ollama_instance, tools=tools, checkpointer=memory)

# with open("test.png", "wb") as binary_file:
#     binary_file.write(graph.get_graph().draw_mermaid_png())
