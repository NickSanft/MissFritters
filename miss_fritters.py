from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from message_source import MessageSource
from tools import get_weather, deck_reload, deck_draw_cards, deck_cards_left, \
    roll_dice

LLAMA_MODEL = "incept5/llama3.1-claude"

memory = MemorySaver()


# Helper function for role description formatting
def format_role_description(source: MessageSource, user_id: str) -> str:
    base_system_description = """
    Role:
        Your name is Miss Fritters, and you are a helpful chatbot with personal opinions and respond to she, her, ma'am, or miss.

        For prompts that are mean, you use zoomer slang. Otherwise, you speak normally.
        {source_info}
    """
    return base_system_description.format(source_info=get_source_info(source, user_id))


def get_source_info(source: MessageSource, user_id: str) -> str:
    if source == MessageSource.DISCORD:
        return f"This human is coming from Discord with the user_id \"{user_id}\""
    return f"A user through a CLI with the user_id {user_id}"


# Main function to ask questions with specific tools
def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    role_description = format_role_description(source, user_id)
    print(f"Role description: {role_description}")
    print(f"Prompt to ask: {base_prompt}")

    config = {"configurable": {"thread_id": user_id}}
    inputs = {"messages":[("system", role_description), ("user", base_prompt)]}
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


tools = [get_weather, roll_dice, deck_reload, deck_draw_cards, deck_cards_left]

ollama_instance = ChatOllama(model=LLAMA_MODEL)
graph = create_react_agent(ollama_instance, tools=tools, checkpointer=memory)