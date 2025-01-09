from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama import ChatOllama

from sqlite_memory import get_session_history
from message_source import MessageSource
import discord_integration
from tools import get_weather, respond_to_user, handle_tool_calls

LLAMA_MODEL = "llama3.1"
HISTORY_KEY = "history"
PROMPT_KEY = "prompt"
USER_ID_KEY = "user_id"
CONVERSATION_ID_KEY = "conversation_id"

use_discord_integration = True

# Setup the chain with the message history
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{role_description}"),
    MessagesPlaceholder(variable_name=HISTORY_KEY),
    ("human", "{prompt}"),
])


# Helper function for role description formatting
def format_role_description(base_prompt: str, source: MessageSource, user_id: str) -> str:
    base_system_description = """
    Your name is MissFritters, and you are a helpful chatbot with personal opinions.
    For prompts that are mean, you use zoomer slang and are very opinionated.
    Otherwise, you speak formally and try to answer as objectively as possible.
    {source_info}
    """
    return base_system_description.format(source_info=get_source_info(source, base_prompt, user_id))

def get_source_info(source: MessageSource, base_prompt: str, user_id: str) -> str:
    if source == MessageSource.DISCORD:
        return f"This prompt is coming from a user on Discord with the name \"{user_id}\" - {base_prompt}"
    return f"A user through a CLI named {user_id} says: {base_prompt}"

# Main function to ask questions with specific tools
def ask_stuff(base_prompt: str, source: MessageSource, user_id: str) -> str:
    role_description = format_role_description(base_prompt, source, user_id)
    print(f"Role description: {role_description}")
    print(f"Prompt to ask: {base_prompt}")

    default_config = {"configurable": {USER_ID_KEY: user_id, CONVERSATION_ID_KEY: "1"}}
    ollama_response = chain_with_message_history.invoke({"role_description": role_description, PROMPT_KEY: base_prompt}, config=default_config)

    print(f"Original Response from model: {ollama_response}")
    print(f"Tool calls: {ollama_response.tool_calls}")

    return handle_tool_calls(ollama_response.tool_calls)

ollama_instance = (ChatOllama(model=LLAMA_MODEL).bind_tools([get_weather, respond_to_user]))

chain_with_message_history = RunnableWithMessageHistory(
    prompt_template | ollama_instance,
    get_session_history=get_session_history,
    input_messages_key=PROMPT_KEY,
    history_messages_key=HISTORY_KEY,
    history_factory_config=[
        ConfigurableFieldSpec(id=USER_ID_KEY, annotation=str, is_shared=True),
        ConfigurableFieldSpec(id=CONVERSATION_ID_KEY, annotation=str, is_shared=True),
    ],
)

if __name__ == '__main__':
    if use_discord_integration:
        discord_integration.__init__()