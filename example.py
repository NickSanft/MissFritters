from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec
from langchain_ollama import ChatOllama

llama_model = "llama3.1"

# Simple example with just user input
simple_ollama_instance = ChatOllama(model=llama_model)
simple_question_a = "Who invented the Theory of Relativity?"
print(f"Question to ask: {simple_question_a}")
simple_answer_a = simple_ollama_instance.invoke(simple_question_a)
simple_answer_a.pretty_print()

simple_question_b = "Where did he live?"
print(f"Question to ask: {simple_question_b}")
simple_answer_b = simple_ollama_instance.invoke(simple_question_b)
simple_answer_b.pretty_print()




# With user and system roles
user_system_question = "Who invested doughnuts?"
print(f"Question to ask: {user_system_question}")
user_system_ollama_instance = ChatOllama(model=llama_model)
user_system_prompt = [
    ("system", "You are a chatbot named Darryl that speaks in an American Deep South accent"),
    ("human", user_system_question),
]

user_system_answer = user_system_ollama_instance.invoke(user_system_prompt)
user_system_answer.pretty_print()




# Using a prompt template to make a chain
prompt_ollama_instance = ChatOllama(model=llama_model)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot named Darryl that speaks in an American Deep South accent"),
    ("human", "{prompt}"),
])
prompt_question = "What can you tell me about Dallas?"
print(f"Question to ask: {prompt_question}")

prompt_chain = prompt_template | prompt_ollama_instance
prompt_answer = prompt_chain.invoke(input=prompt_template.format_prompt(prompt=prompt_question))
prompt_answer.pretty_print()




# Retaining session history
PROMPT_KEY = "prompt"
HISTORY_KEY = "history"
CONVERSATION_KEY = "conversation_id"
store = {}

def get_session_history(conversation_id: str) -> BaseChatMessageHistory:
    if conversation_id not in store:
        store[conversation_id] = InMemoryChatMessageHistory()
    return store[conversation_id]



history_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{role_description}"),
    MessagesPlaceholder(variable_name=HISTORY_KEY),
    ("human", "{prompt}"),
])

history_ollama_instance = ChatOllama(model=llama_model)

chain_with_message_history = RunnableWithMessageHistory(
    history_prompt_template | history_ollama_instance,
    get_session_history=get_session_history,
    input_messages_key=PROMPT_KEY,
    history_messages_key=HISTORY_KEY,
    history_factory_config=[ConfigurableFieldSpec(id=CONVERSATION_KEY, annotation=str, is_shared=True)],
)

default_config = {"configurable": {CONVERSATION_KEY: "1"}}

history_question_a = "Who invented the Theory of Relativity?"
print(f"Question to ask: {history_question_a}")
history_answer_a = chain_with_message_history.invoke(
    {"role_description": "You are helpful chatbot.", PROMPT_KEY: history_question_a}, config=default_config)
history_answer_a.pretty_print()

history_question_b = "What else was he known for?"
print(f"Question to ask: {history_question_b}")
history_answer_b = chain_with_message_history.invoke(
    {"role_description": "You are helpful chatbot.", PROMPT_KEY: history_question_b}, config=default_config)
history_answer_b.pretty_print()

# Let's get crazy and have two LLMs talk to each other

question_config = {"configurable": {CONVERSATION_KEY: "2"}}
question_ollama = ChatOllama(model=llama_model)
question_chain = RunnableWithMessageHistory(
    history_prompt_template | question_ollama,
    get_session_history=get_session_history,
    input_messages_key=PROMPT_KEY,
    history_messages_key=HISTORY_KEY,
    history_factory_config=[ConfigurableFieldSpec(id=CONVERSATION_KEY, annotation=str, is_shared=True)],
)

answer_config = {"configurable": {CONVERSATION_KEY: "3"}}
answer_ollama = ChatOllama(model=llama_model)
answer_chain = RunnableWithMessageHistory(
    history_prompt_template | answer_ollama,
    get_session_history=get_session_history,
    input_messages_key=PROMPT_KEY,
    history_messages_key=HISTORY_KEY,
    history_factory_config=[ConfigurableFieldSpec(id=CONVERSATION_KEY, annotation=str, is_shared=True)],
)

question_role = "You are a helpful chatbot that asks questions based on the information you get. Please respond in 30 words or less."
answer_role = "You are a helpful chatbot that answers questions. Please respond in 30 words or less."

question = "What are some great places to go in Dallas? Please start with the word Dallas in the response."
print(f"Initial question to ask: {question}")

for i in range(10):
    print(f"Loop {i}\r\n")
    answer_full = chain_with_message_history.invoke(
        {"role_description": answer_role, PROMPT_KEY: question}, config=default_config)
    answer_full.pretty_print()

    question_full = chain_with_message_history.invoke(
        {"role_description": question_role, PROMPT_KEY: answer_full.content}, config=default_config)
    question_full.pretty_print()
    question = question_full.content
