import ollama
import langgraph
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langgraph.graph import StateGraph
from typing import TypedDict, Optional

# Load the Ollama model
llm = Ollama(model="codellama")  # You can replace with "mistral", "codellama", etc.

# Define the state schema
class AgentState(TypedDict):
    query: str
    code: Optional[str]
    output: Optional[str]
    error: Optional[str]

# Define memory to track code execution states
memory = ConversationBufferMemory()

# Create a prompt template for AI coding
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="Write a Python function that {query}. Ensure it is well-documented and optimized."
)

# Function to generate code
def generate_code(state: AgentState) -> AgentState:
    query = state["query"]
    response = llm.invoke(prompt_template.format(query=query))  # Invoke LLM
    return {"query": query, "code": response, "output": None, "error": None}

# Function to execute and validate code
def execute_code(state: AgentState) -> AgentState:
    code = state["code"]
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return {"query": state["query"], "code": code, "output": "Execution successful", "error": None}
    except Exception as e:
        return {"query": state["query"], "code": code, "output": None, "error": str(e)}

# Set up LangGraph state transitions
workflow = StateGraph(AgentState)  # Define the state schema

workflow.add_node("generate_code", generate_code)
workflow.add_node("execute_code", execute_code)
workflow.add_edge("generate_code", "execute_code")
workflow.set_entry_point("generate_code")

graph = workflow.compile()

# Run the agent
result = graph.invoke({"query": "sort a list of numbers using quicksort"})
print(result)
