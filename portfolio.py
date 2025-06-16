from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from typing import TypedDict, List
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_experimental.utilities import PythonREPL
import zipfile
import os
import json

BASE_PATH = "./output"

# ---- 1. Define State Schema ----
class PortfolioState(TypedDict):
    user_info: dict
    html: str
    css: str
    js: str
    final_package: str

# ---- 2. Initialize LLM and ReAct Agent ----
llm = ChatOllama(model="llama3.2")

@tool(parse_docstring=True)
def run_python(code: str) -> str:
    """
    Runs a string of Python code.

    Args:
        code (str): The Python code to execute.
    """
    repl = PythonREPL()
    return repl.run(code)

agent = create_react_agent(llm, tools=[run_python])

# ---- 3. Define Nodes with LLM Calls ----

def collect_user_info(state: PortfolioState) -> PortfolioState:
    prompt = """
    I want to create a portfolio website. Please respond with a JSON object that includes:
    - Name
    - Short bio
    - Skills (comma-separated)
    - Projects (a list of objects with title and description)
    - Contact email
    - Theme (dark or light)
    Do not explain anything else, just return the JSON.
    """

    # Direct call to llm for clean JSON output
    response = llm.invoke(prompt)

    content = None
    if isinstance(response, dict) and "message" in response:
        content = response["message"].get("content", None)
    elif hasattr(response, "content"):
        content = response.content
    elif isinstance(response, str):
        content = response

    if not content or content.strip() == "":
        raise ValueError(f"Empty content received from LLM.\nFull response: {response}")

    try:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        json_str = content[json_start:json_end]
        parsed = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from LLM output content:\n{content}\nError: {e}")

    print("User info parsed successfully:", parsed)

    return {"user_info": parsed, "html": "", "css": "", "js": "", "final_package": ""}


def generate_html(state: PortfolioState) -> PortfolioState:
    prompt = f"""
    Generate a complete HTML file for a portfolio website using Bootstrap 5 via CDN.
    Use the following user info to build the content:
    {json.dumps(state['user_info'], indent=2)}

    The HTML should:
    - Include Bootstrap CSS and JS CDN links in the <head> and before </body>
    - Use Bootstrap classes for layout, typography, buttons, navbar, cards, etc.
    - Link to an external script.js file for custom JavaScript.
    Return only the raw HTML content.
    """
    response = llm.invoke(prompt)
    content = response.content
    print(f"Generated HTML length: {len(content)}")
    if not content.strip():
        print("Warning: Generated HTML is empty!")
    state["html"] = content
    return state


def generate_css(state: PortfolioState) -> PortfolioState:
    # With Bootstrap, minimal or no CSS is needed.
    prompt = f"""
    Generate any minimal custom CSS needed for a portfolio website using the {state['user_info']['theme']} theme,
    but primarily rely on Bootstrap 5 for styling.
    Return only the raw CSS content (can be empty if not needed).
    """
    response = llm.invoke(prompt)
    content = response.content
    print(f"Generated CSS length: {len(content)}")
    state["css"] = content
    return state

def generate_js(state: PortfolioState) -> PortfolioState:
    prompt = """
    Generate a simple JavaScript file for a portfolio website.
    Include code that logs a message on page load and adds smooth scrolling behavior.
    Return only the raw JavaScript content.
    """
    response = llm.invoke(prompt)
    content = response.content
    print(f"Generated JS length: {len(content)}")
    if not content.strip():
        print("Warning: Generated JS is empty!")
    state["js"] = content
    return state



def package_site(state: PortfolioState) -> PortfolioState:
    os.makedirs(f"{BASE_PATH}/portfolio", exist_ok=True)

    index_path = f"{BASE_PATH}/portfolio/index.html"
    css_path = f"{BASE_PATH}/portfolio/style.css"
    js_path = f"{BASE_PATH}/portfolio/script.js"

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(state["html"])
    with open(css_path, "w", encoding="utf-8") as f:
        f.write(state["css"])
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(state["js"])

    zip_path = f"{BASE_PATH}/portfolio_site.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(index_path, "index.html")
        zipf.write(css_path, "style.css")
        zipf.write(js_path, "script.js")

    state["final_package"] = zip_path

    print(f"Portfolio website packaged at: {zip_path}")

    return state

# ---- 4. Build the Graph ----
builder = StateGraph(PortfolioState)

builder.add_node("collect_user_info", collect_user_info)
builder.add_node("generate_html", generate_html)
builder.add_node("generate_css", generate_css)
builder.add_node("generate_js", generate_js)
builder.add_node("package_site", package_site)

builder.set_entry_point("collect_user_info")
builder.add_edge("collect_user_info", "generate_html")
builder.add_edge("generate_html", "generate_css")
builder.add_edge("generate_css", "generate_js")
builder.add_edge("generate_js", "package_site")
builder.add_edge("package_site", END)

graph = builder.compile()

result = graph.invoke({})

print("Final zip package path:", result["final_package"])
