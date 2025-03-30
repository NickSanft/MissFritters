import re
import asyncio
import subprocess
import tempfile
import os
from agents import Agent, Runner, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from pydantic import BaseModel

# Initialize models
model = OpenAIChatCompletionsModel(
    model="llama3.2",
    openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1")
)

code_model = OpenAIChatCompletionsModel(
    model="qwen2.5-coder",
    openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1")
)


class BasicSummary(BaseModel):
    short_summary: str
    """A short 2-3 sentence summary of the findings."""

    code: str
    """The code to run. Must be standalone, runnable Python code."""


class TestSummary(BaseModel):
    test_results: str
    """Results of running the generated test cases."""


# Coding agent
coding_agent = Agent(
    model=code_model,
    name="Coding agent",
    instructions=(
        "Generate a complete, standalone Python function based on the user's request. "
        "Ensure the function is called at the end of the script so that it executes when run. "
        "Do NOT include any explanations, comments, or markdown formatting. "
        "Your response must be valid Python code that runs without modification."
    ),
    output_type=BasicSummary,
)

# Testing agent
testing_agent = Agent(
    model=code_model,
    name="Testing agent",
    instructions=(
        "Generate Python unit tests using the `unittest` module. "
        "Ensure the test cases import the correct functions and verify expected behavior. "
        "The response must be valid Python test code that runs without modification. "
        "Do NOT include explanations, comments, or markdown formatting—just the raw test code."
    ),
    output_type=TestSummary,
)

# Triage agent
triage_agent = Agent(
    model=model,
    name="Triage agent",
    instructions=(
        "Determine whether the user's request is for code generation or testing. "
        "If it's for code, route to the Coding Agent. "
        "If it's for testing, route to the Testing Agent. "
        "NEVER attempt to call tools directly. Only return structured JSON output."
    ),
    handoffs=[coding_agent, testing_agent]
)


# Function to clean extracted Python code
def clean_code(code: str) -> str:
    """Cleans generated code and ensures the function is called."""
    if not code:
        return ""

    # Remove markdown code fences
    code = re.sub(r"^```(?:python)?\n?", "", code, flags=re.MULTILINE)  # Start of block
    code = re.sub(r"\n?```$", "", code, flags=re.MULTILINE)  # End of block

    # Ensure the code is valid Python
    if not any(kw in code for kw in ["def ", "import ", "class ", "=", "print(", "return "]):
        return ""

    # Auto-append a function call if missing
    match = re.search(r"def (\w+)\(", code)
    if match and f"{match.group(1)}()" not in code:
        code += f"\n\n{match.group(1)}()"

    return code.strip()


# Function to run Python code safely in a sandbox
def run_code_safely(code: str):
    cleaned_code = clean_code(code)

    if not cleaned_code.strip():
        return "Error: No valid Python code extracted."

    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(cleaned_code.encode())
        temp_file.close()
        try:
            result = subprocess.run(
                ["python", temp_file.name],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout or result.stderr
        except Exception as e:
            return str(e)
        finally:
            os.remove(temp_file.name)


async def main():
    # Step 1: Generate code
    print("🔄 Generating Python code...")
    result = await Runner.run(triage_agent, input="Create a calculator object in Python with methods to add and subtract 2 numbers.")
    summary = result.final_output_as(BasicSummary)

    cleaned_code = clean_code(summary.code)
    if not cleaned_code:
        print("❌ Error: No valid Python code extracted.")
        return

    print(f"✅ Cleaned Code:\n{cleaned_code}")

    # Step 2: Run the cleaned code
    print("🚀 Running the generated code...")
    execution_output = run_code_safely(cleaned_code)
    print(f"🖥️ Execution Output:\n{execution_output}")

    # Step 3: Generate unit tests
    print("🔄 Generating unit tests...")
    test_result = await Runner.run(testing_agent, input=f"Write unit tests for this code:\n{cleaned_code}")
    test_summary = test_result.final_output_as(TestSummary)

    cleaned_tests = clean_code(test_summary.test_results)
    if not cleaned_tests:
        print("❌ Error: No valid test code extracted.")
        return

    print(f"✅ Cleaned Tests:\n{cleaned_tests}")

    # Step 4: Run the unit tests
    print("🧪 Running the unit tests...")
    test_execution_output = run_code_safely(cleaned_tests)
    print(f"📊 Test Results:\n{test_execution_output}")


if __name__ == "__main__":
    asyncio.run(main())
