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
        "Generate a complete, standalone Python class with methods based on the user's request. "
        "The class must be named `Calculator`. Include methods `add` and `subtract` for two numbers. "
        "Ensure these methods are tested by a unit test class. "
        "Do not include any explanations or markdown formatting. "
        "The code should be self-contained and runnable without modification."
    ),
    output_type=BasicSummary,
)

# Testing agent
testing_agent = Agent(
    model=code_model,
    name="Testing agent",
    instructions=(
        "Generate Python unit tests for the `Calculator` class. The class should include methods for adding and subtracting two numbers. "
        "Use the `unittest` module and include tests for: "
        "- adding positive numbers, "
        "- adding negative numbers, "
        "- adding zero, "
        "- subtracting positive numbers, "
        "- subtracting negative numbers, "
        "- subtracting zero. "
        "Ensure that the tests use assertions such as `assertEqual` to verify that the output is correct. "
        "Below is an example of valid unit test cases:\n"
        "```python\n"
        "import unittest\n\n"
        "class TestCalculator(unittest.TestCase):\n"
        "    def test_add_positive_numbers(self):\n"
        "        calc = Calculator()\n"
        "        self.assertEqual(calc.add(2, 3), 5)\n\n"
        "    def test_subtract_positive_numbers(self):\n"
        "        calc = Calculator()\n"
        "        self.assertEqual(calc.subtract(5, 3), 2)\n"
        "```\n"
        "Do NOT include any explanations or markdown formatting."
    ),
    output_type=TestSummary,
)

# Triage agent
triage_agent = Agent(
    model=model,
    name="Triage agent",
    instructions=(
        "You are responsible for routing requests to the correct agent. "
        "If the user asks for code, pass the request to the Coding agent. "
        "If the user asks for unit tests, pass the request to the Testing agent."
        "Do NOT attempt to call tools directly. Only return structured JSON output."
    ),
    handoffs=[coding_agent, testing_agent],
    tools=[]  # Ensure there are no tool references
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

    # Auto-append a function call if missing in non-test code
    if "def " in code and "unittest" not in code:  # Don't add function calls for test code
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
