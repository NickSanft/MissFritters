from agents import Agent, Runner, OpenAIChatCompletionsModel
import asyncio

from langchain_experimental.utilities import PythonREPL
from openai import AsyncOpenAI
from pydantic import BaseModel

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
    """Optional, the code to run. Must be standalone, runnable code."""


coding_agent = Agent(
    model=code_model,
    name="Coding agent",
    instructions="You help with coding.",
    output_type=BasicSummary,
)

conversation_agent = Agent(
    model=model,
    name="Conversation agent",
    instructions="You only speak English",
    output_type=BasicSummary,
)

triage_agent = Agent(
    model=model,
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the request. The coding agent is for coding and the conversation agent is for all conversation.",
    handoffs=[coding_agent, conversation_agent],
)


async def main():

    result = await Runner.run(triage_agent, input="Can you provide Python code that prints out hello?")
    print(result.final_output)
    summary = result.final_output_as(BasicSummary)
    print(f"Short summary: {summary.short_summary}")
    if summary.code:
        print(f"Python code: {summary.code}")
        python_repl = PythonREPL()
        code_result = python_repl.run(python_repl.sanitize_input(summary.code))
        print(f"Code result: {code_result}")


if __name__ == "__main__":
    asyncio.run(main())
