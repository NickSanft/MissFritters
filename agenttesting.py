from agents import Agent, Runner, OpenAIChatCompletionsModel
import asyncio

from openai import AsyncOpenAI

model = OpenAIChatCompletionsModel(
    model="llama3.2",
    openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1")
)

code_model = OpenAIChatCompletionsModel(
    model="qwen2.5-coder",
    openai_client=AsyncOpenAI(base_url="http://localhost:11434/v1")
)

coding_agent = Agent(
    model=code_model,
    name="Coding agent",
    instructions="You help with coding.",
)

conversation_agent = Agent(
    model=model,
    name="Conversation agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    model=model,
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the request. The coding agent is for coding and the conversation agent is for all conversation.",
    handoffs=[coding_agent, conversation_agent],
)


async def main():
    result = await Runner.run(triage_agent,
                              input="Can you give me a Python function to print the numbers 1 through 10?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
