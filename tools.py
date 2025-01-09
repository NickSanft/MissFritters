from langchain_core.tools import tool

import deck_of_cards_integration
import weather_integration

@tool(parse_docstring=True)
def draw_cards(number_of_cards: int, user_id: str) -> str:
    """Receives a number_of_cards and a user_id and draws a number of cards from a deck

    Args:
        number_of_cards: The number of cards to draw.
        user_id: The user_id of the deck of cards to draw from.
    """
    return deck_of_cards_integration.draw(number_of_cards, user_id)

@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """Receives a city and gets the current weather from an API.

    Args:
        city (str): The name of the city.
    """
    return weather_integration.get_weather(city)

@tool(parse_docstring=True)
def respond_to_user(content: str) -> str:
    """This tool call should always be used as default response when a specific tool is not determined.

    Args:
        content (str): The response to return to the user.
    """
    print(f"Response to user: {content}")
    return content

def handle_tool_calls(tool_calls):
    if not tool_calls:
        return "No tool calls made."

    first_tool_call = tool_calls[0]
    tool_name = first_tool_call['name']
    print(f"Tool call: {tool_name}")

    first_result_args = first_tool_call['args']
    return process_tool_call(tool_name, first_result_args)

def process_tool_call(tool_name: str, args: dict) -> str:
    match tool_name:
        case "respond_to_user":
            return respond_to_user(args['content'])
        case "get_weather":
            city = args['city']
            return get_weather(city)
        case "draw_cards":
            num = args['number_of_cards']
            user = args['user_id']
            return deck_of_cards_integration.draw(num, user)
        case _:
            return f"Unknown tool call: {tool_name} with args: {args}"