from langchain_core.tools import tool

import deck_of_cards_integration
import weather_integration

import random

@tool
def roll_dice(num_dice, num_sides, user_id):
    """
    Rolls a specified number of dice, each with a specified number of sides.

    Parameters:
    num_dice (int): The number of dice to roll.
    num_sides (int): The number of sides on each die.
    user_id (str): The id of the user.

    Returns:
    list: A list containing the result of each die roll.

    Example:
    >>> roll_dice(2, 6, "mike")
    [4, 2]
    """
    return secret_roll_dice(num_dice, num_sides, user_id)

def secret_roll_dice(num_dice, num_sides, user_id):
    if num_dice <= 0 or num_sides <= 0:
        raise ValueError("Both number of dice and number of sides must be positive integers.")

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return (f"Here are the results: {user_id}."
            f" {rolls}")

@tool(parse_docstring=True)
def deck_cards_left(user_id: str) -> str:
    """Receives a user_id and returns the number of cards left in their deck.

    Args:
        user_id: The user_id for the deck
    """
    return deck_of_cards_integration.get_remaining_card_number(user_id)

@tool(parse_docstring=True)
def deck_reload(user_id: str) -> str:
    """Receives a user_id and reloads their deck if present

    Args:
        user_id: The user_id of the deck of cards to reload.
    """
    return deck_of_cards_integration.reload_deck(user_id)

@tool(parse_docstring=True)
def deck_draw_cards(number_of_cards: int, user_id: str) -> str:
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
        case "roll_dice":
            return secret_roll_dice(args['num_dice'], args['num_sides'],  args['user_id'])
        case "get_weather":
            city = args['city']
            return get_weather(city)
        case "deck_draw_cards":
            num = args['number_of_cards']
            user = args['user_id']
            return deck_of_cards_integration.draw_cards(num, user)
        case "deck_reload":
            user = args['user_id']
            return deck_of_cards_integration.reload_deck(user)
        case "deck_cards_left":
            user = args['user_id']
            return deck_of_cards_integration.get_remaining_card_number(user)
        case _:
            return f"Unknown tool call: {tool_name} with args: {args}"