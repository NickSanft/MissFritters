from langchain_core.tools import tool

import deck_of_cards_integration
import weather_integration

from duckduckgo_search import DDGS
from datetime import datetime
import pytz

import random

@tool(parse_docstring=True)
def get_current_time():
    """
    Returns the current time as a string in RFC3339 (YYYY-MM-DDTHH:MM:SS) format.

    Example - 2025-01-13T23:11:56.337644-06:00
    """
    # Get the current time in UTC
    utc_now = datetime.now(pytz.utc)

    # Convert to CST (Central Standard Time)
    cst = pytz.timezone('US/Central')
    cst_now = utc_now.astimezone(cst)

    # Format the timestamp in RFC3339 format
    rfc3339_timestamp = cst_now.isoformat()

    print(rfc3339_timestamp)
    return rfc3339_timestamp


@tool(parse_docstring=True)
def search_web(text_to_search: str):
    """
    Takes in a string and returns results from the internet.

    Args:
    text_to_search (str): The text to search the internet for information.

    Returns:
    list: A list of dictionaries, each containing string keys and string values representing the search results.
    """
    results = DDGS().text(text_to_search, max_results=5)
    print(results)
    return results

@tool(parse_docstring=True)
def roll_dice(num_dice: int, num_sides: int, user_id: int):
    """
    Rolls a specified number of dice, each with a specified number of sides.

    Args:
    num_dice (int): The number of dice to roll.
    num_sides (int): The number of sides on each die.
    user_id (str): The user_id provided in the System prompt.

    Returns:
    list: A list containing the result of each die roll.
    """
    if num_dice <= 0 or num_sides <= 0:
        raise ValueError("Both number of dice and number of sides must be positive integers.")

    rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return (f"Here are the results: {user_id}."
            f" {rolls}")

@tool(parse_docstring=True)
def deck_cards_left(user_id: str) -> str:
    """If the user asks how many cards are left, this will return the number of cards left in their deck.

    Args:
        user_id: The user_id for the deck
    """
    return deck_of_cards_integration.get_remaining_card_number(user_id)

@tool(parse_docstring=True)
def deck_reload(user_id: str) -> str:
    """If a user asks to reload their deck, this should be called.

    Args:
        user_id: The user_id of the deck of cards to reload.
    """
    return deck_of_cards_integration.reload_deck(user_id)

@tool(parse_docstring=True)
def deck_draw_cards(number_of_cards: int, user_id: str) -> str:
    """If someone asks to draw cards, this should be called.

    Args:
        number_of_cards: The number of cards to draw.
        user_id: The user_id of the deck of cards to draw from.
    """
    return deck_of_cards_integration.draw_cards(number_of_cards, user_id)

@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """If the user asks about the weather with a specific city this should be called.
    Do not use this function if the user did not specify a city!

    Args:
        city (str): The name of the city. Must have a legitimate city name.
    """
    return weather_integration.get_weather(city)