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
    user_id (str): The user_id provided in the System prompt.

    Returns:
    list: A list containing the result of each die roll.

    Example:
    >>> roll_dice(2, 6, "mike")
    [4, 2]
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