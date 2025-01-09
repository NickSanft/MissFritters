import random
import re

regularSuits = ["Clubs", "Diamonds", "Hearts", "Spades"]
ranks = ["Ace", "2", "3", "4", "5", "6", "7",
         "8", "9", "10", "Jack", "Queen", "King"]
successRanks = ['Ace', 'Jack', 'Queen', 'King']
decksByUser = {}

notNumbersMessages = ["You really think numbers are letters, huh?",
                      "You should really come with a warning label.",
                      "You’re not stupid! You just have bad luck when you’re thinking."]
failureMessages = ["It couldn't be too bad, could it?",
                   "Maybe you'll just slip on a banana peel."]
criticalfailureMessages = ["Pray to your gods.",
                           "Aw man, am I gonna die?", "This is funny in a cosmic sort of way."]

def draw(numCards: int, user_id: str) -> str:
    response_messages = []

    if user_id not in decksByUser:
        decksByUser[user_id] = Deck()
    deck = decksByUser[user_id]

    numSuccesses = 0
    numFailures = 0
    queenOfHearts = False

    response_messages.append("Drawing " + str(numCards) + " for " + user_id + "...")

    for i in range(0, numCards):
        card = deck.drawCard()
        if card.isSuccess():
            numSuccesses += 1
        if card.isFailure():
            numFailures += 1
        if card.isQueenOfHearts():
            queenOfHearts = True

        response_messages.append(user_id + " drew: " + card.description + ". Cards left: " + str(len(deck.cards)))

        if len(deck.cards) == 0:
            response_messages.append("Out of cards! getting a new deck...")
            decksByUser[user_id] = Deck()
            deck = decksByUser[user_id]

    response_messages.append("```Total number of Successes: " + str(numSuccesses) + "\n")

    if queenOfHearts:
        response_messages.append("Queen Of Hearts! Add your charm to the number of successes! \n")

    if numFailures == 1:
        response_messages.append("1 failure. " + random.choice(failureMessages) + "\n")
    elif numFailures == 2:
        response_messages.append("2 failures. " + random.choice(criticalfailureMessages) + "\n")
    else:
        response_messages.append("No failures, phew. \n")

    response_messages.append("```")
    return "\r\n".join(response_messages)

class Deck:

    def __init__(self):
        self.cards = [Card(rank, suit, rank + " of " + suit)
                      for rank in ranks for suit in regularSuits]
        self.cards.append(Card("Joker", "Red", "Red Joker"))
        self.cards.append(Card("Joker", "Black", "Black Joker"))

        random.shuffle(self.cards)

    def drawCard(self):
        return self.cards.pop()


class Card:
    def __init__(self, rank, suit, description):
        self.suit = suit
        self.rank = rank
        self.description = description

    def isSuccess(self):

        if any(re.findall('|'.join(successRanks), self.rank)):
            return True
        return False

    def isFailure(self):
        return self.rank == "Joker"

    def isQueenOfHearts(self):
        return self.suit == "Hearts" and self.rank == "Queen"