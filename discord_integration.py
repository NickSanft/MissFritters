import discord

from main import ask_stuff
import fritters_constants
from message_source import MessageSource

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def __init__(config):
    client.run(config.get_config(fritters_constants.DISCORD_KEY))

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_message(message):
    author = message.author.name
    channel_type = message.channel
    if message.author == client.user:
        print("That's me, not responding :)")
        return
    elif not isinstance(channel_type, discord.DMChannel) and not client.user.mentioned_in(message):
        print("Not a DM or mention, not responding :)")
        return

    print("Incoming message: {} \r\n from: {}".format(message.content, author))

    if message.content.lower().startswith('hello'):
        original_response = "Hello, {}!".format(author)
    else:
        original_response = ask_stuff(message.content, MessageSource.DISCORD, author)
    print("Final response: {}".format(original_response))

    if len(original_response) > 2000:
        response = "The answer was too long, so you're getting multiple messages {} \r\n".format(author)
        responses = split_into_chunks(response)
        for i, response in enumerate(responses):
            await message.channel.send(response)
    else:
        await message.channel.send(original_response)

    #output_file = stuff_sayer.say_stuff(original_response)
    #await message.channel.send(file=discord.File(output_file))


def split_into_chunks(s, chunk_size=2000):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]