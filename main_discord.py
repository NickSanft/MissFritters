import json

import discord
from discord.ext import commands

from main_stt import stuff_sayer
from miss_fritters import ask_stuff
from message_source import MessageSource

command_prefix = "$"
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix=command_prefix, intents=intents)

connection = None

def get_key_from_json_file(file_path: str, key_name: str) -> str | None:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data.get(key_name)  # Get the key value by key name
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return None

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.command()
async def hello(ctx):
    await ctx.send("Hello!")

@client.command()
async def join(ctx):
    try:
        channel = ctx.author.voice.channel
        await channel.connect()
    except AttributeError as e:
        await ctx.send("You are not connected to a voice channel, buddy!")

# @client.command()
# async def play(ctx):
#     try:
#         ctx.voice_client.play(discord.FFmpegPCMAudio(source="./something.mp3"))
#     except AttributeError as e:
#         await ctx.send("You are not connected to a voice channel, buddy!")

@client.command()
async def ask(ctx, *, message):
    author = ctx.author.name
    original_response = ask_stuff(message, MessageSource.DISCORD_VOICE, author)
    output_file = stuff_sayer.say_stuff(original_response)
    await ctx.voice_client.play(discord.FFmpegPCMAudio(source=output_file))

@client.command()
async def leave(ctx):
    try:
        await ctx.voice_client.disconnect()
    except AttributeError as e:
        await ctx.send("I am not connected to a voice channel, buddy!")


@client.event
async def on_message(message):
    author = message.author.name
    channel_type = message.channel
    if message.author == client.user:
        print("That's me, not responding :)")
        return
    elif message.content.startswith(command_prefix):
        await client.process_commands(message)
        return
    elif not isinstance(channel_type, discord.DMChannel) and not client.user.mentioned_in(message):
        print("Not a DM or mention, not responding :)")
        return

    if message.attachments:
        print("Attachment found!")
        split_v1 = str(message.attachments).split("filename='")[1]
        filename = str(split_v1).split("' ")[0]
        filepath = "./input/{}".format(filename)
        if filename.endswith(".jpg"):
            await message.attachments[0].save(fp=filepath)  # saves the file
        print("File saved!")
    else:
        print("There is no attachment")


    print("Incoming message: {} \r\n from: {}".format(message.clean_content, author))

    original_response = ask_stuff(message.clean_content, MessageSource.DISCORD_TEXT, author)
    print("Final response: {}".format(original_response))

    if not original_response:
        original_response = "The bot got sad and doesn't want to talk to you at the moment :("


    resp_len = len(original_response)
    if resp_len > 2000:
        response = "The answer was over 2000 ({}), so you're getting multiple messages {} \r\n".format(resp_len, author) + original_response
        responses = split_into_chunks(response)
        for i, response in enumerate(responses):
            await message.channel.send(response)
    else:
        await message.channel.send(original_response)


def split_into_chunks(s, chunk_size=2000):
    return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]

if __name__ == '__main__':
    discord_secret = get_key_from_json_file("config.json", "discord_bot_token")
    client.run(discord_secret)