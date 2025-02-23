import asyncio
from kasa import Discover
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

import fritters_utils
from fritters_utils import get_key_from_json_config_file, ROOT_USER_ID_KEY

BAD_USER_MESSAGE = "This person tried to mess with someone's lights and was denied access! Please be mean to them."

@tool(parse_docstring=True)
def turn_off_lights(config: RunnableConfig):
    """
    Turns off the lights.

    Args:
        config: The RunnableConfig.
    """
    user_id = config.get("metadata").get("user_id")
    root_user_id = fritters_utils.get_key_from_json_config_file(ROOT_USER_ID_KEY)
    if root_user_id is None or user_id != root_user_id:
        print(BAD_USER_MESSAGE)
        return BAD_USER_MESSAGE
    print("Turning off lights...")
    asyncio.run(turn_off_lights_internal())
    return "The lights have been turned off."

@tool(parse_docstring=True)
def turn_on_lights(config: RunnableConfig):
    """
    Turns on the lights.

    Args:
        config: The RunnableConfig.
    """
    user_id = config.get("metadata").get("user_id")
    root_user_id = fritters_utils.get_key_from_json_config_file(ROOT_USER_ID_KEY)
    if root_user_id is None or user_id != root_user_id:
        print(BAD_USER_MESSAGE)
        return BAD_USER_MESSAGE
    print("Turning on lights...")
    asyncio.run(turn_on_lights_internal())
    return "The lights have been turned on."

async def turn_off_lights_internal():
    found_devices = await Discover.discover(username=get_key_from_json_config_file("kasa_username"),
                                            password=get_key_from_json_config_file("kasa_password"))
    for device in found_devices.values():
        print(device.alias)
        print(device.host)
        print(device.device_type)
        print(device.children)
        print(device.features)
        print(device.modules)
        await device.turn_off()
        await device.update()

async def turn_on_lights_internal():
    found_devices = await Discover.discover(username=get_key_from_json_config_file("kasa_username"),
                                            password=get_key_from_json_config_file("kasa_password"))
    for device in found_devices.values():
        print(device.alias)
        print(device.host)
        print(device.device_type)
        print(device.children)
        print(device.features)
        print(device.modules)
        await device.turn_on()
        await device.update()

async def get_device_info():
    found_devices = await Discover.discover(username=get_key_from_json_config_file("kasa_username"),
                                            password=get_key_from_json_config_file("kasa_password"))
    for device in found_devices.values():
        print(device.alias)
        print(device.host)
        print(device.device_type)
        print(device.children)
        print(device.features)
        print(device.modules)

asyncio.run(get_device_info())