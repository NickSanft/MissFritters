import asyncio
from kasa import Discover, Module
from langchain_core.tools import tool

from fritters_utils import get_key_from_json_config_file

@tool(parse_docstring=True)
def turn_off_lights():
    """
    Turns off the lights.
    """
    asyncio.run(turn_off_lights_internal())

@tool(parse_docstring=True)
def turn_on_lights():
    """
    Turns on the lights.
    """
    asyncio.run(turn_on_lights_internal())

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
