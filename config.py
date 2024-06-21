import json
import os


class Config:
    config_file = "config.json"
    config_json = None

    def __init__(self):
        if not os.path.isfile(self.config_file):
            self.create_default_config_file()
        self.load_config()

    def create_default_config_file(self):
        dictionary = {
            "llama_url": "http://localhost:11434/api/generate",
            "llama_model": "llama3",
            "microphone_device_no": 2
        }
        with open(self.config_file, "w") as outfile:
            json.dump(dictionary, outfile)

    def load_config(self):
        with open(self.config_file, "r") as infile:
            self.config_json = json.load(infile)

    def get_config(self, property_name):
        if property_name in self.config_json:
            return self.config_json[property_name]
        else:
            raise TypeError("Property name {} not found in config.json".format(property_name))