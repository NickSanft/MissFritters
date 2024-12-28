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
            "llama_model": "llama3.1"
        }
        with open(self.config_file, "w") as outfile:
            json.dump(dictionary, outfile)

    def load_config(self):
        with open(self.config_file, "r") as infile:
            self.config_json = json.load(infile)

    def has_config(self, property_name):
        return property_name in self.config_json

    def add_config(self, property_name, property_value):
        config_to_add = {property_name: property_value}
        print("Adding config: {}".format(config_to_add))
        self.config_json.update(config_to_add)
        with open(self.config_file, "w") as file:
            json.dump(self.config_json, file)

    def get_config(self, property_name):
        if self.has_config(property_name):
            return self.config_json[property_name]
        else:
            raise TypeError("Property name {} not found in {}".format(property_name, self.config_file))
