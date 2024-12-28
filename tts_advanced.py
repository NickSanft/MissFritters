import hashlib
import os
import playsound
import pygame
import torch

from TTS.api import TTS


class AdvancedStuffSayer:
    model_to_use = "tts_models/multilingual/multi-dataset/xtts_v2"
    render_device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_to_use).to(render_device)

    def say_stuff(self, message: str):
        print("Message to say: {}".format(message))
        output_file = "output/output_" + self.get_hex_hash(message) + ".wav"
        print("Hashed filename: {}".format(output_file))
        file_exists = os.path.exists(output_file)
        print("File exists: {}".format(file_exists))

        if not file_exists:
            print("Generating file...")
            self.tts.tts_to_file(
                text=message,
                file_path=output_file,
                speaker="Ana Florence",
                language="en",
                split_sentences=True
            )
            print("Generated file as {}! Saying...".format(output_file))
        else:
            print("File already exists! Saying...".format(output_file))
        print("Current working directory:", os.getcwd())
        pygame.mixer.init()  # Initialize mixer
        pygame.mixer.music.load(output_file)  # Load the sound file
        pygame.mixer.music.play()  # Play the sound
        while pygame.mixer.music.get_busy():  # Wait for the sound to finish
            pygame.time.Clock().tick(10)  # Check every 10 ms

    def get_available_speakers(self):
        for speaker in self.tts.speakers:
            print(speaker)

    def print_staff_saying_config(self):
        print(self.tts.list_models())
        print("Initializing")
        print("Initialization Complete")

    def get_hex_hash(self, string_to_hash: str):
        return hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()