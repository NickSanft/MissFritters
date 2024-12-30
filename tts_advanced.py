import hashlib
import os
import pygame
import torch
from TTS.api import TTS


def _play_audio(output_file: str):
    """
    Play the generated or existing audio file using pygame.
    """
    print(f"Current working directory: {os.getcwd()}")
    pygame.mixer.init()  # Initialize pygame mixer
    pygame.mixer.music.load(output_file)  # Load the sound file
    pygame.mixer.music.play()  # Play the sound

    while pygame.mixer.music.get_busy():  # Wait for the sound to finish
        pygame.time.Clock().tick(10)  # Check every 10 ms


def _file_exists(file_path: str) -> bool:
    """
    Check if the file exists at the given path.
    """
    return os.path.exists(file_path)


def get_hex_hash(string_to_hash: str) -> str:
    """
    Generate a hexadecimal hash of the input string.
    """
    return hashlib.md5(string_to_hash.encode('utf-8')).hexdigest()


def get_output_file(message: str) -> str:
    """
    Generate a hashed filename for the message's audio file.
    """
    file_hash = get_hex_hash(message)
    return f"output/output_{file_hash}.wav"


class AdvancedStuffSayer:
    # Constants for model and render device
    MODEL_TO_USE = "tts_models/multilingual/multi-dataset/xtts_v2"
    RENDER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize TTS model on the appropriate device
    tts = TTS(MODEL_TO_USE).to(RENDER_DEVICE)

    def say_stuff(self, message: str):
        """
        Generate and play speech from the given message.
        First checks if the audio file already exists. If not, it generates the file.
        Then, it plays the audio using pygame.
        """
        print(f"Message to say: {message}")
        output_file = get_output_file(message)

        if not _file_exists(output_file):
            print(f"Generating file: {output_file}")
            self._generate_audio_file(message, output_file)
        else:
            print(f"File already exists! Playing: {output_file}")

        _play_audio(output_file)

    def get_available_speakers(self):
        """
        Print available speakers for the TTS model.
        """
        for speaker in self.tts.speakers:
            print(speaker)

    def print_staff_saying_config(self):
        """
        Print the TTS model list and initialization status.
        """
        print(self.tts.list_models())
        print("Initializing")
        print("Initialization Complete")

    def _generate_audio_file(self, message: str, output_file: str):
        """
        Generate an audio file from the message and save it to the specified output file.
        """
        self.tts.tts_to_file(
            text=message,
            file_path=output_file,
            speaker="Ana Florence",
            language="en",
            split_sentences=True
        )

