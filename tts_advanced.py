import os
import playsound
import time
import torch

from TTS.api import TTS


class AdvancedStuffSayer:
    model_to_use = "tts_models/multilingual/multi-dataset/xtts_v2"
    render_device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_to_use).to(render_device)

    def say_stuff(self, message: str):
        print("Message to say: {}".format(message))
        output_file = "output_" + str(round(time.time() * 1000)) + ".wav"
        print("Generating file...")
        self.tts.tts_to_file(
            text=message,
            file_path=output_file,
            speaker="Ana Florence",
            language="en",
            split_sentences=True
        )
        print("Generated file as {}! Saying...".format(output_file))
        playsound.playsound(output_file)
        if os.path.exists(output_file):
            os.remove(output_file)

    def get_available_speakers(self):
        for speaker in self.tts.speakers:
            print(speaker)

    def print_staff_saying_config(self):
        print(self.tts.list_models())
        print("Initializing")
        print("Initialization Complete")
