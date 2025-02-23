import numpy as np
import pygame
import pyaudio
import wave

from message_source import MessageSource
from miss_fritters import ask_stuff
from stt import StuffHearer
from tts import StuffSayer

# Parameters
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Sampling rate

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

user_id = "local"


# Load Audio File
def load_audio(filename):
    wf = wave.open(filename, 'rb')
    return wf


# Audio Stream Setup
p = pyaudio.PyAudio()
sayer = StuffSayer()
stuff_hearer = StuffHearer()


def visualize_audio(thing_to_say):
    wf = load_audio(sayer.say_stuff_simple(thing_to_say))
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    running = True

    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        data = wf.readframes(CHUNK)
        if len(data) == 0:
            break  # Stop when audio file ends

        stream.write(data)

        # Convert audio data to waveform points
        audio_data = np.frombuffer(data, dtype=np.int16)
        points = [(x, HEIGHT // 2 + int(audio_data[x % len(audio_data)] * HEIGHT / 65536)) for x in range(WIDTH)]
        pygame.draw.lines(screen, (0, 255, 0), False, points, 2)

        pygame.display.flip()
        clock.tick(60)

    stream.stop_stream()
    stream.close()
    # Wait for user input with a still sine wave
    running = True
    while running:
        screen.fill((0, 0, 0))
        points = [(x, HEIGHT // 2 + int(50 * np.sin(2 * np.pi * x / 100))) for x in range(WIDTH)]
        pygame.draw.lines(screen, (0, 0, 255), False, points, 2)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            prompt = None
            while prompt is None:
                prompt = stuff_hearer.hear_stuff()  # Wait for a valid prompt
            response = ask_stuff(prompt, MessageSource.LOCAL, user_id)
            visualize_audio(response)

    pygame.quit()
    p.terminate()


# Run the visualizer with an audio file
if __name__ == "__main__":
    visualize_audio("Hello, my name is Miss Fritters. How can I help you today?")
