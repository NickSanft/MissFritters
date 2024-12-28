# Miss Fritters

Miss Fritters is currently a test of Llama3 using Ollama and Coqui TTS.

## Installation

Download Ollama and run llama3 locally first - https://ollama.com/blog/llama3

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install pyttsx3, requests, ollama, playsound, pyaudio, langchain_community, and SpeechRecognition

If you want to use the Simple TTS, just use the SimpleStuffSayer in main.py.

If you want to use Advanced TTS instead of the simple, it requires a bit more setup...

- For Windows, it also requires build tools. Go [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/), Check Desktop development with C++, then install
- Use pip to install coqui-tts

## Use

By default, running will go into an endless loop of listening for a prompt for Llama3 after starting.