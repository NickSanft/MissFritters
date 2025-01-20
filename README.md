# Miss Fritters

Miss Fritters is currently a test of LLMs using Ollama and Coqui TTS.

## Installation

- Download Ollama - https://ollama.com/blog/llama3
- Run the LLMs you want using Ollama. for all the models I use, run ./modelfiles/run.bat.
- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install any dependencies using the requirements.txt file (pip install -r requirements.txt)
- If you want to use the Simple TTS, just use the SimpleStuffSayer in main.py.
- If you want to use Advanced TTS instead of the simple, it requires a bit more setup...
  - For Windows, it also requires build tools. Go [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/), Check Desktop development with C++, then install

## Use

There are three uses at the moment:

- main_discord: Uses a Discord app, only requires a config.json file in the root directory with a valid token in a discord_bot_token key.
  - Has \$join, \$ask, and \$leave commands to have it join a Discord call and use TTS.
  - Images work, but at the moment you will need to indicate to the bot which file.
- main_cli: Your standard command-line in a loop.
- main_stt: An endless loop of listening for user input via voice and responding.