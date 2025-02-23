ollama create -f .\llama3.2-modelfile.txt llama3.2
ollama create -f .\mistral-modelfile.txt mistral
ollama create -f .\codellama-modelfile.txt codellama
ollama create -f .\mistral-orca-modelfile.txt mistral-orca

ollama run llama3.2 /bye

ollama run mistral /bye

ollama run codellama /bye

ollama run mistral-openorca /bye

ollama run hermes3 /bye
