# Runpod Docker Image

## llama.cpp
``` shell
llama cli -hf bartowski/SmolLM2-135M-Instruct-GGUF
```

## Ollama
1) First, run the server.
``` shell
ollama serve > /dev/null 2>&1 &
```

2) Run a LLM. The example below shows how to run a model from HuggingFace (bartowski/SmolLM2-135M-Instruct-GGUF)
``` shell
ollama run hf.co/bartowski/SmolLM2-135M-Instruct-GGUF
```

## vLLM
vLLM was installed using a python virtual environment. To
``` shell
uv run --with vllm vllm serve arthuravianna/Qwen2.5-0.5B-Instruct-GPTQ-4bit
```