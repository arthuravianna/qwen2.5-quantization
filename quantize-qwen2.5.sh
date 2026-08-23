#! /bin/bash

MODEL_PATH="$1"
GPTQ_PATH="${MODEL_PATH}-4bit-GPTQ"
GGUF_PATH="${MODEL_PATH}-4bit-GGUF"
AWQ_PATH="${MODEL_PATH}-4bit-AWQ"
EXL3_PATH="${MODEL_PATH}-4bit-EXL3"

echo "Quantizing $MODEL_PATH"
if [ ! -d "$GPTQ_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$GPTQ_PATH" gptq
else
    echo "Skipping GPTQ: $GPTQ_PATH already exists"
fi

if [ ! -d "$GGUF_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$GGUF_PATH" gguf
else
    echo "Skipping GGUF: $GGUF_PATH already exists"
fi

if [ ! -d "$AWQ_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$AWQ_PATH" awq
else
    echo "Skipping AWQ: $AWQ_PATH already exists"
fi

if [ ! -d "$EXL3_PATH" ]; then
    python3 quantize.py "$MODEL_PATH" "$EXL3_PATH" exl3
else
    echo "Skipping EXL3: $EXL3_PATH already exists"
fi