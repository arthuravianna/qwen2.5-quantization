FROM ghcr.io/ggml-org/llama.cpp:light-cuda13 AS llama_cpp_binaries


# Use Runpod base image
FROM runpod/base:1.0.2-cuda1300-ubuntu2404

# Set environment variables
# This ensures Python output is immediately visible in logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*


# Install llama.cpp
COPY --from=llama_cpp_binaries /app/llama /bin
COPY --from=llama_cpp_binaries /app/llama-cli /bin
COPY --from=llama_cpp_binaries /app/llama-completion /bin
COPY --from=llama_cpp_binaries /app/libggml*.so* /bin
COPY --from=llama_cpp_binaries /app/*.so /usr/local/lib
COPY --from=llama_cpp_binaries /app/*.so.* /usr/local/lib
RUN ldconfig

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install vllm
RUN uv venv --python 3.12 --seed
RUN uv pip install --python /app/.venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio
ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}
RUN uv pip install --python /app/.venv/bin/python vllm
# RUN uv pip install --python /app/.venv/bin/python vllm --torch-backend=auto
# RUN source .venv/bin/activate
# RUN uv pip install vllm --torch-backend=auto
CMD ["/start.sh"]