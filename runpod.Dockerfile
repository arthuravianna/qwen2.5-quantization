# Use Runpod PyTorch base image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

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

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Qwen/Qwen2.5 14B-Instruct
RUN hf download Qwen/Qwen2.5-14B-Instruct --local-dir /app/llm

# Copy application files
COPY *.py .
COPY *.sh .

CMD ["/start.sh"]