# CUDA-enabled base image — vLLM needs a GPU runtime, not a plain Python image.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install serving dependencies only (not the full repo requirements.txt,
# which also includes quantization/load-testing tools this container doesn't need)
COPY requirements-serving.txt .
RUN pip3 install --no-cache-dir -r requirements-serving.txt

COPY vllm_server.py .
COPY config/ ./config/

EXPOSE 8000

# Note: --gpus all is required at `docker run` time for this image to see the GPU.
CMD ["uvicorn", "vllm_server:app", "--host", "0.0.0.0", "--port", "8000"]
