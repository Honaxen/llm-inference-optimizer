# Lightweight container for the GPU metrics exporter.
#
# Needs the NVIDIA driver visible from inside the container (via
# --gpus all / the NVIDIA Container Toolkit) for pynvml to see the GPU —
# no CUDA toolkit itself is required, just driver access.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-exporter.txt .
RUN pip install --no-cache-dir -r requirements-exporter.txt

COPY gpu_exporter.py .

EXPOSE 9400

CMD ["python", "gpu_exporter.py", "--port", "9400", "--interval", "2"]
