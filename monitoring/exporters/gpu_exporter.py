"""
Custom Prometheus exporter for GPU metrics.

vLLM's built-in metrics tell you about requests and tokens, but not what the
GPU itself is doing. This exporter fills that gap: utilization, VRAM usage,
temperature, and power draw — the numbers that explain *why* throughput
looks the way it does under load, and that feed into cost_analysis/cost_calculator.py
(idle GPU = wasted money, maxed-out GPU = the batching config is working).

Usage:
    python gpu_exporter.py --port 9400 --interval 2

Requires an NVIDIA GPU and the `nvidia-ml-py` package (import name: pynvml).
"""

import argparse
import time

import pynvml
from prometheus_client import start_http_server, Gauge


GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent", "GPU compute utilization", ["gpu_index"]
)
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_mb", "GPU memory currently in use", ["gpu_index"]
)
GPU_MEMORY_TOTAL = Gauge(
    "gpu_memory_total_mb", "Total GPU memory available", ["gpu_index"]
)
GPU_TEMPERATURE = Gauge(
    "gpu_temperature_celsius", "GPU temperature", ["gpu_index"]
)
GPU_POWER_WATTS = Gauge(
    "gpu_power_draw_watts", "GPU power draw", ["gpu_index"]
)


def collect_metrics(handles):
    for index, handle in enumerate(handles):
        gpu_index = str(index)

        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        GPU_UTILIZATION.labels(gpu_index=gpu_index).set(utilization.gpu)

        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        GPU_MEMORY_USED.labels(gpu_index=gpu_index).set(memory.used / (1024 ** 2))
        GPU_MEMORY_TOTAL.labels(gpu_index=gpu_index).set(memory.total / (1024 ** 2))

        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        GPU_TEMPERATURE.labels(gpu_index=gpu_index).set(temperature)

        # Power draw is reported in milliwatts; convert to watts for readability.
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        GPU_POWER_WATTS.labels(gpu_index=gpu_index).set(power_mw / 1000)


def main(port: int, interval: float):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]

    print(f"Found {device_count} GPU(s)")
    print(f"Exposing metrics on :{port}/metrics every {interval}s")

    start_http_server(port)

    try:
        while True:
            collect_metrics(handles)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus GPU metrics exporter")
    parser.add_argument("--port", type=int, default=9400)
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between scrapes")
    args = parser.parse_args()

    main(args.port, args.interval)
