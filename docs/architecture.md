# Architecture

## Overview

This project takes a base LLM through five stages, each one answering a
different production question:

```
Base Model
    |
    v
Quantization (GPTQ / AWQ)        "How much smaller/faster can this get
    |                             without losing meaningful quality?"
    v
vLLM Serving (continuous batching) "How do I serve many users from one GPU?"
    |
    v
Load Testing (Locust)             "Does it actually hold up under real traffic?"
    |
    v
Monitoring (Prometheus + Grafana)  "Can I see what's happening right now?"
    |
    v
Cost Analysis                      "What does this actually cost per request?"
    |
    v
Kubernetes Deployment (+ HPA)      "How does this scale automatically?"
```

Each stage produces an artifact the next stage consumes: quantized model
weights feed the server, the server's `/metrics` endpoint feeds Prometheus,
benchmark JSON feeds the cost calculator, and GPU utilization feeds the
autoscaler. Nothing here is a standalone demo — it's a pipeline.

---

## Stage 1: Quantization

Two independent methods are implemented rather than one, because they make
different tradeoffs and a real evaluation needs a comparison, not a single
number:

- **GPTQ** (`quantization/quantize_gptq.py`) — compresses weights layer by
  layer using calibration data, minimizing the change in each layer's output.
  Well-established, broad hardware support.
- **AWQ** (`quantization/quantize_awq.py`) — identifies the small fraction
  of weights that matter most based on activation magnitude and protects
  those specifically, rather than compressing everything uniformly. Often
  preserves accuracy better at the same bit-width.

`quantization/benchmark_quantization.py` runs the base model and both
quantized versions through the same evaluation: perplexity (quality),
tokens/sec (speed), and peak VRAM (memory). The output feeds directly into
the cost analysis stage — you can't say a model is "cheaper" without first
proving it's still good enough to use.

**Tradeoff being measured:** every bit of compression is a bet that the
accuracy lost is smaller than the speed/memory gained. This stage is what
turns that bet into a number instead of a guess.

---

## Stage 2: Serving

`serving/vllm_server.py` wraps vLLM's `AsyncLLMEngine` in a FastAPI app
instead of using vLLM's built-in OpenAI-compatible server. The custom
wrapper exists for two reasons:

1. **Metrics hooks** — request count, latency histogram, and tokens
   generated are tracked with `prometheus_client` directly in the request
   path, so Stage 4 (monitoring) has real data to scrape.
2. **Config-driven model swapping** — `serving/config/server_config.yaml`
   controls which model (base, GPTQ, or AWQ) is loaded, so switching what's
   being benchmarked doesn't require touching code.

**Continuous batching**, vLLM's core mechanism, is what makes this stage
matter: instead of finishing one request before starting the next, the
engine interleaves generation steps across every in-flight request. Ten
concurrent users don't mean ten times the latency — they mean the GPU stays
busy instead of idling between requests. `max_num_seqs` in the config is
the main knob controlling how many sequences batch together.

---

## Stage 3: Load Testing

A single request's latency says very little about whether a server is
production-ready. `load_testing/locustfile.py` simulates concurrent users
with realistic think-time and a mix of prompt lengths, and measures p50/p95/p99
latency rather than an average — averages hide the slow requests that
actually cause complaints.

This stage exists to answer one question honestly: **at what point does
this deployment fall over?** Running it against different `max_num_seqs`
values in the server config is how that batching knob gets tuned instead
of guessed.

---

## Stage 4: Monitoring

Two metric sources feed Prometheus (`monitoring/prometheus.yml`):

- **Application metrics** from `serving/vllm_server.py`'s `/metrics`
  endpoint — request volume, latency, tokens generated.
- **Hardware metrics** from `monitoring/exporters/gpu_exporter.py` —
  utilization, VRAM usage, temperature, power draw.

Neither one alone tells the full story. High latency with low GPU
utilization points to a batching/config problem. High latency with maxed-out
GPU utilization means the hardware is the actual bottleneck. Grafana
dashboards on top of both make that distinction visible at a glance instead
of requiring log-diving.

---

## Stage 5: Cost Analysis

`cost_analysis/cost_calculator.py` converts the raw throughput numbers from
Stage 1's benchmark into `$/1M tokens`, using a GPU hourly rate as input.
This is the translation layer between engineering metrics and a number a
non-technical stakeholder can act on — "AWQ is 2.3x faster" becomes "AWQ
cuts inference cost by roughly half on the same hardware."

---

## Stage 6: Kubernetes Deployment

- `k8s/deployment.yaml` — runs the serving container with GPU resource
  requests/limits and health probes pointed at `/health` (not `/generate`,
  so a slow inference request doesn't look like a dead pod).
- `k8s/service.yaml` — internal ClusterIP in front of the deployment.
- `k8s/configmap.yaml` — the cluster's copy of `server_config.yaml`.
- `k8s/hpa.yaml` — autoscales on **GPU utilization**, not CPU, since CPU is
  never the bottleneck for an LLM server. This requires the Prometheus
  Adapter to expose `gpu_utilization_percent` (from Stage 4's GPU exporter)
  through the Kubernetes custom metrics API — a CPU-based fallback is
  included in the same file for clusters without that set up yet.

Scale-up is fast (30s stabilization) and scale-down is slow (5min
stabilization) on purpose: GPU pods are expensive to spin back up because
of model load time, so the autoscaler is biased against flapping.

---

## Why This Order

Each stage depends on data the previous one produced:

- You can't do cost analysis (Stage 5) without throughput numbers (Stage 1)
  and a working server to measure them from (Stage 2).
- You can't tune the autoscaler (Stage 6) without knowing what "under load"
  looks like (Stage 3) and having a metric to scale on (Stage 4).

The pipeline is meant to be read top to bottom as a single story: shrink
the model, serve it efficiently, prove it holds up, watch it in production,
know what it costs, and let it scale itself.
