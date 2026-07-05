# LLM Inference Optimizer

A production-grade pipeline for making LLM inference faster and cheaper — quantization, high-throughput serving, load testing, monitoring, and cost analysis, deployed on Kubernetes with GPU autoscaling.

---

## What This Project Demonstrates

Every other project in this portfolio focuses on building an ML *system* (RAG, agents, fine-tuning).
This one focuses on making an existing model **fast, cheap, and observable in production**.

| Concern | Solution |
|---|---|
| How fast can it run? | Quantization (GPTQ / AWQ) benchmarked against the base model |
| How many users can it serve? | vLLM with continuous batching |
| Does it hold up under load? | Locust load testing — p50/p95/p99 latency, throughput |
| Can I see what's happening? | Prometheus + Grafana dashboards |
| What does it cost? | Cost-per-request analysis, base vs. optimized |
| Can it scale? | Kubernetes deployment with GPU-based autoscaling |

---

## Architecture

```
Base Model
  ↓
Quantization (GPTQ / AWQ)  →  accuracy/speed/memory benchmark
  ↓
vLLM Server  (continuous batching, FastAPI + Prometheus metrics)
  ↓
Load Testing  (Locust)  →  latency & throughput under concurrent load
  ↓
Monitoring  (Prometheus + Grafana)  →  live dashboards
  ↓
Cost Analysis  →  $/1M tokens, base vs. optimized
  ↓
Kubernetes Deployment  →  GPU-utilization-based autoscaling
```

---

## Project Structure

```
llm-inference-optimizer/
├── quantization/
│   ├── quantize_gptq.py          — GPTQ 4-bit quantization
│   ├── quantize_awq.py           — AWQ 4-bit quantization
│   └── benchmark_quantization.py — perplexity, tokens/sec, VRAM comparison
├── serving/
│   ├── vllm_server.py            — FastAPI + vLLM AsyncLLMEngine + Prometheus metrics
│   ├── config/server_config.yaml
│   ├── Dockerfile
│   └── requirements-serving.txt
├── load_testing/
│   └── locustfile.py             — concurrent load simulation, p50/p95/p99
├── monitoring/
│   ├── prometheus.yml
│   ├── exporters/
│   │   ├── gpu_exporter.py       — GPU utilization, VRAM, temp, power
│   │   ├── Dockerfile
│   │   └── requirements-exporter.txt
│   └── grafana/dashboards/
├── cost_analysis/
│   └── cost_calculator.py        — $/1M tokens, savings vs. base
├── k8s/
│   ├── deployment.yaml           — GPU resources + health probes
│   ├── service.yaml
│   ├── hpa.yaml                  — GPU-utilization-based autoscaling
│   └── configmap.yaml
├── benchmarks/results/
├── tests/
│   └── test_serving.py           — mocked-engine API tests
├── docs/
│   └── architecture.md
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Getting Started

```bash
pip install -r requirements.txt
```

### 1. Quantize a model

```bash
python quantization/quantize_gptq.py \
  --model_id meta-llama/Llama-3.2-1B \
  --output_dir ./models/llama-3.2-1b-gptq

python quantization/quantize_awq.py \
  --model_id meta-llama/Llama-3.2-1B \
  --output_dir ./models/llama-3.2-1b-awq
```

### 2. Benchmark base vs. quantized

```bash
python quantization/benchmark_quantization.py \
  --base_model meta-llama/Llama-3.2-1B \
  --gptq_model ./models/llama-3.2-1b-gptq \
  --awq_model ./models/llama-3.2-1b-awq
```

Example output *(illustrative — replace with your own run)*:
```
=== Quantization Benchmark Summary ===
Model        Perplexity   Tokens/sec   Peak VRAM (MB)
------------------------------------------------------
base         8.42         31.2         2840.5
gptq         8.61         74.8         920.3
awq          8.55         79.1         905.7
```

### 3. Serve with vLLM

Point `serving/config/server_config.yaml` at the model you want to serve (base, GPTQ, or AWQ), then:

```bash
cd serving
uvicorn vllm_server:app --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000/docs

### 4. Run the full stack (server + monitoring)

```bash
docker-compose up --build
```

- Inference API → http://localhost:8000
- Prometheus → http://localhost:9090
- Grafana → http://localhost:3000

### 5. Load test

```bash
locust -f load_testing/locustfile.py --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 2m --headless \
  --csv=benchmarks/results/load_test
```

### 6. Cost analysis

```bash
python cost_analysis/cost_calculator.py \
  --benchmark_file benchmarks/results/quantization_benchmark.json \
  --gpu_hourly_cost 2.50
```

Example output *(illustrative — replace with your own run)*:
```
=== Cost Comparison (per 1M tokens) ===
Model    $/1M tokens    Requests/$    Savings vs base
----------------------------------------------------
base     0.0222         300.5         -
gptq     0.0093         720.9         58.1%
awq      0.0088         760.3         60.4%
```

### 7. Run tests

```bash
pytest tests/ -v
```

### 8. Deploy to Kubernetes

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

---

## API Usage

### Generate a response

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantization in one sentence.", "max_tokens": 128}'
```

Response:
```json
{
  "response": "Quantization reduces the numerical precision of a model's weights to shrink its size and speed up inference, at a small cost to accuracy.",
  "model": "meta-llama/Llama-3.2-1B",
  "prompt_tokens": 7,
  "completion_tokens": 34,
  "latency_ms": 412.3
}
```

### Health check

```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model": "meta-llama/Llama-3.2-1B"}
```

---

## Stack

Python · vLLM · GPTQ (auto-gptq) · AWQ (autoawq) · FastAPI · Prometheus · Grafana · Locust · Docker · Kubernetes

---

## What I Learned

**Quantization isn't free, but it's close.**
Dropping to 4-bit with GPTQ or AWQ cuts VRAM usage substantially and improves throughput, at a perplexity cost small enough to be irrelevant for most applications. AWQ tends to edge out GPTQ slightly on quality by protecting activation-critical weights instead of compressing everything uniformly.

**Continuous batching is the real unlock, not just quantization.**
A smaller model helps, but vLLM's continuous batching is what actually lets one GPU serve many concurrent users without latency growing linearly — the two optimizations compound.

**Averages hide the problem.**
Load testing at p50 can look fine well past the point where p99 latency is already unacceptable. Percentile-based metrics, not averages, are what actually describe user experience under load.

**GPU utilization is the metric that ties everything together.**
It's the signal that connects monitoring (is the GPU busy or idle?), cost (idle GPU = wasted money), and autoscaling (scale on GPU load, not CPU, since CPU is never the bottleneck here).

**Health checks need to point at something cheap.**
Pointing Kubernetes liveness probes at `/generate` would make a slow-but-healthy inference request look like a dead pod. `/health` stays fast regardless of how busy the model is.

---

## Related Projects

- [ml-api-service](https://github.com/Honaxen/ml-api-service) — the auth/caching API layer this optimized serving layer could sit behind
- [document-agent](https://github.com/Honaxen/document-agent) — a RAG system this pipeline could accelerate

---

## Author

[Honaxen](https://github.com/Honaxen)