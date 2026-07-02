"""
FastAPI wrapper around vLLM's AsyncLLMEngine.

Why a custom wrapper instead of vLLM's built-in OpenAI-compatible server:
  - Full control over request/response shape for cost_analysis and monitoring
  - A place to hook in Prometheus metrics (latency, tokens/sec, queue depth)
  - Easier to reason about for the load-testing and k8s parts of this project

Continuous batching, in one sentence:
  Instead of processing one request fully before starting the next,
  vLLM interleaves tokens from many in-flight requests on the GPU each step.
  That's what turns "N requests = N x latency" into "N requests share the same pass".

Usage:
    uvicorn vllm_server:app --host 0.0.0.0 --port 8000
"""

import time
import uuid
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


CONFIG_PATH = "config/server_config.yaml"

REQUEST_COUNT = Counter(
    "inference_requests_total", "Total number of /generate requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds", "End-to-end latency of /generate requests"
)
TOKENS_GENERATED = Counter(
    "inference_tokens_generated_total", "Total number of tokens generated"
)

engine: AsyncLLMEngine | None = None
server_config: dict = {}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, server_config
    server_config = load_config(CONFIG_PATH)

    engine_args = AsyncEngineArgs(
        model=server_config["model"],
        max_num_seqs=server_config.get("max_num_seqs", 32),
        gpu_memory_utilization=server_config.get("gpu_memory_utilization", 0.9),
        quantization=server_config.get("quantization"),  # e.g. "gptq", "awq", or None
        dtype=server_config.get("dtype", "auto"),
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print(f"vLLM engine ready — model: {server_config['model']}")
    yield
    print("Shutting down vLLM engine")


app = FastAPI(title="LLM Inference Optimizer — Serving", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


class GenerateResponse(BaseModel):
    response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    sampling_params = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        final_output = None
        async for output in engine.generate(req.prompt, sampling_params, request_id):
            final_output = output  # last yielded output holds the full completion

        elapsed_ms = (time.perf_counter() - start) * 1000

        completion = final_output.outputs[0]
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(completion.token_ids)

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(elapsed_ms / 1000)
        TOKENS_GENERATED.inc(completion_tokens)

        return GenerateResponse(
            response=completion.text,
            model=server_config["model"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok" if engine is not None else "starting",
        "model": server_config.get("model"),
    }


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
