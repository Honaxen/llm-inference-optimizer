"""
Benchmark base vs GPTQ vs AWQ quantized models on three axes:
  - Quality:    perplexity on a held-out text sample
  - Memory:     peak VRAM usage during inference
  - Throughput: generated tokens per second

Usage:
    python benchmark_quantization.py \
        --base_model meta-llama/Llama-3.2-1B \
        --gptq_model ./models/llama-3.2-1b-gptq \
        --awq_model ./models/llama-3.2-1b-awq \
        --output ../benchmarks/results/quantization_benchmark.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_eval_texts(n_samples: int = 20, seq_len: int = 512):
    """Load a fixed slice of held-out text for perplexity scoring."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = []
    for example in dataset:
        text = example["text"].strip()
        if len(text.split()) >= 50:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    return texts


def compute_perplexity(model, tokenizer, texts, device, max_length=512):
    """
    Lower perplexity = the model is less "surprised" by real text = better quality.
    This is the standard way to check how much quality quantization cost you.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings["input_ids"].to(device)

            outputs = model(input_ids, labels=input_ids)
            # loss is already averaged per-token; weight by token count to combine samples correctly
            n_tokens = input_ids.shape[1]
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def measure_throughput(model, tokenizer, device, prompt="The future of AI is", max_new_tokens=100):
    """Tokens/sec for a single generation run. Rough but consistent across models."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_sec = generated_tokens / elapsed
    return tokens_per_sec


def measure_peak_memory_mb(device):
    """Peak VRAM allocated since the last reset, in MB. 0 on CPU-only runs."""
    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def benchmark_model(label: str, model_path: str, device: str, eval_texts):
    print(f"\n--- Benchmarking: {label} ({model_path}) ---")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
    )

    print("Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer, eval_texts, device)

    print("Measuring throughput...")
    tps = measure_throughput(model, tokenizer, device)

    peak_mem = measure_peak_memory_mb(device)

    result = {
        "label": label,
        "model_path": model_path,
        "perplexity": round(ppl, 4),
        "tokens_per_sec": round(tps, 2),
        "peak_vram_mb": round(peak_mem, 1),
    }

    print(f"  perplexity:     {result['perplexity']}")
    print(f"  tokens/sec:     {result['tokens_per_sec']}")
    print(f"  peak VRAM (MB): {result['peak_vram_mb']}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def print_comparison_table(results):
    print("\n=== Quantization Benchmark Summary ===")
    header = f"{'Model':<12} {'Perplexity':<12} {'Tokens/sec':<12} {'Peak VRAM (MB)':<15}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['label']:<12} {r['perplexity']:<12} {r['tokens_per_sec']:<12} {r['peak_vram_mb']:<15}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    print("Loading evaluation text...")
    eval_texts = load_eval_texts()

    candidates = [("base", args.base_model)]
    if args.gptq_model:
        candidates.append(("gptq", args.gptq_model))
    if args.awq_model:
        candidates.append(("awq", args.awq_model))

    results = []
    for label, path in candidates:
        results.append(benchmark_model(label, path, device, eval_texts))

    print_comparison_table(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark base vs GPTQ vs AWQ models")
    parser.add_argument("--base_model", required=True, help="HF model id or local path (unquantized)")
    parser.add_argument("--gptq_model", default=None, help="Path to GPTQ-quantized model")
    parser.add_argument("--awq_model", default=None, help="Path to AWQ-quantized model")
    parser.add_argument("--output", default="../benchmarks/results/quantization_benchmark.json")
    args = parser.parse_args()

    main(args)
