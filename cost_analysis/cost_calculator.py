"""
Convert raw benchmark numbers into a cost-per-request / cost-per-1M-tokens
comparison between the base model and its quantized versions.

This is the piece that turns "AWQ gave us 2.3x more tokens/sec" into
"AWQ cuts inference cost by 56% on the same GPU" — the number that
actually matters when explaining an optimization to someone non-technical.

Inputs:
  - quantization/benchmark_quantization.py output (tokens/sec, VRAM per model)
  - a GPU hourly rate (yours, or a cloud on-demand price)

Usage:
    python cost_calculator.py \
        --benchmark_file ../benchmarks/results/quantization_benchmark.json \
        --gpu_hourly_cost 2.50 \
        --output ../cost_analysis/reports/cost_report.json
"""

import argparse
import json
from pathlib import Path


def compute_cost_per_1m_tokens(tokens_per_sec: float, gpu_hourly_cost: float) -> float:
    """
    Cost to generate 1M tokens, assuming the GPU runs at this throughput
    the whole time. This is a simplification (real traffic is bursty, not
    a constant stream) but it's the standard way to compare models on an
    apples-to-apples basis: same GPU, same price, different throughput.
    """
    if tokens_per_sec <= 0:
        return float("inf")

    seconds_per_1m_tokens = 1_000_000 / tokens_per_sec
    hours_per_1m_tokens = seconds_per_1m_tokens / 3600
    return hours_per_1m_tokens * gpu_hourly_cost


def compute_requests_per_dollar(tokens_per_sec: float, gpu_hourly_cost: float, avg_tokens_per_request: int = 150) -> float:
    """How many typical requests $1 of GPU time buys, at this throughput."""
    if gpu_hourly_cost <= 0:
        return float("inf")

    tokens_per_dollar = (tokens_per_sec * 3600) / gpu_hourly_cost
    return tokens_per_dollar / avg_tokens_per_request


def build_report(benchmark_results: list, gpu_hourly_cost: float, avg_tokens_per_request: int):
    report = []
    baseline_cost = None

    for entry in benchmark_results:
        cost_per_1m = compute_cost_per_1m_tokens(entry["tokens_per_sec"], gpu_hourly_cost)
        requests_per_dollar = compute_requests_per_dollar(
            entry["tokens_per_sec"], gpu_hourly_cost, avg_tokens_per_request
        )

        if entry["label"] == "base":
            baseline_cost = cost_per_1m

        report.append({
            "label": entry["label"],
            "tokens_per_sec": entry["tokens_per_sec"],
            "peak_vram_mb": entry.get("peak_vram_mb"),
            "cost_per_1m_tokens_usd": round(cost_per_1m, 4),
            "requests_per_dollar": round(requests_per_dollar, 1),
        })

    # Add a percentage savings column relative to the unquantized baseline,
    # since "56% cheaper than base" communicates more than a raw dollar figure.
    if baseline_cost:
        for row in report:
            savings_pct = (1 - (row["cost_per_1m_tokens_usd"] / baseline_cost)) * 100
            row["savings_vs_base_pct"] = round(savings_pct, 1)

    return report


def print_summary(report: list):
    print("\n=== Cost Comparison (per 1M tokens) ===")
    header = f"{'Model':<8} {'$/1M tokens':<14} {'Requests/$':<12} {'Savings vs base':<16}"
    print(header)
    print("-" * len(header))
    for row in report:
        savings = f"{row.get('savings_vs_base_pct', 0)}%" if "savings_vs_base_pct" in row else "-"
        print(f"{row['label']:<8} {row['cost_per_1m_tokens_usd']:<14} {row['requests_per_dollar']:<12} {savings:<16}")


def main(args):
    with open(args.benchmark_file, "r") as f:
        benchmark_results = json.load(f)

    report = build_report(benchmark_results, args.gpu_hourly_cost, args.avg_tokens_per_request)
    print_summary(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved cost report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cost-per-token comparison across models")
    parser.add_argument("--benchmark_file", required=True, help="Output from benchmark_quantization.py")
    parser.add_argument("--gpu_hourly_cost", type=float, required=True, help="e.g. 2.50 for an on-demand A10G")
    parser.add_argument("--avg_tokens_per_request", type=int, default=150)
    parser.add_argument("--output", default="reports/cost_report.json")
    args = parser.parse_args()

    main(args)
