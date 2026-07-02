"""
Quantize a base language model to 4-bit using AWQ (Activation-aware Weight Quantization).

AWQ protects the small subset of weights that activations depend on most,
instead of compressing every weight equally like plain round-to-nearest quantization.
This tends to preserve accuracy better than GPTQ at the same bit-width,
at the cost of a slightly different calibration process.

Usage:
    python quantize_awq.py --model_id meta-llama/Llama-3.2-1B --output_dir ./models/llama-3.2-1b-awq
"""

import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def quantize(model_id: str, output_dir: str, bits: int = 4, group_size: int = 128, zero_point: bool = True):
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",  # GEMM kernel: best throughput for batched inference
    }

    print(f"Quantizing to {bits}-bit (group_size={group_size}) with AWQ...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving quantized model to {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model with AWQ")
    parser.add_argument("--model_id", required=True, help="HF model id or local path")
    parser.add_argument("--output_dir", required=True, help="Where to save the quantized model")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    args = parser.parse_args()

    quantize(args.model_id, args.output_dir, args.bits, args.group_size)
