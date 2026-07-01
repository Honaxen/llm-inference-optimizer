"""
Quantize a base language model to 4-bit using GPTQ.

Usage:
    python quantize_gptq.py --model_id meta-llama/Llama-3.2-1B --output_dir ./models/llama-3.2-1b-gptq
"""

import argparse
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


def load_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """
    Load a small sample of real text for GPTQ calibration.
    GPTQ needs real text to estimate which weights matter most
    before deciding how aggressively to compress each one.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    samples = []
    for example in dataset:
        text = example["text"].strip()
        if len(text) > 0:
            tokenized = tokenizer(text, return_tensors="pt")
            if tokenized["input_ids"].shape[1] >= seq_len:
                samples.append(tokenized)
        if len(samples) >= n_samples:
            break
    return samples


def quantize(model_id: str, output_dir: str, bits: int = 4, group_size: int = 128):
    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,  # faster inference, small accuracy tradeoff
    )

    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

    print("Preparing calibration data...")
    calibration_data = load_calibration_data(tokenizer)

    print(f"Quantizing to {bits}-bit (group_size={group_size})...")
    model.quantize(calibration_data)

    print(f"Saving quantized model to {output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model with GPTQ")
    parser.add_argument("--model_id", required=True, help="HF model id or local path")
    parser.add_argument("--output_dir", required=True, help="Where to save the quantized model")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)
    args = parser.parse_args()

    quantize(args.model_id, args.output_dir, args.bits, args.group_size)
