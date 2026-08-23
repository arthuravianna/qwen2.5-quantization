from datetime import datetime
import os
from pathlib import PurePath
import sys
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import BACKEND, AWQConfig, GGUFConfig, GPTQConfig, GPTQModel 
from gptqmodel.quantization import EXL3Config, FORMAT

QUANTIZATION_METHODS = ["gptq", "gguf", "awq", "exl3"]
BITS=4


def _compute_path_size_bytes(path):
    if not os.path.exists(path):
        return None

    if os.path.isfile(path):
        return os.path.getsize(path)

    total_size = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size


def _size_breakdown(size_bytes):
    if size_bytes is None:
        return {
            "bytes": None,
            "mb": None,
            "gb": None,
        }

    mb = size_bytes / (1024 * 1024)
    gb = mb / 1024
    return {
        "bytes": int(size_bytes),
        "mb": round(mb, 2),
        "gb": round(gb, 2),
    }


def build_quantization_summary(model_path, quant_path, quant_method):
    pre_size_bytes = _compute_path_size_bytes(model_path)
    quant_size_bytes = _compute_path_size_bytes(quant_path)

    size_diff_bytes = None
    percent_reduction = None
    if pre_size_bytes is not None and quant_size_bytes is not None:
        size_diff_bytes = pre_size_bytes - quant_size_bytes
        if pre_size_bytes > 0:
            percent_reduction = round((size_diff_bytes / pre_size_bytes) * 100, 2)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "quantization_method": quant_method,
        "bits": BITS,
        "source_model_path": model_path,
        "quantized_model_path": quant_path,
        "pre_quantized_model_size": _size_breakdown(pre_size_bytes),
        "quantized_model_size": _size_breakdown(quant_size_bytes),
        "size_difference": _size_breakdown(size_diff_bytes),
        "size_reduction_percent": percent_reduction,
    }

    if pre_size_bytes is None:
        summary["notes"] = "Source model path not found locally. Size computed only for local paths."

    return summary


def quantize_gptq(model_path, quant_path):
    print(f"Quantizing {model_path} to {quant_path} using GPTQ method.")
    # If model_id is a Hugging Face repo name, it will:
    #   use the local HF cache if already present
    #   otherwise download tokenizer files
    # If model_id is a local folder, it loads from that folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Simple calibration dataset, following GPTQModel's docs pattern
    # Downloads the calibration dataset if it is not already cached
    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train"
    ).select(range(256))["text"]


    # Then tune if needed:

    # bits: usually 4; 8 keeps more quality but saves less memory
    # group_size: usually 128; sometimes 64 helps quality
    # desc_act: activation-order option; can affect quality/speed
    # sym: symmetric quantization
    # damp_percent: GPTQ damping
    # mse: MSE-based fitting option
    # act_group_aware=True: GPTQModel docs mention GAR; typically used with desc_act=False
    # dynamic={...}: per-module overrides/skips
    
    quant_config = GPTQConfig(
        bits=BITS,
        group_size=128,
    )

    # Loads the original full-precision model
    #  if model_id is a repo name, it may download model weights into the local HF cache
    #  If model_id is a local directory, it loads from there
    model = GPTQModel.load(model_path, quant_config)

    # Start with batch_size=1 on limited VRAM
    # Runs calibration samples through the model
    #  Measures quantization error/sensitivity
    #  Produces quantized weights according to GPTQConfig
    quantization_result = model.quantize(calibration_dataset, batch_size=1)

    # Writes the quantized model to quant_path
    model.save(quant_path)

    return quantization_result


def quantize_gguf(model_path, quant_path):
    print(f"Quantizing {model_path} to {quant_path} using GGUF method.")

    qcfg = GGUFConfig(
        bits=BITS
    )

    model = GPTQModel.load(model_path, qcfg)
    quantization_result = model.quantize(calibration=None)
    model.save(quant_path)

    return quantization_result


def quantize_awq(model_path, quant_path):
    print(f"Quantizing {model_path} to {quant_path} using AWQ method.")

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(1024))["text"]

    qcfg = AWQConfig(
        bits=BITS
    )

    model = GPTQModel.load(model_path, qcfg)
    quantization_result = model.quantize(calibration=calibration_dataset)
    model.save(quant_path)

    return quantization_result


def quantize_exl3(model_path, quant_path):
    print(f"Quantizing {model_path} to {quant_path} using EXL3 method.")

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(1024))["text"]

    qcfg = EXL3Config(
        bits=float(BITS),        # target average bits-per-weight
        head_bits=6.0,   # optional higher bitrate for attention heads / sensitive tensors
        codebook="mcg",  # one of: mcg, mul1, 3inst
    )

    model = GPTQModel.load(model_path, qcfg)
    quantization_result = model.quantize(calibration_dataset, batch_size=1)
    model.save(quant_path)

    return quantization_result


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python quantize.py <model_path> <quant_path> <quantization_method>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    quant_path = sys.argv[2]
    quant_method = sys.argv[3].lower()

    quantization_result = None
    if quant_method == "gptq":
        quantization_result = quantize_gptq(model_path, quant_path)
    elif quant_method == "gguf":
        quantization_result = quantize_gguf(model_path, quant_path)
    elif quant_method == "awq":
        quantization_result = quantize_awq(model_path, quant_path)
    elif quant_method == "exl3":
        quantization_result = quantize_exl3(model_path, quant_path)
    else:
        print(f"Unknown quantization method: {quant_method}")
        sys.exit(1)

    # Save quantization result to a JSON file
    model_name = PurePath(model_path).parts[-1]
    result_dir = f"quantization/{model_name}"
    os.makedirs(result_dir, exist_ok=True)

    result_file = os.path.join(result_dir, f"{quant_method}.json")
    result_summary = build_quantization_summary(model_path, quant_path, quant_method)
    with open(result_file, "w") as f:
        json.dump(result_summary, f, indent=4)