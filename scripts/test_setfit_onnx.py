#!/usr/bin/env python3
"""Test SetFit ONNX model for a given category."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np
from transformers import AutoTokenizer


def load_onnx_session(onnx_path: Path):
    """Load ONNX Runtime session with performance options."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    return ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])


def run_onnx(
    onnx_session,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_seq_length: int = 256,
) -> tuple[np.ndarray, float]:
    """Run ONNX model. Returns (logits, elapsed_ms)."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    onnx_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    t0 = time.perf_counter()
    logits = onnx_session.run(None, onnx_inputs)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return logits, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Test SetFit ONNX model")
    parser.add_argument("--category", "-c", required=True, help="Major category name")
    parser.add_argument(
        "--base-dir",
        "-d",
        default="trainer/checkpoints/setfit/setfit-0324-0125",
        help="Base checkpoint directory",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Max sequence length used during ONNX export",
    )
    args = parser.parse_args()

    category_dir = Path(args.base_dir) / args.category
    onnx_path = category_dir / "best.onnx"
    tokenizer_path = category_dir / "tokenizer"

    for p, name in [(onnx_path, "ONNX model"), (tokenizer_path, "Tokenizer")]:
        if not p.exists():
            print(f"[FAIL] {name} not found: {p}")
            sys.exit(1)

    print(f"Category: {args.category}")
    print(f"ONNX model: {onnx_path}")
    print(f"Tokenizer:  {tokenizer_path}")
    print()

    sample_texts = [
        "人工智能技术在金融领域的应用前景广阔",
        "新能源车销量大幅增长，产业链公司受益",
        "医药板块迎来政策利好，创新药研发加速",
    ]
    print(f"[INFO] Testing with {len(sample_texts)} sample texts:")
    for i, t in enumerate(sample_texts):
        print(f"  [{i + 1}] {t[:50]}...")
    print()

    # Load
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    print("[INFO] Loading ONNX session...")
    onnx_session = load_onnx_session(onnx_path)

    # Warmup
    print("[INFO] Warming up ONNX Runtime...")
    _, _ = run_onnx(onnx_session, tokenizer, sample_texts[:1], args.max_seq_length)

    # Run 3 times and average
    times = []
    for i in range(3):
        logits, t_ms = run_onnx(onnx_session, tokenizer, sample_texts, args.max_seq_length)
        times.append(t_ms)

    preds = logits.argmax(axis=1).tolist()
    avg_time = sum(times) / len(times)

    print(f"[OK] Predictions: {preds}")
    print(f"[OK] Logits shape: {logits.shape}")
    print(f"[OK] Avg inference time: {avg_time:.1f} ms (3 runs)")
    print()
    print("[OK] ONNX model is valid")


if __name__ == "__main__":
    main()
