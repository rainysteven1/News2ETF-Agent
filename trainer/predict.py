"""Batch inference: FinBERT (major+sentiment) + SetFit (sub-category) ONNX.

All configuration is read from config.toml — no CLI arguments.

Phase 1: FinBERT → intermediate parquet (major_category, sentiment, confidences)
Phase 2: SetFit sub-category → final output parquet
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
from loguru import logger
from transformers import AutoTokenizer

from trainer.finbert.dataset import L1_CATEGORIES, SENTIMENT_LABELS
from trainer.config import LabelStats
from trainer.setfit_module.model import _safe_name

# ─── ONNX Helpers ────────────────────────────────────────────────────────────────


def _make_ort_session(
    onnx_path: Path,
    intra_threads: int = 1,
    inter_threads: int = 1,
) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = inter_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _tokenize(texts: list[str], tokenizer: AutoTokenizer, max_length: int) -> dict[str, np.ndarray]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )  # type: ignore
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }


def _run_finbert_from_inputs(
    sess: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Run FinBERT from pre-tokenized inputs; returns (l1_logits, sent_logits)."""
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": np.zeros_like(inputs["input_ids"]).astype(np.int64),
    }
    return sess.run(None, onnx_inputs)  # type: ignore[return-value]


# ─── Phase 1: FinBERT ───────────────────────────────────────────────────────────


def run_finbert(limit_rows: int | None = None) -> Path | None:
    """Run FinBERT inference; saves intermediate parquet and returns its path."""
    from trainer.config import load_config

    cfg = load_config()
    p = cfg.predict

    input_path = p.input_path
    finbert_output_path = p.finbert_output_path
    finbert_onnx_dir = p.finbert_onnx_dir
    finbert_workers = p.finbert_workers
    batch_size = p.batch_size
    finbert_max_length = p.finbert_max_length

    assert input_path is not None and finbert_output_path is not None, (
        "input_path and finbert_output_path must be set in config.toml"
    )

    logger.info(f"[FinBERT] Loading raw data: {input_path}")
    df_raw = pl.read_parquet(input_path)

    required_cols = {"datetime", "title", "content"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")

    both_empty = df_raw.filter(pl.col("title").is_null() & pl.col("content").is_null()).height
    if both_empty > 0:
        logger.info(f"[FinBERT] Skipping {both_empty} rows with both title and content null")
        df_raw = df_raw.filter(~(pl.col("title").is_null() & pl.col("content").is_null()))

    null_title_valid_content = df_raw.filter(pl.col("title").is_null() & pl.col("content").is_not_null()).height
    if null_title_valid_content > 0:
        logger.info(f"[FinBERT] {null_title_valid_content} rows have null title (will use content)")

    if len(df_raw) == 0:
        raise ValueError("No valid rows to process")

    df_raw = df_raw.sort("datetime")

    if limit_rows is not None:
        df_raw = df_raw.head(limit_rows)

    n = len(df_raw)
    logger.info(f"[FinBERT] {n} valid rows to process")

    # ── Load FinBERT ONNX ────────────────────────────────────────────────────
    cpu_count = os.cpu_count() or 1
    intra_threads = 1
    inter_threads = max(1, cpu_count // 8)
    logger.info(f"[FinBERT] CPU cores={cpu_count}, ORT threads: intra={intra_threads}, inter={inter_threads}")

    assert finbert_onnx_dir is not None, "FinBERT ONNX directory must be set in config.toml"
    finbert_onnx_path = finbert_onnx_dir / "best.onnx"
    finbert_tokenizer_path = finbert_onnx_dir / "tokenizer"

    if not finbert_onnx_path.exists():
        raise FileNotFoundError(f"FinBERT ONNX not found: {finbert_onnx_path}")
    if not finbert_tokenizer_path.exists():
        raise FileNotFoundError(f"FinBERT tokenizer not found: {finbert_tokenizer_path}")

    logger.info(f"[FinBERT] Loading FinBERT ONNX: {finbert_onnx_path}")
    finbert_sess = _make_ort_session(finbert_onnx_path, intra_threads, inter_threads)
    finbert_tokenizer = AutoTokenizer.from_pretrained(str(finbert_tokenizer_path))

    # ── Pre-compute texts ────────────────────────────────────────────────────
    texts_for_finbert: list[str | None] = []
    for row in df_raw.iter_rows(named=True):
        title, content = row["title"], row["content"]
        if title is not None and title != "" and content is not None and content != "":
            texts_for_finbert.append(f"{title} [SEP] {content[:256]}")
        elif content is not None and content != "":
            texts_for_finbert.append(content[:256])
        else:
            texts_for_finbert.append(None)

    # ── Pre-tokenize all batches ────────────────────────────────────────────
    total_batches = (n + batch_size - 1) // batch_size
    logger.info(f"[FinBERT] {n} rows, {total_batches} batches...")
    t0 = time.monotonic()

    finbert_inputs_all: list[dict[str, np.ndarray]] = []
    for i in range(0, n, batch_size):
        batch_replaced = ["" if t is None else t for t in texts_for_finbert[i : i + batch_size]]
        finbert_inputs_all.append(_tokenize(batch_replaced, finbert_tokenizer, finbert_max_length))

    # ── Run FinBERT batch-by-batch ────────────────────────────────────────────
    majors_out = np.empty(n, dtype=object)
    sents_out = np.empty(n, dtype=object)
    l1_confs_out = np.empty(n, dtype=np.float64)
    sent_confs_out = np.empty(n, dtype=np.float64)

    with ThreadPoolExecutor(max_workers=finbert_workers) as executor:
        futures = []
        for batch_idx in range(total_batches):
            futures.append(
                (batch_idx, executor.submit(_run_finbert_from_inputs, finbert_sess, finbert_inputs_all[batch_idx]))
            )

        for batch_idx, fb_future in futures:
            start = batch_idx * batch_size
            end = min(start + batch_size, n)

            l1_logits, sent_logits = fb_future.result()

            l1_probs = _softmax(l1_logits)
            l1_pred_idx = l1_probs.argmax(axis=1)
            l1_conf = l1_probs[np.arange(len(l1_pred_idx)), l1_pred_idx]

            sent_probs = _softmax(sent_logits)
            sent_pred_idx = sent_probs.argmax(axis=1)
            sent_conf = sent_probs[np.arange(len(sent_pred_idx)), sent_pred_idx]

            batch_majors = [L1_CATEGORIES[i] for i in l1_pred_idx]
            batch_sents = [SENTIMENT_LABELS[i] for i in sent_pred_idx]

            majors_out[start:end] = batch_majors
            sents_out[start:end] = batch_sents
            l1_confs_out[start:end] = l1_conf
            sent_confs_out[start:end] = sent_conf

            elapsed = time.monotonic() - t0
            logger.info(
                f"  Batch {batch_idx + 1}/{total_batches} | FinBERT rows {start}-{end}/{n} | {elapsed:.1f}s total"
            )

    # ── Save intermediate ───────────────────────────────────────────────────
    result_df = df_raw.with_columns(
        pl.Series("major_category", majors_out),
        pl.Series("sentiment", sents_out),
        pl.Series("l1_confidence", l1_confs_out),
        pl.Series("sentiment_confidence", sent_confs_out),
    ).sort("datetime")

    finbert_output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(finbert_output_path)
    logger.success(f"[FinBERT] Done → {finbert_output_path}")
    return finbert_output_path


# ─── Phase 2: SetFit sub-category ──────────────────────────────────────────────


def run_setfit(limit_rows: int | None = None) -> None:
    """Run SetFit sub-category classification on FinBERT intermediate output."""
    from trainer.config import load_config

    cfg = load_config()
    p = cfg.predict

    output_path = p.output_path
    finbert_output_path = p.finbert_output_path
    setfit_base_dir = p.setfit_base_dir
    setfit_max_length = p.setfit_max_length

    intermediate_path = Path(finbert_output_path) if finbert_output_path else None
    if intermediate_path is None:
        assert output_path is not None, "output_path must be set in config.toml"
        intermediate_path = (
            Path(output_path).parent / f"{Path(output_path).stem}_finbert_only{Path(output_path).suffix}"
        )

    if not intermediate_path.exists():
        raise FileNotFoundError(f"Intermediate FinBERT result not found: {intermediate_path}")

    logger.info(f"[SetFit] Loading FinBERT intermediate: {intermediate_path}")
    df = pl.read_parquet(intermediate_path)

    if limit_rows is not None:
        df = df.head(limit_rows)

    n = len(df)
    logger.info(f"[SetFit] {n} rows to classify")

    # ── Load SetFit ONNX per major ───────────────────────────────────────────
    cpu_count = os.cpu_count() or 1
    intra_threads = 1
    inter_threads = max(1, cpu_count // 8)
    logger.info(f"[SetFit] CPU cores={cpu_count}, ORT threads: intra={intra_threads}, inter={inter_threads}")

    label_stats = LabelStats()
    major_categories = label_stats.get_major_categories()
    major_to_subcats: dict[str, list[str]] = {m: label_stats.get_sub_categories(m) for m in major_categories}

    setfit_sessions: dict[str, ort.InferenceSession] = {}
    setfit_tokenizers: dict[str, AutoTokenizer] = {}

    assert setfit_base_dir is not None, "SetFit base directory must be set in config.toml"

    for m in major_categories:
        safe = _safe_name(m)
        major_dir = setfit_base_dir / safe
        onnx_path = major_dir / "best.onnx"
        tokenizer_path = major_dir / "tokenizer"

        if not onnx_path.exists() or not tokenizer_path.exists():
            logger.warning(f"[SetFit] ONNX/tokenizer not found for '{safe}', will use fallback '其他'")
            continue

        logger.info(f"[SetFit] Loading SetFit for '{safe}'")
        setfit_sessions[safe] = _make_ort_session(onnx_path, intra_threads, inter_threads)
        setfit_tokenizers[safe] = AutoTokenizer.from_pretrained(str(tokenizer_path))

    if not setfit_sessions:
        raise RuntimeError("No SetFit ONNX models loaded")

    subcats_lookup = {
        **{k: v for k, v in major_to_subcats.items()},
        **{_safe_name(k): v for k, v in major_to_subcats.items()},
    }

    # ── Prepare texts ───────────────────────────────────────────────────────
    texts_for_setfit: list[str | None] = []
    for row in df.iter_rows(named=True):
        title, content = row["title"], row["content"]
        if title is not None and title != "" and content is not None and content != "":
            texts_for_setfit.append(f"{title} [SEP] {content[:256]}")
        elif content is not None and content != "":
            texts_for_setfit.append(content[:256])
        else:
            texts_for_setfit.append(None)

    # ── Group by safe major ──────────────────────────────────────────────────
    safe_to_global_idx: dict[str, list[int]] = {}
    safe_to_texts: dict[str, list[str]] = {}

    for i, row in df.iter_rows(named=True):
        major = row["major_category"]
        safe = _safe_name(major)
        if safe not in safe_to_global_idx:
            safe_to_global_idx[safe] = []
            safe_to_texts[safe] = []
        safe_to_global_idx[safe].append(i)
        t = texts_for_setfit[i]
        safe_to_texts[safe].append("" if t is None else t)

    # ── Classify per major ───────────────────────────────────────────────────
    sub_cats_out = np.full(n, "其他", dtype=object)
    sub_confs_out = np.zeros(n, dtype=np.float64)

    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}

        for safe, g_indices in safe_to_global_idx.items():
            if safe not in setfit_sessions:
                sub_cats = subcats_lookup.get(safe, ["其他"])
                for g in g_indices:
                    sub_cats_out[g] = sub_cats[-1]
                    sub_confs_out[g] = 0.0
                logger.info(f"  SetFit fallback {safe}: {len(g_indices)} rows (no model)")
                continue

            texts_list = safe_to_texts[safe]

            def _classify(safe_m: str, g_idx: list[int], txts: list[str]):
                inputs = _tokenize(txts, setfit_tokenizers[safe_m], setfit_max_length)
                onnx_in = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }
                logits = setfit_sessions[safe_m].run(None, onnx_in)[0]
                probs = _softmax(logits)
                pred_idx = probs.argmax(axis=1)
                conf = probs[np.arange(len(pred_idx)), pred_idx]
                sub_cats = subcats_lookup.get(safe_m, ["其他"])
                results = []
                for g, p_idx, c_val in zip(g_idx, pred_idx.tolist(), conf.tolist()):
                    safe_idx = min(p_idx, len(sub_cats) - 1)
                    results.append((g, sub_cats[safe_idx], float(c_val)))
                return safe_m, results

            futures[executor.submit(_classify, safe, g_indices, texts_list)] = safe

        for future in futures:
            safe_major, results = future.result()
            for g, sub_cat, conf_val in results:
                sub_cats_out[g] = sub_cat
                sub_confs_out[g] = conf_val
            elapsed = time.monotonic() - t0
            logger.info(f"  SetFit {safe_major}: {len(results)} rows | {elapsed:.1f}s total")

    # ── Save final output ────────────────────────────────────────────────────
    result_df = df.with_columns(
        pl.Series("sub_category", sub_cats_out),
        pl.Series("sub_category_confidence", sub_confs_out),
    ).sort("datetime")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)
    logger.success(f"[SetFit] Complete output → {output_path}")


# ─── Main ───────────────────────────────────────────────────────────────────────


def run(limit_rows: int | None = None) -> None:
    """Run full pipeline: FinBERT → SetFit."""
    logger.info("[Predict] === Phase 1: FinBERT ===")
    run_finbert(limit_rows=limit_rows)

    logger.info("[Predict] === Phase 2: SetFit sub-category ===")
    run_setfit(limit_rows=limit_rows)

    logger.info("[Predict] === All done ===")
