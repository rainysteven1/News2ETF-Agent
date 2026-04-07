"""Batch inference: Major + Sub ONNX pipeline.

Usage:
    python -m trainer.main predict major --major-shard-workers 4
    python -m trainer.main predict sub --sub-shard-workers 4 --sub-major-workers 8
    python -m trainer.main predict all --major-shard-workers 4 --sub-shard-workers 4 --sub-major-workers 8

Recommended:
    - Configure `predict.major_input_dir` for raw news parquet shards
    - Configure `predict.sub_input_dir` or `predict.sub_input_paths` explicitly for sub classification inputs
    - Organize sub ONNX models as `predict.sub_onnx_dir/<backend>/<major>/...`
    - Use process-level shard parallelism to exploit multi-core CPUs
    - Keep per-process ONNX threads modest to avoid CPU oversubscription

Phase 1: Major → intermediate parquet (major_category, sentiment, confidences)
Phase 2: Sub-category → final output parquet
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import onnxruntime as ort
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from transformers import AutoTokenizer

from trainer.src.config import LabelStats, load_config, safe_name
from trainer.src.datasets.major import L1_CATEGORIES, SENTIMENT_LABELS


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
    available = ort.get_available_providers()
    preferred: list[str] = []
    if "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")
    preferred.append("CPUExecutionProvider")
    session = ort.InferenceSession(str(onnx_path), opts, providers=preferred)
    logger.info(f"[ORT] {onnx_path.name} providers={session.get_providers()}")
    return session


def _resolve_ort_threads(cpu_count: int, shard_workers: int) -> tuple[int, int]:
    per_process_budget = max(1, cpu_count // max(1, shard_workers))
    intra_threads = max(1, min(per_process_budget, 16))
    inter_threads = 1
    return intra_threads, inter_threads


def _has_cuda_ort_provider() -> bool:
    try:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


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
    )
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }


def _run_major_from_inputs(
    sess: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
        "token_type_ids": np.zeros_like(inputs["input_ids"]).astype(np.int64),
    }
    return sess.run(None, onnx_inputs)


def _scan_input_dir(input_dir: Path, pattern: str, required_columns: set[str] | None = None) -> list[Path]:
    candidates = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    valid_inputs: list[Path] = []
    for path in candidates:
        if required_columns is None:
            valid_inputs.append(path)
            continue
        try:
            schema_names = set(pl.scan_parquet(path).collect_schema().names())
        except Exception as exc:
            logger.warning(f"[Predict] Skip unreadable parquet {path}: {exc}")
            continue
        if required_columns <= schema_names:
            valid_inputs.append(path)
        else:
            logger.info(f"[Predict] Skip parquet with incompatible schema: {path}")
    return valid_inputs


def _get_major_input_paths(prediction_cfg) -> list[Path]:
    if prediction_cfg.major_input_dir:
        input_dir = Path(prediction_cfg.major_input_dir)
        pattern = prediction_cfg.major_input_glob or "*.parquet"
        return _scan_input_dir(input_dir, pattern, required_columns={"datetime", "title", "content"})
    if prediction_cfg.major_input_paths:
        return [Path(p) for p in prediction_cfg.major_input_paths]
    if prediction_cfg.major_input_path:
        return [Path(prediction_cfg.major_input_path)]
    if prediction_cfg.input_dir:
        input_dir = Path(prediction_cfg.input_dir)
        pattern = prediction_cfg.input_glob or "*.parquet"
        return _scan_input_dir(input_dir, pattern, required_columns={"datetime", "title", "content"})
    if prediction_cfg.input_paths:
        return [Path(p) for p in prediction_cfg.input_paths]
    assert prediction_cfg.input_path is not None, "major input path(s) must be set in config.toml"
    return [Path(prediction_cfg.input_path)]


def _get_sub_input_paths(prediction_cfg) -> list[Path]:
    if prediction_cfg.sub_input_dir:
        input_dir = Path(prediction_cfg.sub_input_dir)
        pattern = prediction_cfg.sub_input_glob or "*.parquet"
        return _scan_input_dir(
            input_dir,
            pattern,
            required_columns={"datetime", "title", "content", "major_category", "sentiment"},
        )
    if prediction_cfg.sub_input_paths:
        return [Path(p) for p in prediction_cfg.sub_input_paths]
    if prediction_cfg.sub_input_path:
        return [Path(prediction_cfg.sub_input_path)]
    return _get_major_intermediate_paths(prediction_cfg)


def _get_prediction_input_paths(prediction_cfg) -> list[Path]:
    if prediction_cfg.input_dir:
        input_dir = Path(prediction_cfg.input_dir)
        pattern = prediction_cfg.input_glob or "*.parquet"
        return _scan_input_dir(input_dir, pattern, required_columns={"datetime", "title", "content"})
    if prediction_cfg.input_paths:
        return [Path(p) for p in prediction_cfg.input_paths]
    assert prediction_cfg.input_path is not None, "input_path or input_paths must be set in config.toml"
    return [Path(prediction_cfg.input_path)]


def _get_major_intermediate_paths(prediction_cfg) -> list[Path]:
    if prediction_cfg.major_output_dir:
        output_dir = Path(prediction_cfg.major_output_dir)
        return sorted(p for p in output_dir.glob("*_major_only.parquet") if p.is_file())
    if prediction_cfg.input_dir:
        input_paths = _get_prediction_input_paths(prediction_cfg)
        return [
            _derive_shard_output_path(input_path, prediction_cfg.major_output_path, "_major_only", len(input_paths))
            for input_path in input_paths
        ]
    if prediction_cfg.input_paths:
        return [
            _derive_shard_output_path(
                input_path, prediction_cfg.major_output_path, "_major_only", len(prediction_cfg.input_paths)
            )
            for input_path in prediction_cfg.input_paths
        ]
    assert prediction_cfg.major_output_path is not None, (
        "major_output_path or major_output_dir must be set in config.toml"
    )
    return [Path(prediction_cfg.major_output_path)]


def _derive_major_output_path(input_path: Path, prediction_cfg, total_inputs: int) -> Path:
    if prediction_cfg.major_output_dir is not None:
        return Path(prediction_cfg.major_output_dir) / f"{input_path.stem}_major_only.parquet"
    return _derive_shard_output_path(input_path, prediction_cfg.major_output_path, "_major_only", total_inputs)


def _derive_sub_output_path(intermediate_path: Path, prediction_cfg, total_inputs: int) -> Path:
    if prediction_cfg.output_dir is not None:
        stem = intermediate_path.stem.removesuffix("_major_only")
        return Path(prediction_cfg.output_dir) / f"{stem}_sub.parquet"
    source_input = intermediate_path.with_name(
        f"{intermediate_path.stem.removesuffix('_major_only')}{intermediate_path.suffix}"
    )
    return _derive_shard_output_path(source_input, prediction_cfg.output_path, "_sub", total_inputs)


def _derive_shard_output_path(
    input_path: Path,
    configured_output: Path | None,
    suffix: str,
    total_inputs: int,
) -> Path:
    if configured_output is None:
        return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")
    if total_inputs == 1:
        return Path(configured_output)
    base_output = Path(configured_output)
    output_dir = base_output.parent if base_output.suffix else base_output
    return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"


def _effective_parallelism(requested: int | None, total_items: int) -> int:
    if total_items <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    max_reasonable = min(total_items, cpu_count)
    if requested is None:
        return max(1, min(max_reasonable, 4))
    return max(1, min(requested, max_reasonable))


def _should_log_progress(current_batch: int, total_batches: int, interval: int = 100) -> bool:
    return current_batch == 1 or current_batch == total_batches or current_batch % interval == 0


def _should_log_heartbeat(
    current_batch: int,
    total_batches: int,
    now: float,
    last_log_at: float,
    min_interval_seconds: float = 5.0,
) -> bool:
    return _should_log_progress(current_batch, total_batches) or (now - last_log_at) >= min_interval_seconds


def _build_major_batch_texts(batch_df: pl.DataFrame) -> list[str]:
    texts: list[str] = []
    for row in batch_df.iter_rows(named=True):
        title, content = row["title"], row["content"]
        if title is not None and title != "" and content is not None and content != "":
            texts.append(f"{title} [SEP] {content[:256]}")
        elif content is not None and content != "":
            texts.append(content[:256])
        else:
            texts.append("")
    return texts


def _build_sub_text_expr() -> pl.Expr:
    title = pl.col("title").cast(pl.Utf8).fill_null("")
    content = pl.col("content").cast(pl.Utf8).fill_null("")
    content_short = content.str.slice(0, 256)
    return (
        pl.when((title != "") & (content != ""))
        .then(pl.concat_str([title, pl.lit(" [SEP] "), content_short], separator=""))
        .when(content != "")
        .then(content_short)
        .otherwise(title)
        .alias("sub_text")
    )


def _build_sub_groups(df: pl.DataFrame) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    grouped_df = df.with_row_index("row_idx").with_columns(_build_sub_text_expr())
    safe_to_global_idx: dict[str, list[int]] = {}
    safe_to_texts: dict[str, list[str]] = {}
    for part in grouped_df.partition_by("major_category", maintain_order=True):
        major = part.item(0, "major_category")
        safe = safe_name(major)
        safe_to_global_idx[safe] = part["row_idx"].to_list()
        safe_to_texts[safe] = part["sub_text"].to_list()
    return safe_to_global_idx, safe_to_texts


def _load_sub_labels(
    safe_major: str,
    label_map_path: Path,
    subcats_lookup: dict[str, list[str]],
) -> list[str]:
    if label_map_path.exists():
        with open(label_map_path, encoding="utf-8") as f:
            label_map = json.load(f)
        idx_to_label = label_map.get("idx_to_label", {})
        if idx_to_label:
            return [idx_to_label[str(i)] for i in range(len(idx_to_label))]
    return subcats_lookup.get(safe_major, ["其他"])


def _discover_sub_model_dirs(sub_onnx_dir: Path, majors: list[str]) -> dict[str, tuple[str, Path]]:
    backend_dirs = sorted(p for p in sub_onnx_dir.iterdir() if p.is_dir())
    if not backend_dirs:
        raise FileNotFoundError(f"No backend directories found under sub_onnx_dir: {sub_onnx_dir}")

    resolved: dict[str, tuple[str, Path]] = {}
    errors: list[str] = []
    for major in majors:
        safe = safe_name(major)
        matches: list[tuple[str, Path]] = []
        for backend_dir in backend_dirs:
            candidate = backend_dir / safe
            if candidate.is_dir():
                matches.append((backend_dir.name, candidate))

        if len(matches) != 1:
            found = [f"{backend}/{safe}" for backend, _ in matches] or ["<none>"]
            errors.append(f"{major} ({safe}) -> expected exactly 1 match, found {len(matches)}: {found}")
            continue

        resolved[safe] = matches[0]

    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(f"Invalid sub_onnx_dir layout under {sub_onnx_dir}:\n{joined}")
    return resolved


def _log_sub_model_layout(resolved_model_dirs: dict[str, tuple[str, Path]], majors: list[str]) -> None:
    logger.info("[Sub] Validated sub model layout:")
    for major in majors:
        safe = safe_name(major)
        backend_name, model_dir = resolved_model_dirs[safe]
        logger.info(f"[Sub/{backend_name}] major={major} | dir={model_dir}")


def _filter_major_batch_rows(batch_dict: dict[str, list[object]]) -> tuple[dict[str, list[object]], list[str], int]:
    titles = batch_dict["title"]
    contents = batch_dict["content"]
    keep_indices: list[int] = []
    texts: list[str] = []

    for idx, (title, content) in enumerate(zip(titles, contents, strict=False)):
        title_text = str(title) if title is not None else ""
        content_text = str(content) if content is not None else ""
        if title_text == "" and content_text == "":
            continue
        keep_indices.append(idx)
        if title_text and content_text:
            texts.append(f"{title_text} [SEP] {content_text[:256]}")
        elif content_text:
            texts.append(content_text[:256])
        else:
            texts.append("")

    filtered_batch = {key: [values[i] for i in keep_indices] for key, values in batch_dict.items()}
    skipped = len(titles) - len(keep_indices)
    return filtered_batch, texts, skipped


def _major_writer_worker(
    output_path: Path,
    queue: Queue[tuple[int, pa.Table] | None],
) -> None:
    writer: pq.ParquetWriter | None = None
    try:
        while True:
            item = queue.get()
            try:
                if item is None:
                    return
                _, table = item
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
            finally:
                queue.task_done()
    finally:
        if writer is not None:
            writer.close()


def _run_major_single(
    input_path: Path,
    major_output_path: Path,
    major_onnx_dir: Path,
    major_workers: int,
    shard_workers: int,
    batch_size: int,
    major_max_length: int,
    limit_rows: int | None = None,
) -> Path:
    logger.info(f"[Major] Loading raw data: {input_path}")
    parquet_file = pq.ParquetFile(input_path)
    total_rows = parquet_file.metadata.num_rows
    schema_names = set(parquet_file.schema_arrow.names)
    logger.info(f"[Major] Parquet opened: {input_path} | rows={total_rows}")

    required_cols = {"datetime", "title", "content"}
    missing = required_cols - schema_names
    if missing:
        raise ValueError(f"Input parquet missing required columns: {missing}")

    cpu_count = os.cpu_count() or 1
    intra_threads, inter_threads = _resolve_ort_threads(cpu_count, shard_workers)
    logger.info(f"[Major] CPU cores={cpu_count}, ORT threads: intra={intra_threads}, inter={inter_threads}")

    assert major_onnx_dir is not None, "major_onnx_dir must be set in config.toml"
    major_onnx_path = major_onnx_dir / "best.onnx"
    major_tokenizer_path = major_onnx_dir / "tokenizer"

    if not major_onnx_path.exists():
        raise FileNotFoundError(f"Major ONNX not found: {major_onnx_path}")
    if not major_tokenizer_path.exists():
        raise FileNotFoundError(f"Major tokenizer not found: {major_tokenizer_path}")

    logger.info(f"[Major] Loading Major ONNX: {major_onnx_path}")
    major_sess = _make_ort_session(major_onnx_path, intra_threads, inter_threads)
    major_tokenizer = AutoTokenizer.from_pretrained(str(major_tokenizer_path))

    row_budget = limit_rows if limit_rows is not None else total_rows
    total_batches = (row_budget + batch_size - 1) // batch_size
    logger.info(f"[Major] Streaming {row_budget} rows, {total_batches} batches | tokenize + inference + parquet write")
    t0 = time.monotonic()

    processed_input_rows = 0
    written_rows = 0
    skipped_rows = 0
    last_tokenize_log_at = t0
    last_infer_log_at = t0
    write_queue: Queue[tuple[int, pa.Table] | None] = Queue(maxsize=8)
    writer_thread = Thread(
        target=_major_writer_worker,
        args=(major_output_path, write_queue),
        name=f"major-writer-{input_path.stem}",
        daemon=True,
    )
    writer_thread.start()

    if major_workers != 1:
        logger.warning(
            f"[Major] major_workers={major_workers} is over-parallel for CPU ONNX inference;"
            " using synchronous per-process batching instead"
        )

    batch_idx = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        if processed_input_rows >= row_budget:
            break

        batch_dict = batch.to_pydict()
        raw_batch_rows = len(next(iter(batch_dict.values()))) if batch_dict else 0
        if raw_batch_rows == 0:
            continue

        if processed_input_rows + raw_batch_rows > row_budget:
            keep = row_budget - processed_input_rows
            batch_dict = {key: values[:keep] for key, values in batch_dict.items()}
            raw_batch_rows = keep

        processed_input_rows += raw_batch_rows
        batch_idx += 1
        filtered_batch, batch_texts, batch_skipped = _filter_major_batch_rows(batch_dict)
        skipped_rows += batch_skipped

        if batch_skipped > 0 and _should_log_progress(batch_idx, total_batches):
            logger.info(f"[Major] Skipping {batch_skipped} empty rows in batch {batch_idx}/{total_batches}")

        if not batch_texts:
            continue

        now = time.monotonic()
        if _should_log_heartbeat(batch_idx, total_batches, now, last_tokenize_log_at):
            logger.info(
                f"[Major] Tokenizing batch {batch_idx}/{total_batches} | input_rows={processed_input_rows}/{row_budget}"
            )
            last_tokenize_log_at = now

        inputs = _tokenize(batch_texts, major_tokenizer, major_max_length)
        l1_logits, sent_logits = _run_major_from_inputs(major_sess, inputs)

        l1_probs = _softmax(l1_logits)
        l1_pred_idx = l1_probs.argmax(axis=1)
        l1_conf = l1_probs[np.arange(len(l1_pred_idx)), l1_pred_idx]

        sent_probs = _softmax(sent_logits)
        sent_pred_idx = sent_probs.argmax(axis=1)
        sent_conf = sent_probs[np.arange(len(sent_pred_idx)), sent_pred_idx]

        batch_majors = [L1_CATEGORIES[i] for i in l1_pred_idx]
        batch_sents = [SENTIMENT_LABELS[i] for i in sent_pred_idx]
        filtered_batch["major_category"] = batch_majors
        filtered_batch["sentiment"] = batch_sents
        filtered_batch["l1_confidence"] = l1_conf.tolist()
        filtered_batch["sentiment_confidence"] = sent_conf.tolist()
        table = pa.table(filtered_batch)
        write_queue.put((batch_idx, table))
        written_rows += len(batch_majors)

        now = time.monotonic()
        if _should_log_heartbeat(batch_idx, total_batches, now, last_infer_log_at):
            elapsed = now - t0
            logger.info(
                f"[Major] Inference batch {batch_idx}/{total_batches}"
                f" | written_rows={written_rows}/{row_budget - skipped_rows}"
                f" | {elapsed:.1f}s total"
            )
            last_infer_log_at = now

    write_queue.put(None)
    write_queue.join()
    writer_thread.join()
    if written_rows == 0:
        raise ValueError("No valid rows to process")
    logger.success(f"[Major] Done → {major_output_path}")
    return major_output_path


def run_major(
    limit_rows: int | None = None,
    overwrite: bool = False,
    shard_workers: int | None = None,
) -> list[Path]:
    """Run Major inference shard-by-shard and return all intermediate output paths."""
    cfg = load_config()
    p = cfg.prediction

    input_paths = _get_major_input_paths(p)
    assert input_paths, "No valid raw parquet files found in predict input source"
    assert p.major_output_dir is not None or p.major_output_path is not None, (
        "major_output_dir or major_output_path must be set in config.toml"
    )
    assert p.major_onnx_dir is not None, "major_onnx_dir must be set in config.toml"

    total_inputs = len(input_paths)
    outputs: list[Path] = []
    pending_jobs: list[tuple[Path, Path]] = []
    for input_path in input_paths:
        output_path = _derive_major_output_path(input_path, p, total_inputs)
        if output_path.exists() and not overwrite:
            logger.info(f"[Major] Skip existing shard output: {output_path}")
            outputs.append(output_path)
            continue
        pending_jobs.append((input_path, output_path))

    requested_shard_workers = shard_workers if shard_workers is not None else p.major_shard_workers
    workers = _effective_parallelism(requested_shard_workers, len(pending_jobs))
    logger.info(
        f"[Major] Effective parallelism | "
        f"cuda_provider={_has_cuda_ort_provider()} | "
        f"shard_workers={workers} | "
        f"major_workers={p.major_workers} | "
        f"pending_shards={len(pending_jobs)}"
    )
    if pending_jobs and workers > 1:
        logger.info(f"[Major] Running {len(pending_jobs)} shards with {workers} processes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_major_single,
                    input_path,
                    output_path,
                    p.major_onnx_dir,
                    p.major_workers,
                    workers,
                    p.batch_size,
                    p.major_max_length,
                    limit_rows,
                ): output_path
                for input_path, output_path in pending_jobs
            }
            for future in as_completed(futures):
                outputs.append(future.result())
    else:
        for input_path, output_path in pending_jobs:
            outputs.append(
                _run_major_single(
                    input_path=input_path,
                    major_output_path=output_path,
                    major_onnx_dir=p.major_onnx_dir,
                    major_workers=p.major_workers,
                    shard_workers=workers,
                    batch_size=p.batch_size,
                    major_max_length=p.major_max_length,
                    limit_rows=limit_rows,
                )
            )
    outputs.sort()
    return outputs


def _run_sub_single(
    intermediate_path: Path,
    output_path: Path,
    sub_onnx_dir: Path,
    sub_max_length: int,
    sub_major_workers: int,
    shard_workers: int,
    limit_rows: int | None = None,
) -> Path:
    """Run sub-category classification on one Major intermediate shard."""
    logger.info(f"[Sub] Loading Major intermediate: {intermediate_path}")
    df = pl.read_parquet(intermediate_path)
    logger.info(f"[Sub] Intermediate parquet loaded: {intermediate_path} | rows={len(df)}")

    if limit_rows is not None:
        df = df.head(limit_rows)
        logger.info(f"[Sub] Applied row limit: {limit_rows}")

    n = len(df)
    logger.info(f"[Sub] {n} rows to classify")

    cpu_count = os.cpu_count() or 1
    intra_threads, inter_threads = _resolve_ort_threads(cpu_count, shard_workers)
    logger.info(f"[Sub] CPU cores={cpu_count}, ORT threads: intra={intra_threads}, inter={inter_threads}")

    logger.info(f"[Sub] Grouping rows by major category")
    safe_to_global_idx, safe_to_texts = _build_sub_groups(df)
    logger.info(f"[Sub] Grouping complete: {len(safe_to_global_idx)} major buckets")

    label_stats = LabelStats.load()
    major_categories = label_stats.get_major_categories()
    major_to_subcats: dict[str, list[str]] = {m: label_stats.get_sub_categories(m) for m in major_categories}
    subcats_lookup = {
        **{k: v for k, v in major_to_subcats.items()},
        **{safe_name(k): v for k, v in major_to_subcats.items()},
    }

    sub_sessions: dict[str, ort.InferenceSession] = {}
    sub_tokenizers: dict[str, AutoTokenizer] = {}
    sub_labels_by_major: dict[str, list[str]] = {}
    backend_by_major: dict[str, str] = {}

    assert sub_onnx_dir is not None, "sub_onnx_dir must be set in config.toml"
    major_model_dirs = _discover_sub_model_dirs(sub_onnx_dir, major_categories)

    present_safes = sorted(safe_to_global_idx.keys())
    logger.info(f"[Sub] Loading models only for majors in shard: {present_safes}")

    for safe in present_safes:
        if safe not in major_model_dirs:
            logger.warning(f"[Sub] Major '{safe}' not found in validated model directory map, will use fallback '其他'")
            continue
        backend_name, major_dir = major_model_dirs[safe]
        onnx_path = major_dir / "best.onnx"
        tokenizer_path = major_dir / "tokenizer"
        label_map_path = major_dir / "label_map.json"

        if not onnx_path.exists() or not tokenizer_path.exists():
            logger.warning(
                f"[Sub/{backend_name}] ONNX/tokenizer not found for '{safe}' in {major_dir}, will use fallback '其他'"
            )
            continue

        logger.info(f"[Sub/{backend_name}] Loading ONNX for '{safe}' from {major_dir}")
        sub_sessions[safe] = _make_ort_session(onnx_path, intra_threads, inter_threads)
        sub_tokenizers[safe] = AutoTokenizer.from_pretrained(str(tokenizer_path))
        sub_labels_by_major[safe] = _load_sub_labels(safe, label_map_path, subcats_lookup)
        backend_by_major[safe] = backend_name

    if not sub_sessions:
        raise RuntimeError(f"No sub ONNX models loaded from sub_onnx_dir={sub_onnx_dir}")
    logger.info(f"[Sub] Loaded {len(sub_sessions)} per-major ONNX models for this shard")

    sub_cats_out = np.full(n, "其他", dtype=object)
    sub_confs_out = np.zeros(n, dtype=np.float64)

    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=_effective_parallelism(sub_major_workers, len(safe_to_global_idx))) as executor:
        futures = {}
        total_groups = len(safe_to_global_idx)
        submitted_groups = 0

        for safe, g_indices in safe_to_global_idx.items():
            if safe not in sub_sessions:
                sub_cats = sub_labels_by_major.get(safe, subcats_lookup.get(safe, ["其他"]))
                for g in g_indices:
                    sub_cats_out[g] = sub_cats[-1]
                    sub_confs_out[g] = 0.0
                logger.info(f"[Sub] Fallback {safe}: {len(g_indices)} rows (no model)")
                continue

            texts_list = safe_to_texts[safe]
            submitted_groups += 1
            backend_name = backend_by_major.get(safe, "unknown")
            logger.info(
                f"[Sub/{backend_name}] Tokenizing group {submitted_groups}/{total_groups}"
                f" | major={safe} | rows={len(g_indices)}"
            )

            def _classify(safe_m: str, g_idx: list[int], txts: list[str]):
                inputs = _tokenize(txts, sub_tokenizers[safe_m], sub_max_length)
                onnx_in = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }
                logits = sub_sessions[safe_m].run(None, onnx_in)[0]
                probs = _softmax(logits)
                pred_idx = probs.argmax(axis=1)
                conf = probs[np.arange(len(pred_idx)), pred_idx]
                sub_cats = sub_labels_by_major.get(safe_m, subcats_lookup.get(safe_m, ["其他"]))
                results = []
                for g, p_idx, c_val in zip(g_idx, pred_idx.tolist(), conf.tolist()):
                    safe_idx = min(p_idx, len(sub_cats) - 1)
                    results.append((g, sub_cats[safe_idx], float(c_val)))
                return safe_m, results

            futures[executor.submit(_classify, safe, g_indices, texts_list)] = safe

        for future in as_completed(futures):
            safe_major, results = future.result()
            for g, sub_cat, conf_val in results:
                sub_cats_out[g] = sub_cat
                sub_confs_out[g] = conf_val
            elapsed = time.monotonic() - t0
            backend_name = backend_by_major.get(safe_major, "unknown")
            logger.info(
                f"[Sub/{backend_name}] Inference complete | major={safe_major}"
                f" | rows={len(results)} | {elapsed:.1f}s total"
            )

    result_df = df.with_columns(
        pl.Series("sub_category", sub_cats_out),
        pl.Series("sub_category_confidence", sub_confs_out),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.write_parquet(output_path)
    logger.success(f"[Sub] Complete output → {output_path}")
    return output_path


def run_sub(
    limit_rows: int | None = None,
    overwrite: bool = False,
    shard_workers: int | None = None,
    sub_major_workers: int | None = None,
) -> list[Path]:
    """Run sub-category classification shard-by-shard and return final output paths."""
    cfg = load_config()
    p = cfg.prediction

    intermediate_paths = _get_sub_input_paths(p)
    assert intermediate_paths, "No sub input parquet files found in configured source"
    assert p.output_dir is not None or p.output_path is not None, "output_dir or output_path must be set in config.toml"
    assert p.sub_onnx_dir is not None, "sub_onnx_dir must be set in config.toml"
    major_categories = LabelStats.load().get_major_categories()
    resolved_model_dirs = _discover_sub_model_dirs(p.sub_onnx_dir, major_categories)
    _log_sub_model_layout(resolved_model_dirs, major_categories)

    total_inputs = len(intermediate_paths)
    outputs: list[Path] = []
    pending_jobs: list[tuple[Path, Path]] = []
    for intermediate_path in intermediate_paths:
        output_path = _derive_sub_output_path(intermediate_path, p, total_inputs)
        if output_path.exists() and not overwrite:
            logger.info(f"[Sub] Skip existing shard output: {output_path}")
            outputs.append(output_path)
            continue
        pending_jobs.append((intermediate_path, output_path))

    requested_sub_major_workers = sub_major_workers if sub_major_workers is not None else p.sub_major_workers
    cuda_provider_enabled = _has_cuda_ort_provider()

    requested_shard_workers = shard_workers if shard_workers is not None else p.sub_shard_workers
    if shard_workers is None and cuda_provider_enabled:
        if requested_shard_workers != 1:
            logger.info(
                f"[Sub] CUDAExecutionProvider detected; "
                f"defaulting sub shard workers from {requested_shard_workers} to 1 for single-GPU efficiency"
            )
        requested_shard_workers = 1
    workers = _effective_parallelism(requested_shard_workers, len(pending_jobs))
    logger.info(
        f"[Sub] Effective parallelism | "
        f"cuda_provider={cuda_provider_enabled} | "
        f"shard_workers={workers} | "
        f"sub_major_workers={requested_sub_major_workers} | "
        f"pending_shards={len(pending_jobs)}"
    )
    if pending_jobs and workers > 1:
        logger.info(f"[Sub] Running {len(pending_jobs)} shards with {workers} processes")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_sub_single,
                    intermediate_path,
                    output_path,
                    p.sub_onnx_dir,
                    p.sub_max_length,
                    requested_sub_major_workers,
                    workers,
                    limit_rows,
                ): output_path
                for intermediate_path, output_path in pending_jobs
            }
            for future in as_completed(futures):
                outputs.append(future.result())
    else:
        for intermediate_path, output_path in pending_jobs:
            outputs.append(
                    _run_sub_single(
                        intermediate_path=intermediate_path,
                        output_path=output_path,
                        sub_onnx_dir=p.sub_onnx_dir,
                        sub_max_length=p.sub_max_length,
                        sub_major_workers=requested_sub_major_workers,
                        shard_workers=workers,
                        limit_rows=limit_rows,
                )
            )
    outputs.sort()
    return outputs


def run(
    limit_rows: int | None = None,
    overwrite: bool = False,
    major_shard_workers: int | None = None,
    sub_shard_workers: int | None = None,
    sub_major_workers: int | None = None,
) -> None:
    """Run full pipeline shard-by-shard: Major → sub-category classification."""
    logger.info("[Predict] === Phase 1: Major ===")
    run_major(limit_rows=limit_rows, overwrite=overwrite, shard_workers=major_shard_workers)

    logger.info("[Predict] === Phase 2: Sub-category ===")
    run_sub(
        limit_rows=limit_rows,
        overwrite=overwrite,
        shard_workers=sub_shard_workers,
        sub_major_workers=sub_major_workers,
    )

    logger.info("[Predict] === All done ===")
