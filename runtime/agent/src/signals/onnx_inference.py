"""FinBERT + SetFit ONNX inference pipeline with LRU cache.

Public API:
    get_onnx_predictions(week_start, raw_news_df, config) -> pl.DataFrame
        Returns predictions with columns: major_category, sentiment, l1_confidence,
        sentiment_confidence, sub_category, sub_category_confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from src.logger import logger
from src.signals.label_stats import LabelStats, safe_name

if TYPE_CHECKING:
    from src.config import AgentRootConfig


L1_CATEGORIES: list[str] = [
    "主题策略",
    "医药健康",
    "基础设施/公共",
    "消费文娱",
    "科技信息",
    "资源材料",
    "金融地产",
    "高端制造",
]

SENTIMENT_LABELS: list[str] = ["negative", "neutral", "positive"]


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "onnxruntime is required only when ONNX cache is missing and live inference is needed."
        ) from exc
    return ort


def _import_auto_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required only when ONNX cache is missing and live inference is needed."
        ) from exc
    return AutoTokenizer


# ─── ONNX Helpers ───────────────────────────────────────────────────────────────


def _make_ort_session(onnx_path: Path) -> Any:
    ort = _import_onnxruntime()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 4
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _tokenize(texts: list[str], tokenizer: Any, max_length: int) -> dict[str, np.ndarray]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}


# ─── Inference Pipeline ────────────────────────────────────────────────────────


class ONNXInferencePipeline:
    """FinBERT + SetFit ONNX inference. Lazy-loaded singleton per process."""

    _instance: "ONNXInferencePipeline | None" = None

    def __init__(self, config: AgentRootConfig) -> None:
        self.finbert_onnx_dir = Path(config.predict.finbert_onnx_dir)
        self.finbert_max_length = config.predict.finbert_max_length
        self.setfit_base_dir = Path(config.predict.setfit_base_dir)
        self.setfit_max_length = config.predict.setfit_max_length

        self._finbert_sess: Any | None = None
        self._finbert_tokenizer: Any | None = None
        self._setfit_sessions: dict[str, Any] = {}
        self._setfit_tokenizers: dict[str, Any] = {}
        self._subcats_lookup: dict[str, list[str]] = {}

    @classmethod
    def get_instance(cls, config: AgentRootConfig) -> "ONNXInferencePipeline":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def _ensure_finbert(self) -> None:
        if self._finbert_sess is None:
            AutoTokenizer = _import_auto_tokenizer()
            self._finbert_sess = _make_ort_session(self.finbert_onnx_dir / "best.onnx")
            self._finbert_tokenizer = AutoTokenizer.from_pretrained(str(self.finbert_onnx_dir / "tokenizer"))
            logger.info("[ONNX] FinBERT session loaded")

    def _ensure_setfit(self) -> None:
        if self._setfit_sessions:
            return

        label_stats = LabelStats.load()
        majors = label_stats.get_major_categories()
        self._subcats_lookup = {
            **{m: label_stats.get_sub_categories(m) for m in majors},
            **{safe_name(k): v for k, v in {m: label_stats.get_sub_categories(m) for m in majors}.items()},
        }

        for m in majors:
            safe = safe_name(m)
            onnx_path = self.setfit_base_dir / safe / "best.onnx"
            tok_path = self.setfit_base_dir / safe / "tokenizer"
            if onnx_path.exists() and tok_path.exists():
                AutoTokenizer = _import_auto_tokenizer()
                self._setfit_sessions[safe] = _make_ort_session(onnx_path)
                self._setfit_tokenizers[safe] = AutoTokenizer.from_pretrained(str(tok_path))

        logger.info(f"[ONNX] SetFit sessions loaded: {list(self._setfit_sessions.keys())}")

    def predict(self, texts: list[str]) -> pl.DataFrame:
        """Run FinBERT + SetFit ONNX on a list of texts.

        Returns a DataFrame with columns:
            major_category, sentiment, l1_confidence, sentiment_confidence,
            sub_category, sub_category_confidence
        """
        self._ensure_finbert()
        self._ensure_setfit()

        # ── Phase 1: FinBERT ─────────────────────────────────────────────────
        inputs = _tokenize(texts, self._finbert_tokenizer, self.finbert_max_length)
        onnx_in = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": np.zeros_like(inputs["input_ids"]).astype(np.int64),
        }
        l1_logits, sent_logits = self._finbert_sess.run(None, onnx_in)

        l1_probs = _softmax(l1_logits)
        sent_probs = _softmax(sent_logits)
        l1_pred = l1_probs.argmax(axis=1)
        sent_pred = sent_probs.argmax(axis=1)

        majors = [L1_CATEGORIES[i] for i in l1_pred]
        sents = [SENTIMENT_LABELS[i] for i in sent_pred]
        l1_confs = l1_probs[np.arange(len(l1_pred)), l1_pred]
        sent_confs = sent_probs[np.arange(len(sent_pred)), sent_pred]

        # ── Phase 2: SetFit per major category ─────────────────────────────────
        sub_cats = ["其他"] * len(texts)
        sub_confs = np.zeros(len(texts), dtype=np.float64)

        safe_to_texts: dict[str, list[str]] = {}
        safe_to_idx: dict[str, list[int]] = {}
        for i, m in enumerate(majors):
            safe = safe_name(m)
            safe_to_texts.setdefault(safe, []).append(texts[i])
            safe_to_idx.setdefault(safe, []).append(i)

        for safe, txts in safe_to_texts.items():
            if safe not in self._setfit_sessions:
                continue
            inputs = _tokenize(txts, self._setfit_tokenizers[safe], self.setfit_max_length)
            onnx_in = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }
            logits = self._setfit_sessions[safe].run(None, onnx_in)[0]
            probs = _softmax(logits)
            pred_idx = probs.argmax(axis=1).tolist()
            confs = probs[np.arange(len(pred_idx)), pred_idx]
            subcats = self._subcats_lookup.get(safe, ["其他"])
            for g, p, c in zip(safe_to_idx[safe], pred_idx, confs):
                sub_cats[g] = subcats[min(p, len(subcats) - 1)]
                sub_confs[g] = float(c)

        return pl.DataFrame(
            {
                "major_category": majors,
                "sentiment": sents,
                "l1_confidence": l1_confs,
                "sentiment_confidence": sent_confs,
                "sub_category": sub_cats,
                "sub_category_confidence": sub_confs,
            }
        )


# ─── LRU Cache ─────────────────────────────────────────────────────────────────


class ONNXCacheManager:
    """LRU rolling cache: memory dict + parquet persistence.

    - parquet path: {onnx_cache_dir}/{week_start_str}.parquet
    - max_cache_weeks: evict oldest when exceeded (default 4)
    """

    def __init__(self, config: AgentRootConfig) -> None:
        self.cache_dir = Path(config.predict.onnx_cache_dir)
        self.max_cache_weeks = config.predict.max_cache_weeks
        self._memory: dict[str, pl.DataFrame] = {}
        self._order: list[str] = []  # insertion order for LRU
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _evict_if_needed(self) -> None:
        while len(self._memory) >= self.max_cache_weeks and self._order:
            oldest = self._order.pop(0)
            self._memory.pop(oldest, None)
            parquet_path = self.cache_dir / f"{oldest}.parquet"
            if parquet_path.exists():
                parquet_path.unlink()
            logger.info(f"[ONNXCache] Evicted oldest: {oldest}")

    def get(self, week_start: str) -> pl.DataFrame | None:
        return self._memory.get(week_start)

    def put(self, week_start: str, df: pl.DataFrame) -> None:
        if week_start in self._memory:
            return
        self._evict_if_needed()
        self._memory[week_start] = df
        self._order.append(week_start)
        df.write_parquet(self.cache_dir / f"{week_start}.parquet")
        logger.info(f"[ONNXCache] Cached week {week_start} ({len(df)} rows)")

    @staticmethod
    def load_from_disk(cache_dir: Path, week_start: str) -> pl.DataFrame | None:
        path = cache_dir / f"{week_start}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None


# ─── Public API ────────────────────────────────────────────────────────────────


def get_onnx_predictions(
    week_start: str,
    raw_news_df: pl.DataFrame,
    config: AgentRootConfig,
) -> pl.DataFrame:
    """Get ONNX predictions for a given week (with LRU cache).

    Returns a DataFrame with columns:
        datetime, title, source, major_category, sentiment,
        l1_confidence, sentiment_confidence, sub_category, sub_category_confidence

    Cache lookup order: memory → disk parquet → inference → store
    """
    cache = ONNXCacheManager(config)

    # 1. Memory cache hit
    cached = cache.get(week_start)
    if cached is not None:
        logger.info(f"[ONNXCache] HIT memory: {week_start}")
        return cached

    # 2. Disk parquet hit
    disk_df = ONNXCacheManager.load_from_disk(config.predict.onnx_cache_dir, week_start)
    if disk_df is not None:
        logger.info(f"[ONNXCache] HIT disk: {week_start}")
        cache.put(week_start, disk_df)
        return disk_df

    # 3. Inference miss
    logger.info(f"[ONNXCache] MISS: running inference for {week_start}")
    pipeline = ONNXInferencePipeline.get_instance(config)

    texts: list[str] = []
    for row in raw_news_df.iter_rows(named=True):
        t = row.get("title") or ""
        c = row.get("content") or ""
        if t and c:
            texts.append(f"{t} [SEP] {c[:256]}")
        elif c:
            texts.append(c[:256])
        else:
            texts.append(t)

    results = pipeline.predict(texts)

    # Preserve original news info
    out = raw_news_df.select(["datetime", "title", "source"]).with_columns(
        [
            results["major_category"],
            results["sentiment"],
            results["l1_confidence"],
            results["sentiment_confidence"],
            results["sub_category"],
            results["sub_category_confidence"],
        ]
    )

    cache.put(week_start, out)
    return out
