"""Sub (L2) dataset — SubCatDataset for supervised + SetFitDatasetPreparer for contrastive."""

from __future__ import annotations

import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from trainer.src.config import get_config, safe_name

# ─── SubCatDataset (supervised fine-tune) ─────────────────────────────────────


class SubCatDataset(Dataset):
    """Tokenized dataset for sub-category supervised classification."""

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        label_to_idx: dict[str, int],
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = label_to_idx

        df = pl.read_parquet(parquet_path)

        self.titles = df["title"].to_list()

        raw_labels = df["sub_category"].to_list()
        self.labels = [self.label_to_idx[str(v)] for v in raw_labels]

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.titles[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids))
        if token_type_ids.dim() > 1:
            token_type_ids = token_type_ids.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for focal loss."""
        counts = Counter(self.labels)
        n_classes = len(self.label_to_idx)
        total = len(self.labels)
        weights = torch.zeros(n_classes)
        for cls_idx in range(n_classes):
            count = counts.get(cls_idx, 1)
            weights[cls_idx] = total / (n_classes * count)
        return weights


def preprocess_split(
    raw_path: Path,
    major: str,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Split raw data for a specific major into train/val parquet files."""
    data_dir = raw_path.parent / f"subcat_{major.replace('/', '_')}"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    if train_path.exists() and val_path.exists():
        return train_path, val_path

    df = pl.read_parquet(raw_path).filter(pl.col("major_category") == major)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["sub_category"],
        random_state=seed,
    )
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)
    return train_path, val_path


# ─── SetFitDatasetPreparer (contrastive learning) ──────────────────────────────


class SetFitDatasetPreparer:
    """Prepare per-major-category parquet caches for SetFit contrastive training.

    The output for each major is saved to
    ``{raw_data_dir}/{safe_name(major)}/``
    with columns ``text``, ``label_text``, ``major_category``.
    A ``meta.json`` in the same folder records sampling mode and config.
    """

    def __init__(self) -> None:
        self.cfg = get_config().sub.setfit
        self.dcfg = self.cfg.data
        self._embed_model = None
        self._embed_lock = threading.Lock()

    def _get_embed_model(self):
        """Lazy-load SentenceTransformer once (thread-safe)."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            with self._embed_lock:
                if self._embed_model is None:
                    self._embed_model = SentenceTransformer(self.cfg.model.pretrained_model)
        return self._embed_model

    def prepare_all(self, majors: list[str] | None = None) -> None:
        """Prepare datasets for all (or a subset of) major categories."""
        from trainer.src.config import LabelStats

        label_stats = LabelStats.load()
        all_majors = label_stats.get_major_categories()
        target_majors = majors if majors is not None else all_majors

        from trainer.src.utils import get_logger

        logger = get_logger()
        logger.info(f"[SetFit] Preparing {len(target_majors)} major categories: {target_majors}")

        workers = max(1, self.dcfg.prepare_max_workers)
        if workers == 1:
            for major in target_majors:
                self._prepare_one(major)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(self._prepare_one, target_majors))

        logger.info("[SetFit] Dataset preparation complete.")

    def _prepare_one(self, major: str) -> None:
        """Prepare dataset for a single major category."""
        from trainer.src.utils import get_logger

        logger = get_logger()
        assert self.dcfg.raw_data_dir is not None, "setfit.data.raw_data_dir must be set"

        use_cluster = major in self.dcfg.cluster_sampling_majors
        cache_dir = self.dcfg.raw_data_dir / safe_name(major)
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / "prepared.parquet"

        logger.info(f"[SetFit] Preparing '{major}' -> {cache_dir} (cluster={use_cluster})")

        df = self._load_raw(major)

        if df.is_empty():
            logger.warning(f"[SetFit] No data found for major '{major}', skipping.")
            return

        if use_cluster:
            df_sampled = self._cluster_sample(df)
            meta = {
                "major": major,
                "cluster": True,
                "samples": len(df_sampled),
                "n_cap": self.dcfg.cluster.n_cap,
                "n_clusters": self.dcfg.cluster.n_clusters,
                "samples_per_cluster": self.dcfg.cluster.samples_per_cluster,
                "min_samples_per_class": self.dcfg.cluster.min_samples_per_class,
            }
        else:
            df_sampled = self._random_sample(df)
            meta = {
                "major": major,
                "cluster": False,
                "samples": len(df_sampled),
                "samples_per_class": self.dcfg.random.samples_per_class,
                "min_samples_per_class": self.dcfg.random.min_samples_per_class,
            }

        df_sampled.write_parquet(out_path)

        with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[SetFit] '{major}' prepared: {len(df_sampled)} samples -> {out_path}")

    def _load_raw(self, major: str) -> pl.DataFrame:
        """Load rows for a specific major category from raw.parquet."""
        assert self.dcfg.raw_data_dir is not None, "setfit.data.raw_data_dir must be set"

        raw_path = self.dcfg.raw_data_dir / "raw.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"[SetFit] Raw parquet not found: {raw_path}")

        df = pl.read_parquet(raw_path)
        df_major = df.filter(pl.col("major_category") == major)
        return (
            df_major.with_columns((pl.col("title") + " " + pl.col("content")).alias("text"))
            .select(["text", "sub_category", "major_category"])
            .rename({"sub_category": "label_text"})
        )

    def _random_sample(self, df: pl.DataFrame) -> pl.DataFrame:
        """Random sample per sub-category, capped at samples_per_class with min_samples_per_class floor."""
        rcfg = self.dcfg.random
        sampled_parts: list[pl.DataFrame] = []
        for label in df["label_text"].unique():
            label_df = df.filter(pl.col("label_text") == label)
            n_available = len(label_df)
            n_to_keep = min(rcfg.samples_per_class, max(rcfg.min_samples_per_class, n_available))
            sampled_parts.append(label_df.sample(n=min(n_to_keep, n_available), shuffle=True))
        return pl.concat(sampled_parts)

    def _hard_negative_boost(self, df_sampled: pl.DataFrame, df_pool: pl.DataFrame) -> pl.DataFrame:
        """Boost samples for confused class pairs by mining hard negatives."""
        ccfg = self.dcfg.cluster
        boost_factor = getattr(ccfg, "hard_negative_boost", 3)
        confused_pairs = getattr(ccfg, "confused_pairs", [])

        if not confused_pairs or boost_factor <= 0:
            return df_sampled

        from trainer.src.utils import get_logger

        logger = get_logger()
        texts_pool = df_pool["text"].to_list()
        self._get_embed_model().encode(texts_pool, show_progress_bar=False, batch_size=256)

        boosted_parts = [df_sampled.clone()]

        for pair in confused_pairs:
            cls_a, cls_b = pair[0], pair[1]
            df_a = df_pool.filter(pl.col("label_text") == cls_a)
            df_b = df_pool.filter(pl.col("label_text") == cls_b)
            if df_a.is_empty() or df_b.is_empty():
                continue

            texts_a = df_a["text"].to_list()
            texts_b = df_b["text"].to_list()
            emb_a = self._get_embed_model().encode(texts_a, show_progress_bar=False, batch_size=256)
            emb_b = self._get_embed_model().encode(texts_b, show_progress_bar=False, batch_size=256)

            from sklearn.metrics.pairwise import cosine_similarity

            sim_ab = cosine_similarity(emb_a, emb_b)
            for i in range(len(texts_a)):
                top_b_idx = int(sim_ab[i].argmax())
                candidate = df_b.filter(pl.col("text") == texts_b[top_b_idx]).drop("cluster")
                if not candidate.is_empty():
                    n_add = min(boost_factor, len(candidate))
                    boosted_parts.append(candidate.sample(n=n_add))

            for j in range(len(texts_b)):
                top_a_idx = int(sim_ab[:, j].argmax())
                candidate = df_a.filter(pl.col("text") == texts_a[top_a_idx]).drop("cluster")
                if not candidate.is_empty():
                    n_add = min(boost_factor, len(candidate))
                    boosted_parts.append(candidate.sample(n=n_add))

        result = pl.concat(boosted_parts).unique(subset=["text"])
        logger.info(f"[SetFit] Hard negative boost: {len(df_sampled)} -> {len(result)} samples")
        return result

    def _cluster_sample(self, df: pl.DataFrame) -> pl.DataFrame:
        """Major-level joint clustering with label-balance + hard-negative + global-coverage."""
        ccfg = self.dcfg.cluster
        n_cap = ccfg.n_cap
        n_per_cluster = ccfg.samples_per_cluster
        n_min = ccfg.min_samples_per_class

        capped_parts: list[pl.DataFrame] = []
        for label in df["label_text"].unique():
            label_df = df.filter(pl.col("label_text") == label)
            if len(label_df) > n_cap:
                label_df = label_df.sample(n=n_cap, shuffle=True)
            capped_parts.append(label_df)
        df_capped = pl.concat(capped_parts)

        texts = df_capped["text"].to_list()
        if len(texts) < 2:
            return df_capped

        vec = self._get_embed_model().encode(texts, show_progress_bar=False, batch_size=256)

        n_clusters = max(50, min(ccfg.n_clusters, len(texts) // 10))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.cfg.seed, batch_size=512)
        clusters = km.fit_predict(vec)

        df_capped = df_capped.with_columns(pl.Series("cluster", clusters))

        global_counts: dict[str, int] = {label: 0 for label in df["label_text"].unique()}

        sampled_parts: list[pl.DataFrame] = []
        for cluster_id in sorted(df_capped["cluster"].unique()):
            cluster_df = df_capped.filter(pl.col("cluster") == cluster_id)
            unique_labels = cluster_df["label_text"].unique()

            if len(unique_labels) == 1:
                label = unique_labels[0]
                n_global = global_counts.get(label, 0)
                n_remaining = max(0, n_min - n_global)
                n_keep = min(n_per_cluster, len(cluster_df), n_remaining)
                if n_keep > 0:
                    sampled_parts.append(cluster_df.sample(n=n_keep, shuffle=True))
                    global_counts[label] = global_counts.get(label, 0) + n_keep
            else:
                for label in unique_labels:
                    label_df = cluster_df.filter(pl.col("label_text") == label)
                    n_global = global_counts.get(label, 0)
                    n_remaining = max(0, n_min - n_global)
                    n_keep = min(n_per_cluster, len(label_df), n_remaining)
                    if n_keep > 0:
                        sampled_parts.append(label_df.sample(n=n_keep, shuffle=True))
                        global_counts[label] = global_counts.get(label, 0) + n_keep

        df_sampled = pl.concat(sampled_parts).drop("cluster") if sampled_parts else pl.DataFrame()

        filled_parts: list[pl.DataFrame] = []
        if not df_sampled.is_empty():
            filled_parts.append(df_sampled)

        for label in df["label_text"].unique():
            label_df = df_sampled.filter(pl.col("label_text") == label) if not df_sampled.is_empty() else pl.DataFrame()
            deficit = n_min - len(label_df)
            if deficit > 0:
                already_texts = set(label_df["text"].to_list()) if not df_sampled.is_empty() else set()
                candidates = df_capped.filter((pl.col("label_text") == label) & (~pl.col("text").is_in(already_texts)))
                if len(candidates) > 0:
                    n_extra = min(deficit, len(candidates))
                    filled_parts.append(candidates.sample(n=n_extra, shuffle=True))

        df_result = pl.concat(filled_parts).unique(subset=["text"]) if filled_parts else df_capped
        return self._hard_negative_boost(df_result, df_capped)
