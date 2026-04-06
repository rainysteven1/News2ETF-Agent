"""Major (L1) dataset — news classification for major category + sentiment."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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

L1_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(L1_CATEGORIES)}
IDX_TO_L1: dict[int, str] = {i: name for i, name in enumerate(L1_CATEGORIES)}

SENTIMENT_STR_TO_INT: dict[str | int, int] = {
    "利空": 0,
    "中性": 1,
    "利好": 2,
}

SENTIMENT_LABELS: list[str] = ["negative", "neutral", "positive"]


def preprocess_split(
    raw_path: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Split raw labeled data into train/val parquet files using stratified sampling."""
    data_dir = raw_path.parent
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    if train_path.exists() and val_path.exists():
        return

    df = pl.read_parquet(raw_path)
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["major_category"],
        random_state=seed,
    )
    train_df.write_parquet(train_path)
    val_df.write_parquet(val_path)


class NewsClassificationDataset(Dataset):
    """Tokenized news dataset for major category + sentiment classification."""

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        use_content: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_content = use_content

        df = pl.read_parquet(parquet_path)
        self._validate(df)

        self.titles = df["title"].to_list()
        self.contents = df["content"].to_list() if use_content and "content" in df.columns else None

        raw_l1 = df["major_category"].to_list()
        self.l1_labels = [L1_TO_IDX[str(v)] for v in raw_l1]

        raw_sentiment = df["sentiment"].to_list()
        self.sentiment_labels = [SENTIMENT_STR_TO_INT[s] if isinstance(s, str) else int(s) for s in raw_sentiment]

    @staticmethod
    def _validate(df: pl.DataFrame) -> None:
        required = {"title", "major_category", "sentiment"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Parquet file missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.titles[idx]
        if self.contents is not None:
            content = self.contents[idx]
            if content is not None and content:
                text = f"{text}[SEP]{str(content)[:256]}"

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
            "l1_label": torch.tensor(self.l1_labels[idx], dtype=torch.long),
            "sentiment_label": torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }
