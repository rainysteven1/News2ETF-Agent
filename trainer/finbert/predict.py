"""Batch inference for FinBERT classifier.

Loads a trained checkpoint and runs classification on unlabeled parquet files.
Outputs: major_category, sentiment, l1_confidence, sentiment_confidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from trainer.finbert.dataset import IDX_TO_L1, NewsInferenceDataset, SENTIMENT_LABELS
from trainer.finbert.model import FinBERTClassifier


def predict(
    checkpoint_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    batch_size: int = 256,
    max_length: int = 128,
    use_content: bool = False,
) -> pl.DataFrame:
    """Run FinBERT classification on an unlabeled parquet file.

    Returns a Polars DataFrame with added columns:
    major_category, sentiment, l1_confidence, sentiment_confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[FinBERT Predict] Device: {device}")

    checkpoint_dir = Path(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = FinBERTClassifier.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    dataset = NewsInferenceDataset(
        input_path,
        tokenizer,
        max_length=max_length,
        use_content=use_content,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_l1_preds: list[int] = []
    all_l1_confs: list[float] = []
    all_sent_preds: list[int] = []
    all_sent_confs: list[float] = []

    logger.info(f"[FinBERT Predict] Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )

            l1_probs = torch.softmax(outputs["l1_logits"], dim=-1)
            l1_conf, l1_pred = l1_probs.max(dim=-1)

            sent_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
            sent_conf, sent_pred = sent_probs.max(dim=-1)

            all_l1_preds.extend(l1_pred.cpu().tolist())
            all_l1_confs.extend(l1_conf.cpu().tolist())
            all_sent_preds.extend(sent_pred.cpu().tolist())
            all_sent_confs.extend(sent_conf.cpu().tolist())

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {(i + 1) * batch_size}/{len(dataset)} samples")

    idx_to_l1 = {int(k): v for k, v in IDX_TO_L1.items()}
    df = dataset.meta.with_columns(
        pl.Series("major_category", [idx_to_l1[i] for i in all_l1_preds]),
        pl.Series("sentiment", [SENTIMENT_LABELS[i] for i in all_sent_preds]),
        pl.Series("l1_confidence", np.round(all_l1_confs, 4).tolist()),
        pl.Series("sentiment_confidence", np.round(all_sent_confs, 4).tolist()),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.success(f"[FinBERT Predict] Results saved → {output_path} ({len(df)} rows)")

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def predict_cmd(
        checkpoint: str = typer.Option(..., help="Path to FinBERT checkpoint directory"),
        input: str = typer.Option(..., help="Path to input parquet file"),
        output: str = typer.Option(..., help="Path to output parquet file"),
        batch_size: int = 256,
        max_length: int = 128,
        use_content: bool = False,
    ) -> None:
        """Run FinBERT inference on a parquet file."""
        predict(
            checkpoint_dir=checkpoint,
            input_path=input,
            output_path=output,
            batch_size=batch_size,
            max_length=max_length,
            use_content=use_content,
        )

    app()
