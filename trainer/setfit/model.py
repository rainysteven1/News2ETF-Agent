"""SetFit sub-category classifier — one SetFit model per major category."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from setfit import SetFitModel


def export_setfit_to_onnx(
    model_dir: Path,
    onnx_path: Path,
    max_seq_length: int = 256,
    opset_version: int = 14,
) -> None:
    """Export a SetFit model to ONNX format.

    The exported model takes tokenized text input (input_ids, attention_mask)
    and outputs logits for each sub-category.

    Falls back to exporting the sentence-encoder embeddings only if the
    full pipeline export fails.
    """
    try:
        model = SetFitModel.from_pretrained(str(model_dir))
        model.eval()

        # Use sentence-transformers' built-in ONNX export if available
        st_model = model.model
        if hasattr(st_model, "export_to_onnx"):
            st_model.export_to_onnx(str(onnx_path), opset_version=opset_version)
            logger.info(f"[ONNX] Exported SetFit sentence-encoder to {onnx_path}")
            return

        # Fallback: export the model_body (transformer) with the classifier
        # This wraps the full SetFit inference: text -> embeddings -> logits
        from transformers import AutoTokenizer

        model_body_name = getattr(model.model_body, "name_or_path", None) or getattr(model.model_body, "config", None)
        if model_body_name is None:
            raise RuntimeError("Cannot determine model_body name for tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(model_body_name)
        dummy_text = ["dummy input for ONNX export"]
        inputs = tokenizer(
            dummy_text,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        # Build a wrapper that reproduces SetFitModel.predict() behavior
        class SetFitOnnxWrapper(torch.nn.Module):
            def __init__(self, setfit_model):
                super().__init__()
                self.model_body = setfit_model.model_body
                self.model_head = setfit_model.model_head

            def forward(self, input_ids, attention_mask):
                features = self.model_body({"input_ids": input_ids, "attention_mask": attention_mask})
                embeddings = features["sentence_embedding"]
                logits = self.model_head(embeddings)
                return logits

        wrapper = SetFitOnnxWrapper(model)

        torch.onnx.export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
        )
        logger.info(f"[ONNX] Exported SetFit full pipeline to {onnx_path}")

    except Exception as exc:
        logger.error(f"[ONNX] Export failed ({exc}), skipping ONNX export for {model_dir}")
        raise  # Re-raise so caller knows it failed


class LabelStats:
    """Load and access label_stats.json via config.toml key (singleton)."""

    _instance: LabelStats | None = None
    _initialized: bool = False

    def __new__(cls, stats_path: Path | None = None) -> LabelStats:
        if cls._instance is None:
            instance = super().__new__(cls)
            cls._instance = instance
        return cls._instance

    def __init__(self, stats_path: Path | None = None):
        if LabelStats._initialized:
            return
        if stats_path is None:
            from trainer.config import load_config as _load_trainer_config

            cfg = _load_trainer_config()
            stats_path = cfg.setfit.label_stats
        with open(stats_path, encoding="utf-8") as f:
            self._stats: dict[str, Any] = json.load(f)
        LabelStats._initialized = True

    def get_major_categories(self) -> list[str]:
        """Return sorted list of major categories."""
        return sorted(self._stats["major_category"].keys())

    def get_sub_categories(self, major: str) -> list[str]:
        """Return sorted list of sub-categories for a major."""
        return sorted(self._stats["sub_category_by_major"][major].keys())


class SetFitSubCategoryClassifier:
    """Holds one SetFit model per major category for sub-category classification."""

    def __init__(self):
        self.models: dict[str, SetFitModel] = {}

    def set_model(self, major: str, model: SetFitModel) -> None:
        self.models[major] = model

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Save all models to output_dir."""
        output_dir = Path(output_dir)
        paths: dict[str, Path] = {}
        for major, model in self.models.items():
            major_dir = output_dir / _safe_name(major)
            model.save_pretrained(str(major_dir))
            paths[major] = major_dir
        return paths

    @classmethod
    def load(cls, input_dir: Path) -> SetFitSubCategoryClassifier:
        """Load all models from input_dir."""
        clf = cls()
        for major in LabelStats().get_major_categories():
            major_dir = Path(input_dir) / _safe_name(major)
            if not major_dir.exists():
                continue
            clf.models[major] = SetFitModel.from_pretrained(str(major_dir))
        return clf


def _safe_name(name: str) -> str:
    """Make a string safe for use as a directory name."""
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")
