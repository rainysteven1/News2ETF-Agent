"""SetFit sub-category classifier — one SetFit model per major category."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from setfit import SetFitModel

from trainer.config import LabelStats


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
        # 1. Load model
        model = SetFitModel.from_pretrained(model_dir)

        # Force all modules to CPU to avoid CUDA/CPU mismatch during ONNX export
        model.to(torch.device("cpu"))
        if model.model_body is not None:
            model.model_body.to(torch.device("cpu"))
        if model.model_head is not None and isinstance(model.model_head, torch.nn.Module):
            model.model_head.to(torch.device("cpu"))

        # 2. Extract sub-modules (st_body = Transformer + Pooling)
        st_body = model.model_body
        assert st_body is not None, "SetFit model must have model_body"
        st_body.eval()
        st_body.to(torch.device("cpu"))

        # Process head: ensure it is a PyTorch Linear layer
        assert model.model_head is not None, "SetFit model must have model_head"
        if not isinstance(model.model_head, torch.nn.Module):
            weights = torch.from_numpy(model.model_head.coef_).float()
            bias = torch.from_numpy(model.model_head.intercept_).float()
            head_module = torch.nn.Linear(weights.shape[1], weights.shape[0])
            with torch.no_grad():
                head_module.weight.copy_(weights)
                head_module.bias.copy_(bias)
        else:
            head_module = model.model_head

        head_module.eval()
        head_module.to(torch.device("cpu"))

        # 3. Wrap the full inference pipeline (replicate SetFit source logic)
        class PredictWrapper(torch.nn.Module):
            def __init__(self, body, head, normalize):
                super().__init__()
                self.transformer = body[0]  # Transformer module
                self.pooler = body[1]  # Pooling module
                self.head = head
                self.normalize = normalize  # matches self.normalize_embeddings in SetFit source

            def forward(self, input_ids, attention_mask):
                # A. Encode
                out = self.transformer({"input_ids": input_ids, "attention_mask": attention_mask})
                out = self.pooler(out)
                embeddings = out["sentence_embedding"]

                # B. Normalize (SetFit encode may do this)
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # C. Classify
                logits = self.head(embeddings)
                return logits

        wrapper = PredictWrapper(st_body, head_module, model.normalize_embeddings)

        # 4. Export
        # Prepare dummy input using the model's tokenizer
        from transformers import AutoTokenizer

        # Auto-detect the underlying Transformer path
        tokenizer_name = st_body[0].auto_model.config._name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        dummy_text = ["dummy input for ONNX export"]
        inputs = tokenizer(
            dummy_text,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset_version,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
        )

        tokenizer_dir = onnx_path.parent / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(tokenizer_dir)  # Save tokenizer alongside for later loading

        logger.info(f"[ONNX] Exported SetFit full pipeline to {onnx_path}")

    except Exception as exc:
        logger.error(f"[ONNX] Export failed ({exc}), skipping ONNX export for {model_dir}")
        raise  # Re-raise so caller knows it failed


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
