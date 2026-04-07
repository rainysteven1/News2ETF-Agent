"""Sub (L2) classifier — BERT backbone + single classification head with focal loss."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class SubFocalLoss(nn.Module):
    """Focal loss for multi-class classification with class imbalance."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class SubClassifierConfig(PretrainedConfig):
    model_type = "sub_classifier"

    def __init__(
        self,
        num_classes: int = 4,
        classifier_dropout: float = 0.1,
        focal_gamma: float = 2.0,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        position_embedding_type: str = "absolute",
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            **kwargs,
        )
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.focal_gamma = focal_gamma


class SubClassifier(BertPreTrainedModel):
    """BERT backbone + single sub-category classification head."""

    config_class = SubClassifierConfig

    def __init__(self, config: SubClassifierConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        hidden = config.hidden_size
        drop = config.classifier_dropout

        drop = drop if drop is not None else 0.1
        self.dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.activation = nn.GELU()
        self.fc1_dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden // 2, config.num_classes)

        self.focal_gamma = config.focal_gamma
        self._class_weights: torch.Tensor | None = None

        self.apply(self._init_weights)

    def set_class_weights(self, weights: torch.Tensor) -> None:
        self._class_weights = weights.to(self.device)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        hidden = self.activation(self.fc1(pooled))
        hidden = self.fc1_dropout(hidden)
        logits = self.fc2(hidden)

        result: dict[str, torch.Tensor] = {"logits": logits}

        if label is not None:
            w = self._class_weights.to(logits.device) if self._class_weights is not None else None
            loss_fn = SubFocalLoss(gamma=self.focal_gamma, weight=w)
            result["loss"] = loss_fn(logits, label)

        return result


def load_sub_classifier(
    pretrained_model: str,
    num_classes: int,
    dropout: float = 0.1,
    focal_gamma: float = 2.0,
) -> SubClassifier:
    config = SubClassifierConfig.from_pretrained(
        pretrained_model,
        num_classes=num_classes,
        classifier_dropout=dropout,
        focal_gamma=focal_gamma,
    )
    model = SubClassifier.from_pretrained(
        pretrained_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model  # type: ignore


def save_sub_classifier(model: SubClassifier, output_dir: Path) -> None:
    """Save checkpoint without calling `save_pretrained()`.

    `transformers.save_pretrained()` may import DeepSpeed via accelerate when
    unwrapping the model, which breaks in some environments with numpy>=2.
    For this model, writing `config.json` and `pytorch_model.bin` is sufficient
    for `SubClassifier.from_pretrained(...)` to load it back.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.config.save_pretrained(output_dir)
    state_dict = {k: v.detach().cpu() for k, v in model_to_save.state_dict().items()}
    torch.save(state_dict, output_dir / "pytorch_model.bin")


def export_sub_to_onnx(
    model_dir: Path,
    onnx_path: Path,
    max_seq_length: int = 128,
    opset_version: int = 14,
) -> None:
    """Export Sub model to ONNX. Interface: input_ids + attention_mask → logits."""
    from trainer.src.utils import get_logger

    logger = get_logger()

    try:
        model = SubClassifier.from_pretrained(str(model_dir))
        model.to("cpu")
        model.eval()

        class OnnxWrapper(torch.nn.Module):
            def __init__(self, m: SubClassifier):
                super().__init__()
                self.bert = m.bert
                self.dropout = m.dropout
                self.fc1 = m.fc1
                self.activation = m.activation
                self.fc1_dropout = m.fc1_dropout
                self.fc2 = m.fc2

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                pooled = sum_embeddings / sum_mask
                pooled = self.dropout(pooled)
                hidden = self.activation(self.fc1(pooled))
                hidden = self.fc1_dropout(hidden)
                return self.fc2(hidden)

        wrapper = OnnxWrapper(model)

        dummy_input_ids = torch.ones(1, max_seq_length, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)

        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
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

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        tokenizer_dir = onnx_path.parent / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(tokenizer_dir)

        label_map_src = Path(model_dir) / "label_map.json"
        if label_map_src.exists():
            import shutil

            shutil.copy(label_map_src, onnx_path.parent / "label_map.json")

        logger.info(f"[ONNX] Exported Sub to {onnx_path}")

    except Exception as exc:
        logger.warning(f"[ONNX] Export failed ({exc})")
        raise
