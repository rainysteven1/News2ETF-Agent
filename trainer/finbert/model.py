"""Simplified FinBERT for 8 large categories + 3 sentiment classification.

Architecture:
  - BERT backbone (shared)
  - L1 head: mean pooling → 8 classes (major industry category)
  - Sentiment head: CLS pooled → 3 classes (negative/neutral/positive)

Loss: L = α * CE(L1) + γ * CE(Sentiment)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over non-padded tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class FinBERTClassifierConfig(PretrainedConfig):
    """Configuration for simplified FinBERT classifier."""

    model_type = "finbert_classifier"

    def __init__(
        self,
        num_level1: int = 8,
        num_sentiment: int = 3,
        classifier_dropout: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_level1 = num_level1
        self.num_sentiment = num_sentiment
        self.classifier_dropout = classifier_dropout
        self.alpha = alpha
        self.gamma = gamma


class FinBERTClassifier(BertPreTrainedModel):
    """BERT backbone + dual classification heads (L1 category + sentiment)."""

    config_class = FinBERTClassifierConfig

    def __init__(self, config: FinBERTClassifierConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        hidden = config.hidden_size
        drop = config.classifier_dropout

        self.dropout = nn.Dropout(drop)

        self.l1_fc1 = nn.Linear(hidden, hidden // 2)
        self.l1_activation = nn.GELU()
        self.l1_dropout = nn.Dropout(drop)
        self.l1_fc2 = nn.Linear(hidden // 2, config.num_level1)

        self.sent_fc1 = nn.Linear(hidden, hidden // 4)
        self.sent_activation = nn.GELU()
        self.sent_dropout = nn.Dropout(drop)
        self.sent_fc2 = nn.Linear(hidden // 4, config.num_sentiment)

        self.alpha = config.alpha
        self.gamma = config.gamma
        self.loss_fn = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.out_features in [self.config.hidden_size // 2, self.config.hidden_size // 4]:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        l1_label: torch.Tensor | None = None,
        sentiment_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled = self.dropout(outputs.pooler_output)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        mean_pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        mean_pooled = self.dropout(mean_pooled)

        l1_hidden = self.l1_activation(self.l1_fc1(mean_pooled))
        l1_hidden = self.l1_dropout(l1_hidden)
        l1_logits = self.l1_fc2(l1_hidden)

        sent_hidden = self.sent_activation(self.sent_fc1(pooled))
        sent_hidden = self.sent_dropout(sent_hidden)
        sent_logits = self.sent_fc2(sent_hidden)

        result: dict[str, torch.Tensor] = {
            "l1_logits": l1_logits,
            "sentiment_logits": sent_logits,
        }

        if l1_label is not None:
            l1_loss = self.loss_fn(l1_logits, l1_label)
            total_loss = self.alpha * l1_loss
            result["loss"] = total_loss
            result["l1_loss"] = l1_loss

            if sentiment_label is not None:
                sent_loss = self.loss_fn(sent_logits, sentiment_label)
                total_loss = total_loss + self.gamma * sent_loss
                result["sentiment_loss"] = sent_loss
                result["loss"] = total_loss

        return result


def load_finbert_classifier(
    pretrained_model: str,
    num_level1: int = 8,
    num_sentiment: int = 3,
    dropout: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.1,
) -> FinBERTClassifier:
    """Initialize FinBERT classifier from a pretrained BERT checkpoint."""
    config = FinBERTClassifierConfig.from_pretrained(
        pretrained_model,
        num_level1=num_level1,
        num_sentiment=num_sentiment,
        classifier_dropout=dropout,
        alpha=alpha,
        gamma=gamma,
    )
    model = FinBERTClassifier.from_pretrained(
        pretrained_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model  # type: ignore


# ─── ONNX export wrapper ─────────────────────────────────────────────────────────


class OnnxFinBERTWrapper(torch.nn.Module):
    """ONNX-compatible wrapper: accepts raw tensors, returns only logits dict.

    This wrapper strips the train-time loss computation so the exported
    model performs pure inference (matching the user's export script).
    """

    def __init__(self, finbert_model: FinBERTClassifier):
        super().__init__()
        self.bert = finbert_model.bert
        self.dropout = finbert_model.dropout
        self.l1_fc1 = finbert_model.l1_fc1
        self.l1_activation = finbert_model.l1_activation
        self.l1_dropout = finbert_model.l1_dropout
        self.l1_fc2 = finbert_model.l1_fc2
        self.sent_fc1 = finbert_model.sent_fc1
        self.sent_activation = finbert_model.sent_activation
        self.sent_dropout = finbert_model.sent_dropout
        self.sent_fc2 = finbert_model.sent_fc2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled = self.dropout(outputs.pooler_output)

        # Mean pooling (same as model.py)
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        mean_pooled = self.dropout(mean_pooled)

        l1_hidden = self.l1_activation(self.l1_fc1(mean_pooled))
        l1_hidden = self.l1_dropout(l1_hidden)
        l1_logits = self.l1_fc2(l1_hidden)

        sent_hidden = self.sent_activation(self.sent_fc1(pooled))
        sent_hidden = self.sent_dropout(sent_hidden)
        sent_logits = self.sent_fc2(sent_hidden)

        return {"logits": l1_logits, "sentiment_logits": sent_logits}


def export_finbert_to_onnx(
    model_dir: Path,
    onnx_path: Path,
    max_seq_length: int = 128,
    opset_version: int = 14,
) -> None:
    """Load best checkpoint and export to ONNX."""
    model = FinBERTClassifier.from_pretrained(str(model_dir))
    model.eval()

    wrapper = OnnxFinBERTWrapper(model)

    dummy_input_ids = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, max_seq_length, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits", "sentiment_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
            "sentiment_logits": {0: "batch_size"},
        },
    )
    print(f"✓ ONNX model exported to {onnx_path}")
