"""Major (L1) classifier — BERT backbone + dual heads (major category + sentiment)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class MajorClassifierConfig(PretrainedConfig):
    model_type = "major_classifier"

    def __init__(
        self,
        num_level1: int = 8,
        num_sentiment: int = 3,
        classifier_dropout: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.1,
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
        self.num_level1 = num_level1
        self.num_sentiment = num_sentiment
        self.classifier_dropout = classifier_dropout
        self.alpha = alpha
        self.gamma = gamma


class MajorClassifier(BertPreTrainedModel):
    """BERT backbone + dual classification heads (major category + sentiment)."""

    config_class = MajorClassifierConfig

    def __init__(self, config: MajorClassifierConfig):
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


def load_major_classifier(
    pretrained_model: str,
    num_level1: int = 8,
    num_sentiment: int = 3,
    dropout: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 0.1,
) -> MajorClassifier:
    config = MajorClassifierConfig.from_pretrained(
        pretrained_model,
        num_level1=num_level1,
        num_sentiment=num_sentiment,
        classifier_dropout=dropout,
        alpha=alpha,
        gamma=gamma,
    )
    model = MajorClassifier.from_pretrained(
        pretrained_model,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model  # type: ignore


def export_major_to_onnx(
    model_dir: Path,
    onnx_path: Path,
    max_seq_length: int = 128,
    opset_version: int = 14,
) -> None:
    """Export a Major model to ONNX format."""
    from trainer.src.utils import get_logger

    logger = get_logger()

    try:
        model = MajorClassifier.from_pretrained(str(model_dir))
        model.to("cpu")
        model.eval()

        class OnnxWrapper(torch.nn.Module):
            def __init__(self, m: MajorClassifier):
                super().__init__()
                self.bert = m.bert
                self.dropout = m.dropout
                self.l1_fc1 = m.l1_fc1
                self.l1_activation = m.l1_activation
                self.l1_dropout = m.l1_dropout
                self.l1_fc2 = m.l1_fc2
                self.sent_fc1 = m.sent_fc1
                self.sent_activation = m.sent_activation
                self.sent_dropout = m.sent_dropout
                self.sent_fc2 = m.sent_fc2

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
            ):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                pooled = self.dropout(outputs.pooler_output)
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

        wrapper = OnnxWrapper(model)

        dummy_input_ids = torch.ones(1, max_seq_length, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
        dummy_token_type_ids = torch.zeros(1, max_seq_length, dtype=torch.long)

        onnx_path.parent.mkdir(parents=True, exist_ok=True)

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

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        tokenizer_dir = onnx_path.parent / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(tokenizer_dir)

        logger.info(f"[ONNX] Exported Major to {onnx_path}")

    except Exception as exc:
        logger.warning(f"[ONNX] Export failed ({exc}), pth checkpoint saved for manual conversion")
        raise
