"""LSTM + Attention model for sentiment momentum prediction.

torch is required — only imported/used when GPU mode is enabled.
CPU fallback is handled by RawScorer._score_cpu().
"""

from __future__ import annotations

from typing import Literal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise RuntimeError("torch not installed — models require torch. Install with: pip install torch")


class LSTMWithAttention(nn.Module):
    """LSTM + Multi-Head Self-Attention for multi-task sentiment prediction.

    Input:  (batch, seq_len, input_size)
    Output:
        - regression:  (batch, 1) — momentum score in [-1, 1]
        - classification: (batch, 1) — probability of anomaly (异常波动)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Multi-task heads
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),  # output: momentum in [-1, 1]
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),  # output: prob of anomaly
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reg_output, cls_output)."""
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

        # Self-attention (key=query=value=lstm_out)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # (batch, seq_len, hidden_size)
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual connection

        # Use last timestep for both tasks
        last = attn_out[:, -1, :]  # (batch, hidden_size)

        reg_out = torch.tanh(self.reg_head(last))          # [-1, 1]
        cls_out = torch.sigmoid(self.cls_head(last))       # [0, 1]

        return reg_out, cls_out


class LSTMModel(nn.Module):
    """Legacy LSTM (regression only) — kept for backwards compat."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.tanh(out)
