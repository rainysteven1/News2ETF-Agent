"""Temporal CNN (TCN) model for sentiment momentum prediction.

TCN uses stacked 1D dilated convolutions with causal padding to preserve temporal ordering.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import TracerWarning


class TemporalBlock(nn.Module):
    """One TCN residual block: two conv layers with dilation + weight norm + residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.dropout = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.out_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = out[:, :, : -self.conv1.padding[0]]  # type: ignore
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, : -self.conv2.padding[0]]  # type: ignore
        out = F.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        # Trim res to match out length if needed
        min_len = min(out.size(2), res.size(2))
        out = out[:, :, :min_len]
        res = res[:, :, :min_len]
        return self.out_activation(out + res)


class TCN(nn.Module):
    """Temporal CNN for multi-task sentiment prediction.

    Input:  (batch, seq_len, input_size)
    Output:
        - regression:  (batch, 1)  -- momentum score in [-1, 1]
        - classification: (batch, 1) -- probability of anomaly
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, dilation, dropout))

        self.network = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_size)
        out = x.transpose(1, 2)  # (batch, input_size, seq_len)

        for block in self.network:
            out = block(out)

        out = out.transpose(1, 2)  # (batch, seq_len, hidden)
        pooled = out.mean(dim=1)  # (batch, hidden)

        return torch.tanh(self.reg_head(pooled)), torch.sigmoid(self.cls_head(pooled))


def export_tcn_to_onnx(
    model: TCN,
    onnx_path: Path,
    seq_len: int = 5,
    input_size: int = 1,
    opset_version: int = 14,
) -> None:
    """Export a TCN model to ONNX format.

    The exported model takes (batch, seq_len, input_size) input
    and outputs two tensors: reg_output (batch, 1) and cls_output (batch, 1).
    """
    model.eval()

    class OnnxTCNWrapper(torch.nn.Module):
        def __init__(self, tcn: TCN):
            super().__init__()
            self.tcn = tcn

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.tcn(x)

    wrapper = OnnxTCNWrapper(model)
    dummy = torch.randn(1, seq_len, input_size)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        torch.onnx.export(
            wrapper,
            (dummy,),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["reg_output", "cls_output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "seq_len"},
                "reg_output": {0: "batch_size"},
                "cls_output": {0: "batch_size"},
            },
        )


class SpatialDropout(nn.Module):
    """Spatial Dropout：对 47 维中的随机子集（所有 6 通道一起丢）"""

    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 47, 6)
        if not self.training:
            return x
        mask = torch.rand(x.shape[0], 1, x.shape[2], 1, device=x.device) > self.p
        return x * mask / (1 - self.p)


class TCNFanIn(nn.Module):
    """Fan-in TCN: 输入 (batch, seq_len, 47, 6) → 输出 (batch, 8) 元板块动量

    架构：
      1. Spatial Dropout（防止过拟合）
      2. 空间压缩：47*6=282 → 128 → 64（MLP）
      3. 时间建模：TCN stack（kernel_size=3, dilation=[1,2,4,8]）
      4. 输出头：Linear(hidden, 8) — 8 个元板块动量
    """

    def __init__(
        self,
        n_sub=47,
        n_meta=8,
        input_size=6,
        hidden_size=64,
        num_layers=4,
        kernel_size=3,
        dropout=0.2,
        spatial_dropout_p=0.3,
    ):
        super().__init__()
        self.n_sub = n_sub
        self.n_meta = n_meta
        self.input_size = input_size
        self.spatial_dropout = SpatialDropout(p=spatial_dropout_p)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(n_sub * input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_size),
        )
        self.tcn = TCN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_meta),
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_meta),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, 47, 6)
        x = self.spatial_dropout(x)
        batch, seq_len, n_sub, channels = x.shape
        x = x.reshape(batch, seq_len, n_sub * channels)
        x = self.spatial_mlp(x)  # (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        # Use TCN's temporal blocks directly (skip TCN's own heads/pooling)
        for block in self.tcn.network:
            x = block(x)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_size)
        x = x.mean(dim=1)  # (batch, hidden_size)
        return torch.tanh(self.reg_head(x) * self.scale), torch.sigmoid(self.cls_head(x))


def export_tcn_fanin_to_onnx(
    model: TCNFanIn,
    onnx_path: Path,
    seq_len: int = 5,
    n_sub: int | None = None,
    input_size: int | None = None,
    opset_version: int = 14,
) -> None:
    model.eval()
    if n_sub is None:
        n_sub = model.n_sub
    if input_size is None:
        input_size = model.input_size
    dummy = torch.randn(1, seq_len, n_sub, input_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        torch.onnx.export(
            model,
            (dummy,),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["reg_output", "cls_output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "reg_output": {0: "batch_size"},
                "cls_output": {0: "batch_size"},
            },
        )
