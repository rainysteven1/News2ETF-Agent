"""CUDA/CPU auto-detection — all ML components honour this."""

from __future__ import annotations

_device: str | None = None
_torch_available = False

try:
    import torch
    _torch_available = True
except ImportError:
    pass


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    global _device
    if _device is None:
        if _torch_available:
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            _device = "cpu"
    return _device


def is_cuda() -> bool:
    return get_device() == "cuda"
