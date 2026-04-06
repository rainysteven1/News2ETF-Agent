"""Models package — major (L1) and sub (L2) classifiers."""

from trainer.src.models.major import (
    MajorClassifier,
    export_major_to_onnx,
    load_major_classifier,
)
from trainer.src.models.signals import (
    TCN,
    SpatialDropout,
    TCNFanIn,
    export_tcn_fanin_to_onnx,
    export_tcn_to_onnx,
)
from trainer.src.models.sub import (
    SubClassifier,
    export_sub_to_onnx,
    load_sub_classifier,
)

__all__ = [
    "MajorClassifier",
    "load_major_classifier",
    "export_major_to_onnx",
    "SpatialDropout",
    "TCN",
    "TCNFanIn",
    "export_tcn_to_onnx",
    "export_tcn_fanin_to_onnx",
    "SubClassifier",
    "load_sub_classifier",
    "export_sub_to_onnx",
]
