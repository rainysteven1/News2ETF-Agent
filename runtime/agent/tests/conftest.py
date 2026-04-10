from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

runtime_root_str = str(RUNTIME_ROOT)
repo_root_str = str(REPO_ROOT)

if runtime_root_str not in sys.path:
    sys.path.insert(0, runtime_root_str)
if repo_root_str not in sys.path:
    sys.path.insert(1, repo_root_str)

# Keep trainer-side optional SetFit imports from breaking runtime test collection.
sys.modules["setfit"] = MagicMock()
sys.modules["trainer.setfit_module"] = MagicMock()
sys.modules["trainer.setfit_module.model"] = MagicMock()
