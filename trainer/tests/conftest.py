from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]

repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

# Some trainer modules lazily reference SetFit; mock it for stable collection.
sys.modules["setfit"] = MagicMock()
sys.modules["trainer.setfit_module"] = MagicMock()
sys.modules["trainer.setfit_module.model"] = MagicMock()
