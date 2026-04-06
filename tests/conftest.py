"""Global pytest configuration — mock setfit before any module imports."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Mock only the problematic modules, NOT the parent packages
# so that submodule imports still work
sys.modules["setfit"] = MagicMock()
sys.modules["trainer.setfit_module"] = MagicMock()
sys.modules["trainer.setfit_module.model"] = MagicMock()
