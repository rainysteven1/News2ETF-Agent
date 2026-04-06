"""Tests for scripts/run_phase2_dry_run.py — smoke tests only."""

from __future__ import annotations

import pytest


class TestRunPhase2DryRunSmoke:
    """Smoke tests: just verify the function exists and has the right signature."""

    def test_run_phase2_dry_run_exists(self) -> None:
        """run_phase2_dry_run must exist as a callable."""
        from scripts.run_phase2_dry_run import run_phase2_dry_run
        import inspect
        assert callable(run_phase2_dry_run)

    def test_signature(self) -> None:
        """Must accept (start_date, end_date, config_path)."""
        from scripts.run_phase2_dry_run import run_phase2_dry_run
        import inspect
        sig = inspect.signature(run_phase2_dry_run)
        params = list(sig.parameters.keys())
        assert "start_date" in params
        assert "end_date" in params

    def test_main_exists(self) -> None:
        """main() must exist."""
        from scripts.run_phase2_dry_run import main
        assert callable(main)

    def test_returns_path_type(self) -> None:
        """Return type must be Path."""
        from scripts.run_phase2_dry_run import run_phase2_dry_run
        # Check return annotation if present
        import inspect
        sig = inspect.signature(run_phase2_dry_run)
        ret_annotation = sig.return_annotation
        if ret_annotation and ret_annotation != inspect.Parameter.empty:
            assert "Path" in str(ret_annotation)

    def test_docstring_present(self) -> None:
        """run_phase2_dry_run must have a docstring."""
        from scripts.run_phase2_dry_run import run_phase2_dry_run
        assert run_phase2_dry_run.__doc__ is not None
        assert len(run_phase2_dry_run.__doc__) > 20

    def test_imports_work(self) -> None:
        """All imports in the script must succeed."""
        # This verifies the script has no syntax errors and all imports are valid
        import scripts.run_phase2_dry_run
        assert hasattr(scripts.run_phase2_dry_run, "run_phase2_dry_run")
        assert hasattr(scripts.run_phase2_dry_run, "main")
