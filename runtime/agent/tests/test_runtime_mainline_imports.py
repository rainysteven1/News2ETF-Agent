from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_runtime_mainline_modules_resolve_local_files() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    runtime_src = runtime_root / "src"
    script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import src.agent.workflow as workflow_module
import src.agent.single_agent as single_agent_module
import src.backtest.engine as engine_module
print(json.dumps({
    "workflow": workflow_module.__file__,
    "single_agent": single_agent_module.__file__,
    "engine": engine_module.__file__,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(runtime_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert Path(payload["workflow"]).resolve() == runtime_src / "agent" / "workflow.py"
    assert Path(payload["single_agent"]).resolve() == runtime_src / "agent" / "single_agent.py"
    assert Path(payload["engine"]).resolve() == runtime_src / "backtest" / "engine.py"
