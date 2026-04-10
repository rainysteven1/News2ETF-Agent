from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.prompt_manager import PromptManager
from src.agent.prompts import researcher_prompt, tool_descriptions, trader_prompt
from src.agent.state import AgentState, MetaSectorPlan, SectorStatus, TradeDecision


def test_runtime_agent_modules_resolve_local_files() -> None:
    runtime_agent_root = Path(__file__).resolve().parents[1] / "src" / "agent"
    runtime_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
import src.agent.state as state_module
import src.agent.prompts as prompts_module
import src.agent.prompt_manager as prompt_manager_module
print(json.dumps({
    "state": state_module.__file__,
    "prompts": prompts_module.__file__,
    "prompt_manager": prompt_manager_module.__file__,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(runtime_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert Path(payload["state"]).resolve() == runtime_agent_root / "state.py"
    assert Path(payload["prompts"]).resolve() == runtime_agent_root / "prompts.py"
    assert Path(payload["prompt_manager"]).resolve() == runtime_agent_root / "prompt_manager.py"


def test_runtime_prompts_use_runtime_prompt_files() -> None:
    research = researcher_prompt("2024-01-01", "env ctx")
    trader = trader_prompt(
        date="2024-01-01",
        research_summary="summary",
        last_week_pnl=0.1,
        holdings="{}",
        max_weight=0.3,
        max_total=1.0,
    )
    tool_text = tool_descriptions()

    assert "2024-01-01" in research
    assert "env ctx" in research
    assert "summary" in trader
    assert "Tool Descriptions" in tool_text


def test_runtime_state_models_still_validate() -> None:
    plan = MetaSectorPlan(meta_sector="科技成长", action="buy", weight=0.2)
    decision = TradeDecision(industry="半导体/芯片", action="buy", weight=0.2, level1_plan=[plan])
    state: AgentState = {
        "date": "2024-01-01",
        "messages": [],
        "observations": {},
        "decisions": [decision],
        "is_risk_passed": False,
        "retry_count": 0,
        "last_error": "",
        "loop_step": 0,
        "last_week_pnl": 0.0,
        "last_week_holdings": {},
        "last_week_returns": {},
        "forbidden_sectors": {"科技成长": "test"},
        "tcn_sequence": {},
        "decision_context": {},
        "last_guardrail_events": [],
    }

    assert state["decisions"][0].level1_plan[0].meta_sector == "科技成长"
    assert SectorStatus.NORMAL.value == "normal"


def test_runtime_prompt_manager_works_with_mock_logger() -> None:
    mock_logger = MagicMock()
    mock_logger.load_recent_decisions.return_value = []
    mock_logger.get_patterns_for_context.return_value = ([], [])

    manager = PromptManager(logger=mock_logger)
    good, bad, summary = manager.update_prompt(
        {
            "market_state": "neutral",
            "vol_percentile": 0.5,
            "sector_signals": {},
            "forbidden_zones": {},
        }
    )

    assert isinstance(good, list)
    assert isinstance(bad, list)
    assert isinstance(summary, str)
