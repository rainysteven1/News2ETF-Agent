"""Phase 2 Dry Run: 训练 Agent 决策能力的"练兵场"。

运行 2024-10 ~ 2025-06 每周回测，生成 decision_logs.jsonl。
每周一：Agent 决策 → 执行 → 记录 Guardrail 事件 → 周末计算收益

用法：
    python scripts/run_phase2_dry_run.py
    python scripts/run_phase2_dry_run.py --start 2024-10-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.agent.decision_logger import DecisionLogger, DecisionRecord
from src.agent.daily_guardrail import DailyGuardrailMonitor, FORBIDDEN_ZONEStateMachine
from src.agent.features import AgentFeatureBuilder
from src.agent.prompt_manager import PromptManager
from src.agent.rule_engine import WeeklyRuleEngine
from src.agent.state import AgentState
from src.agent.workflow import build_workflow
from src.config import load_config


def run_phase2_dry_run(
    start_date: str = "2024-10-01",
    end_date: str = "2025-06-30",
    config_path: str | None = None,
) -> Path:
    """运行 Phase 2 Dry Run.

    流程：
      1. 加载 TCN + LightGBM + IForest ONNX 模型（如存在）
      2. export_phase2_dataset() 导出 agent_features.parquet（如存在）
      3. 每周一：Agent 决策 → 执行 → 记录 Guardrail 事件 → 周末计算收益
      4. 生成 decision_logs.jsonl
    """
    # Load config
    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()

    output_path = config.data.output_logs
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    logger.info(f"[Phase2 Dry Run] start={start_date} end={end_date}")

    # Decision logger
    decision_logger = DecisionLogger(log_path=output_path)

    # Guardrail monitor
    guardrail_monitor = DailyGuardrailMonitor()

    # FORBIDDEN_ZONE state machine
    forbidden_machine = FORBIDDEN_ZONEStateMachine()

    # Weekly rule engine
    rule_engine = WeeklyRuleEngine()

    # Prompt manager (for good/bad patterns)
    prompt_manager = PromptManager(decision_logger)

    # Feature builder (reads from configured data paths)
    feature_builder = AgentFeatureBuilder()

    # Build LangGraph workflow
    workflow = build_workflow(config)

    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Iterate week by week
    current_dt = start_dt
    week_count = 0

    while current_dt <= end_dt:
        week_start = current_dt.strftime("%Y-%m-%d")
        week_end = (current_dt + timedelta(days=6)).strftime("%Y-%m-%d")

        logger.info(f"[Phase2] Week {week_start} ~ {week_end}")

        # Build agent features for this week
        try:
            agent_features = feature_builder.build_agent_features(
                date=week_start,
                current_holdings={},  # Will be populated from previous week
                current_time=f"{week_start} 09:30:00",
            )
        except Exception as e:
            logger.warning(f"[Phase2] Failed to build features for {week_start}: {e}")
            agent_features = {}

        current_context = {
            "market_state": agent_features.get("market_state", {}).get("market_state", "neutral"),
            "vol_percentile": agent_features.get("market_state", {}).get("market_volatility", 0.0),
            "sector_signals": {k: (v[-1] if v else 0.0) for k, v in agent_features.get("tcn_sequence", {}).items()},
            "forbidden_zones": {sector: "forbidden" for sector in forbidden_machine.get_forbidden_sectors(week_start)},
            "date": week_start,
        }
        good_patterns, bad_patterns, reasoning_summary = prompt_manager.update_prompt(current_context)
        agent_features["good_patterns"] = good_patterns
        agent_features["bad_patterns"] = bad_patterns
        agent_features["reasoning_summary"] = reasoning_summary

        forbidden_dict = {
            sector: (forbidden_machine.get_forbidden_info(sector) or {}).get("reason", "forbidden")
            for sector in forbidden_machine.get_forbidden_sectors(week_start)
        }

        # Build agent state
        agent_state: AgentState = {
            "date": week_start,
            "last_week_pnl": 0.0,
            "last_week_holdings": {},
            "last_week_returns": {},
            "observations": {},
            "messages": [],
            "decisions": [],
            "is_risk_passed": False,
            "retry_count": 0,
            "last_error": "",
            "loop_step": 0,
            "forbidden_sectors": forbidden_dict,
            "tcn_sequence": agent_features.get("tcn_sequence", {}),
            "decision_context": agent_features,
            "last_guardrail_events": [],
        }

        # Run agent workflow
        try:
            result = workflow.invoke(agent_state)
            decisions = result.get("decisions", [])
            logger.info(f"[Phase2] {week_start}: {len(decisions)} decisions made")
        except Exception as e:
            logger.error(f"[Phase2] Workflow failed for {week_start}: {e}")
            decisions = []

        # Apply weekly rules
        try:
            level1_plan = []
            level2_plan = []
            if decisions:
                first = decisions[0]
                level1_plan = [p.model_dump() for p in getattr(first, "level1_plan", [])]
                level2_plan = [p.model_dump() for p in getattr(first, "level2_plan", [])]

            adjusted_plan, violations = rule_engine.apply_weekly_rules(
                level1_plan=level1_plan,
                last_week_pnl=0.0,  # Will be filled from previous week
                last_week_holdings={},
                last_week_returns={},
            )
            if violations:
                logger.warning(f"[Phase2] Rule violations for {week_start}: {violations}")
        except Exception as e:
            logger.warning(f"[Phase2] Rule engine failed for {week_start}: {e}")
            adjusted_plan = decisions

        # Check guardrails
        try:
            guardrail_signals = guardrail_monitor.check_guardrail_trigger(
                current_date=week_start,
                positions={},
                etf_prices={},
                news_df=None,
            )
            for signal in guardrail_signals:
                forbidden_machine.mark_forbidden(
                    sector=signal.meta_sector,
                    reason=signal.reason,
                    trigger_type=signal.trigger_type,
                    current_date=week_start,
                )
                guardrail_monitor.state_machine.mark_forbidden(
                    sector=signal.meta_sector,
                    reason=signal.reason,
                    trigger_type=signal.trigger_type,
                    current_date=week_start,
                )
            adjusted_plan, overrides = guardrail_monitor.apply_forbidden_zone(
                agent_plan=adjusted_plan,
                current_date=week_start,
            )
            if overrides:
                logger.info(f"[Phase2] Guardrail overrode {len(overrides)} decisions for {week_start}")
        except Exception as e:
            logger.warning(f"[Phase2] Guardrail check failed for {week_start}: {e}")
            guardrail_signals = []

        # Log decision
        try:
            record = DecisionRecord(
                monday_date=week_start,
                agent_input=agent_features,
                level1_plan=adjusted_plan,
                level2_plan=level2_plan,
                weekly_return=0.0,
                guardrail_events=[signal.__dict__ for signal in guardrail_signals],
                tcn_prediction_errors=[],
                reasoning_summary=reasoning_summary,
                quality_label="neutral",
            )
            decision_logger.log_decision(record)
        except Exception as e:
            logger.error(f"[Phase2] Failed to log decision for {week_start}: {e}")

        week_count += 1
        current_dt += timedelta(days=7)

    logger.success(f"[Phase2 Dry Run] Done. {week_count} weeks processed. Log: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Dry Run")
    parser.add_argument("--start", default="2024-10-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-06-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", default=None, help="Path to config.toml")
    args = parser.parse_args()

    run_phase2_dry_run(start_date=args.start, end_date=args.end, config_path=args.config)


if __name__ == "__main__":
    main()
