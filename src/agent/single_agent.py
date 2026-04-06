"""LangGraph node functions — agent / tools / decide / risk_check.

Topology:
    agent ──should_continue──→ tools ──→ agent (loop)
                    │
                    └──→ finalize ──→ risk_check ──→ END
                                              │
                                         retry ──→ agent
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.agent.client import LLMClient
from src.agent.prompts import researcher_prompt, tool_descriptions, trader_prompt
from src.agent.state import AgentState, ETFSelections, MetaSectorPlan, SectorStatus, TradeDecision
from src.config import AgentRootConfig

# ─── Conditional Edges ──────────────────────────────────────────────────────────


def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    """Router: if last AI message has tool_calls → tools; else → finalize.

    Forced finalize when loop_step >= 6 to prevent infinite ReAct loops.
    """
    if state.get("loop_step", 0) >= 6:
        return "finalize"

    messages = state.get("messages", [])
    if not messages:
        return "finalize"

    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "finalize"


def risk_should_retry(state: AgentState) -> Literal["trader", "researcher", "end"]:
    """Router:
    - risk passed → end
    - weight/size error → trader (Trader can self-correct without re-research)
    - logic error → researcher (need fresh research)
    """
    if state.get("is_risk_passed", True):
        return "end"
    last_error = state.get("last_error", "")
    if "weight" in last_error or "total" in last_error or "max" in last_error:
        return "trader"
    return "researcher"


# ─── Agent Node (Researcher) ───────────────────────────────────────────────────


def _langchain_to_openai_message(m: BaseMessage | ToolMessage) -> dict:
    """Convert a LangChain message to OpenAI message dict format."""
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    role = role_map.get(getattr(m, "type", "ai"), "assistant")

    msg: dict = {"role": role}

    if hasattr(m, "content") and m.content:
        msg["content"] = m.content

    # ToolMessage → role=tool + tool_call_id
    if hasattr(m, "tool_call_id") and getattr(m, "tool_call_id", None):
        msg["tool_call_id"] = m.tool_call_id
    if hasattr(m, "name") and getattr(m, "name", None):
        msg["name"] = m.name

    return msg


def _format_prompt_context(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _build_trader_prompt(state: AgentState, config: AgentRootConfig, risk_warning: str = "") -> str:
    msg_contents = []
    for m in state.get("messages", []):
        if not hasattr(m, "content") or not m.content:
            continue
        if isinstance(m, ToolMessage):
            msg_contents.append(f"[{m.name}] {m.content[:300]}")
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            msg_contents.append(m.content)
    research_summary = "\n\n".join(msg_contents)

    holdings_str = (
        "\n".join(f"  - {ind}: {w:.3f}" for ind, w in state.get("last_week_holdings", {}).items() if w > 0)
        or "  (empty)"
    )
    ctx = state.get("decision_context", {})
    prompt = trader_prompt(
        date=state["date"],
        research_summary=research_summary,
        last_week_pnl=state.get("last_week_pnl", 0.0),
        holdings=holdings_str,
        max_weight=config.agent.max_weight_per_industry,
        max_total=config.agent.max_total_weight,
        tcn_sequence=_format_prompt_context(ctx.get("tcn_sequence") or state.get("tcn_sequence")),
        news_summary=_format_prompt_context(ctx.get("news_summary")),
        market_state=_format_prompt_context(ctx.get("market_state")),
        position_state=_format_prompt_context(ctx.get("position_state")),
        sent_p_divergence=_format_prompt_context(ctx.get("sent_p_divergence")),
        forbidden_sectors=_format_prompt_context(state.get("forbidden_sectors")),
        good_patterns=_format_prompt_context(ctx.get("good_patterns")),
        bad_patterns=_format_prompt_context(ctx.get("bad_patterns")),
    )
    if risk_warning:
        return f"## Risk Guard Warning\n{risk_warning}\n\n{prompt}"
    return prompt


def _parse_trade_decisions(payload: Any) -> list[TradeDecision]:
    if isinstance(payload, str):
        payload = json.loads(payload)

    if isinstance(payload, list):
        return [TradeDecision(**item) for item in payload]

    if not isinstance(payload, dict):
        return []

    level1_raw = payload.get("level1_plan", [])
    level2_raw = payload.get("level2_plan", [])
    if level1_raw or level2_raw:
        return [
            TradeDecision(
                industry="meta_allocation",
                action="hold",
                weight=0.0,
                reason=payload.get("reasoning_summary", payload.get("market_outlook", "")),
                level1_plan=[MetaSectorPlan(**item) for item in level1_raw],
                level2_plan=[ETFSelections(**item) for item in level2_raw],
            )
        ]

    raw_decisions = payload.get("decisions", [])
    return [
        TradeDecision(
            industry=item["industry"],
            action=item["action"],
            weight=item["weight"],
            reason=item.get("reason", ""),
            selected_indices=item.get("selected_indices", []),
            selected_etf=item.get("selected_etf", ""),
        )
        for item in raw_decisions
    ]


def agent_node(state: AgentState, config: AgentRootConfig, bound_tools: list) -> dict:
    """Researcher: LLM thinks + calls tools to gather market intelligence.

    Uses chat_with_messages for true multi-turn ReAct: the full message history
    is passed to the LLM so it can see all prior tool results and its own thinking.
    """
    client = LLMClient(
        model=config.agent.llm_model,
        temperature=config.agent.llm_temperature,
    )

    holdings_str = (
        "\n".join(f"  - {ind}: {w:.3f}" for ind, w in state.get("last_week_holdings", {}).items() if w > 0)
        or "  (empty)"
    )
    pnl = state.get("last_week_pnl", 0.0)
    env_context = (
        f"## Week Context for {state['date']}\n- Last week return: {pnl:.2%}\n- Last week holdings:\n{holdings_str}\n"
    )

    new_user_msg = researcher_prompt(date=state["date"], env_context=env_context) + "\n" + tool_descriptions()

    # Build full message history in OpenAI format
    openai_messages = [_langchain_to_openai_message(m) for m in state.get("messages", [])]
    openai_messages.append({"role": "user", "content": new_user_msg})

    try:
        response = client.chat_with_messages(
            messages=openai_messages,
            tools=bound_tools,
        )
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"[ERROR] Researcher failed: {e}")],
        }

    content = response.get("content", "")
    tool_calls = response.get("tool_calls", [])

    return {
        "messages": [AIMessage(content=content, tool_calls=tool_calls or None)],
        "observations": {
            **state.get("observations", {}),
            "researcher_summary": content[:500],
        },
        "loop_step": state.get("loop_step", 0) + 1,
    }


# ─── Decide Node (Trader) ─────────────────────────────────────────────────────


def decide_node(state: AgentState, config: AgentRootConfig) -> dict:
    """Trader: reads all messages from research, outputs structured trade decisions.

    Single LLM call — no tool loop.
    """
    client = LLMClient(
        model=config.agent.llm_model,
        temperature=config.agent.llm_temperature,
    )

    prompt = _build_trader_prompt(state, config)

    # Try structured output first (OpenAI response_format)
    decisions: list[TradeDecision] = []
    try:
        result = client.chat_structured(
            system_prompt="你是一名专业的 ETF 基金经理，负责最终仓位拍板。严格输出 JSON。",
            user_prompt=prompt,
            response_model={
                "type": "json_schema",
                "json_schema": {
                    "name": "WeeklyTradePlanV2",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "level1_plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "meta_sector": {"type": "string"},
                                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                        "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["meta_sector", "action", "weight", "reason"],
                                },
                            },
                            "level2_plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "meta_sector": {"type": "string"},
                                        "selected_indices": {"type": "array", "items": {"type": "string"}},
                                        "selected_etf": {"type": "string"},
                                    },
                                    "required": ["meta_sector", "selected_indices", "selected_etf"],
                                },
                            },
                            "market_outlook": {"type": "string"},
                            "reasoning_summary": {"type": "string"},
                        },
                        "required": ["level1_plan", "level2_plan"],
                    },
                },
            },
        )
        decisions = _parse_trade_decisions(result)
    except Exception:
        try:
            text = client.chat("", prompt).strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            decisions = _parse_trade_decisions(text)
        except Exception:
            decisions = []

    return {
        "decisions": decisions,
        "is_risk_passed": False,
        "retry_count": 0,
    }


# ─── Tools Node ────────────────────────────────────────────────────────────────


def tools_node(state: AgentState, config: AgentRootConfig) -> dict:
    """Execute tools called by LLM.

    Returns {"messages": [ToolMessage, ...]} — add_messages MERGES into state["messages"].
    Each ToolMessage carries tool_call_id so LLM can match results to calls.
    """
    from src.agent.tools import TOOL_REGISTRY

    messages = state.get("messages", [])
    if not messages:
        return {}

    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", []) or []
    if not tool_calls:
        return {}

    new_messages = []
    observations = dict(state.get("observations", {}))

    for tc in tool_calls:
        # Get tool_call_id for proper result attribution
        tc_id = getattr(tc, "id", str(id(tc)))
        tool_name = tc.get("name") or ""
        raw_args = tc.get("arguments", {})

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = dict(raw_args)

        # Execute via TOOL_REGISTRY (explicit, no getattr)
        if tool_name in TOOL_REGISTRY:
            fn = TOOL_REGISTRY[tool_name]
            try:
                # Only pass args that the tool expects (exclude config/state)
                result = fn.invoke({**args})
            except Exception as e:
                result = f"ERROR: {e}"
        else:
            result = f"ERROR: unknown tool '{tool_name}'"

        # ToolMessage with tool_call_id — critical for LLM to match result to call
        tool_msg = ToolMessage(
            content=str(result),
            tool_call_id=tc_id,
            name=tool_name,
        )
        new_messages.append(tool_msg)
        observations[f"tool_{tool_name}"] = str(result)[:500]

    # Return messages list — add_messages in TypedDict will APPEND, not overwrite
    return {
        "messages": new_messages,
        "observations": observations,
    }


# ─── Trader Retry Node (Soft Correction) ─────────────────────────────────────


def trader_retry_node(state: AgentState, config: AgentRootConfig) -> dict:
    """Trader self-corrects after risk guard warning on weight/size.

    Trader re-runs with the risk warning prepended — no fresh research needed.
    """
    messages = state.get("messages", [])
    risk_warning = ""
    if messages and isinstance(messages[-1], AIMessage):
        risk_warning = messages[-1].content

    client = LLMClient(
        model=config.agent.llm_model,
        temperature=config.agent.llm_temperature,
    )

    prompt = _build_trader_prompt(state, config, risk_warning=risk_warning)

    decisions: list[TradeDecision] = []
    try:
        result = client.chat_structured(
            system_prompt="你是一名专业的 ETF 基金经理，负责最终仓位拍板。严格输出 JSON。",
            user_prompt=prompt,
            response_model={
                "type": "json_schema",
                "json_schema": {
                    "name": "WeeklyTradePlanV2",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "level1_plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "meta_sector": {"type": "string"},
                                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                        "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["meta_sector", "action", "weight", "reason"],
                                },
                            },
                            "level2_plan": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "meta_sector": {"type": "string"},
                                        "selected_indices": {"type": "array", "items": {"type": "string"}},
                                        "selected_etf": {"type": "string"},
                                    },
                                    "required": ["meta_sector", "selected_indices", "selected_etf"],
                                },
                            },
                            "market_outlook": {"type": "string"},
                            "reasoning_summary": {"type": "string"},
                        },
                        "required": ["level1_plan", "level2_plan"],
                    },
                },
            },
        )
        decisions = _parse_trade_decisions(result)
    except Exception:
        try:
            text = client.chat("", prompt).strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            decisions = _parse_trade_decisions(text)
        except Exception:
            decisions = []

    return {
        "decisions": decisions,
        "is_risk_passed": False,
        "retry_count": state.get("retry_count", 0),
    }


# ─── Risk Check Node ──────────────────────────────────────────────────────────


def _meta_sector_beta(mapper, meta_sector: str) -> str:
    """Get the highest beta among all sub-categories of a meta sector."""
    if mapper is None:
        return "medium"
    from src.utils.meta_sector_map import meta_to_subs

    subs = meta_to_subs(meta_sector)
    if not subs:
        return "medium"
    betas = [mapper.small_cat_beta(sub) for sub in subs]
    # Priority: very_high > high > medium > low
    priority = {"very_high": 4, "high": 3, "medium": 2, "low": 1}
    return max(betas, key=lambda b: priority.get(b, 0), default="medium")


def _meta_sector_cluster(mapper, meta_sector: str) -> str:
    """Get the most common cluster among sub-categories of a meta sector."""
    if mapper is None:
        return meta_sector
    from src.utils.meta_sector_map import meta_to_subs

    subs = meta_to_subs(meta_sector)
    if not subs:
        return meta_sector
    clusters = [mapper.small_cat_cluster(sub) for sub in subs]
    # Return the most common cluster
    from collections import Counter

    return Counter(clusters).most_common(1)[0][0]


def _normalize_level1_plan(
    level1_plan: list[MetaSectorPlan],
    meta_sectors: list[str],
) -> tuple[list[MetaSectorPlan], list[str]]:
    plan_by_sector = {item.meta_sector: item for item in level1_plan if item.meta_sector in meta_sectors}
    notes: list[str] = []
    normalized: list[MetaSectorPlan] = []
    for sector in meta_sectors:
        item = plan_by_sector.get(sector)
        if item is None:
            item = MetaSectorPlan(
                meta_sector=sector,
                action="hold",
                weight=0.0,
                reason="[AUTO_FILL] Missing sector in level1_plan",
            )
            notes.append(f"[{sector}] missing from level1_plan, auto-filled as hold")
        normalized.append(item)
    return normalized, notes


def _apply_level1_risk_rules(
    level1_plan: list[MetaSectorPlan],
    state: AgentState,
    config: AgentRootConfig,
    mapper,
) -> tuple[list[MetaSectorPlan], list[str]]:
    last_week_pnl: float = state.get("last_week_pnl", 0.0)
    last_holdings: dict[str, float] = state.get("last_week_holdings", {})
    forbidden = state.get("forbidden_sectors", {})
    errors: list[str] = []
    adjusted: list[MetaSectorPlan] = []

    for item in level1_plan:
        plan = item.model_copy(deep=True)

        if plan.meta_sector in forbidden and plan.action == "buy":
            plan.action = "hold"
            plan.weight = 0.0
            plan.reason = f"[FORBIDDEN_ZONE] {plan.reason}".strip()
            errors.append(f"[{plan.meta_sector}] forbidden zone active, buy downgraded to hold")

        if plan.action == "buy" and 0 < plan.weight < 0.05:
            plan.action = "hold"
            plan.weight = 0.0
            plan.reason = f"[MIN_THRESHOLD] {plan.reason}".strip()
            errors.append(f"[{plan.meta_sector}] weight below 5%, downgraded to hold")

        if plan.action == "buy" and plan.weight > config.agent.max_weight_per_industry:
            plan.weight = config.agent.max_weight_per_industry
            plan.reason = f"[WEIGHT_CAP] {plan.reason}".strip()
            errors.append(f"[{plan.meta_sector}] weight capped to {config.agent.max_weight_per_industry:.2f}")

        if last_week_pnl < 0 and mapper is not None and plan.action == "buy" and plan.weight > 0:
            is_new_position = last_holdings.get(plan.meta_sector, 0.0) < 0.01
            if is_new_position and _meta_sector_beta(mapper, plan.meta_sector) == "very_high":
                plan.action = "hold"
                plan.weight = 0.0
                plan.reason = f"[BETA_PENALTY] {plan.reason}".strip()
                errors.append(f"[{plan.meta_sector}] new very_high beta buy forbidden after losing week")

        adjusted.append(plan)

    total_buy_weight = sum(item.weight for item in adjusted if item.action == "buy")
    if total_buy_weight > config.agent.max_total_weight > 0:
        scale = config.agent.max_total_weight / total_buy_weight
        errors.append(
            f"[TOTAL_WEIGHT] scaled buy weights by {scale:.3f} to respect total cap {config.agent.max_total_weight:.2f}"
        )
        for item in adjusted:
            if item.action == "buy":
                item.weight = float(item.weight * scale)

    if mapper is not None:
        cluster_groups: dict[str, list[MetaSectorPlan]] = {}
        for item in adjusted:
            if item.action == "buy" and item.weight >= 0.15:
                cluster_groups.setdefault(_meta_sector_cluster(mapper, item.meta_sector), []).append(item)
        for cluster, items in cluster_groups.items():
            if cluster == "unknown" or len(items) < 2:
                continue
            sorted_items = sorted(items, key=lambda p: p.weight, reverse=True)
            for extra in sorted_items[1:]:
                extra.weight = min(extra.weight, 0.14)
                extra.reason = f"[MIRROR_CONFLICT] {extra.reason}".strip()
                errors.append(f"[{extra.meta_sector}] reduced below 15% due to mirror cluster '{cluster}'")

    return adjusted, errors


def _apply_level2_risk_rules(
    level2_plan: list[ETFSelections],
    allowed_meta_sectors: set[str],
) -> tuple[list[ETFSelections], list[str]]:
    adjusted: list[ETFSelections] = []
    errors: list[str] = []
    seen: set[str] = set()
    for item in level2_plan:
        if item.meta_sector in seen:
            errors.append(f"[{item.meta_sector}] duplicate Level 2 entry dropped")
            continue
        seen.add(item.meta_sector)
        if item.meta_sector not in allowed_meta_sectors:
            errors.append(f"[{item.meta_sector}] removed from Level 2 because Level 1 not active after risk")
            continue
        adjusted.append(item)
    return adjusted, errors


def risk_check_node(state: AgentState, config: AgentRootConfig, mapper=None) -> dict:
    """Apply Level 1 then Level 2 risk constraints and auto-normalize the final plan."""
    from loguru import logger

    from src.utils.meta_sector_map import get_meta_sectors

    decisions = state.get("decisions", [])

    if not decisions:
        logger.warning(
            "[RISK GUARD] No valid decisions for week {} — defaulting to empty position. "
            "Researcher may have failed to reach a conclusion.",
            state.get("date", "unknown"),
        )
        return {
            "is_risk_passed": True,
            "retry_count": state.get("retry_count", 0),
            "decisions": [],
        }

    first_decision = decisions[0]
    use_meta_format = bool(first_decision.level1_plan)

    if not use_meta_format:
        return {"is_risk_passed": True, "retry_count": state.get("retry_count", 0)}

    meta_sectors = get_meta_sectors()
    normalized_level1, norm_notes = _normalize_level1_plan(first_decision.level1_plan, meta_sectors)
    adjusted_level1, level1_notes = _apply_level1_risk_rules(
        normalized_level1,
        state=state,
        config=config,
        mapper=mapper,
    )

    allowed_level2 = {item.meta_sector for item in adjusted_level1 if item.action == "buy" and item.weight > 0}
    adjusted_level2, level2_notes = _apply_level2_risk_rules(first_decision.level2_plan, allowed_level2)

    updated_decision = first_decision.model_copy(deep=True)
    updated_decision.level1_plan = adjusted_level1
    updated_decision.level2_plan = adjusted_level2

    risk_notes = norm_notes + level1_notes + level2_notes
    if risk_notes:
        logger.info("[RISK GUARD] Applied {} adjustments for week {}", len(risk_notes), state.get("date", "unknown"))

    return {
        "is_risk_passed": True,
        "retry_count": state.get("retry_count", 0),
        "decisions": [updated_decision],
        "last_error": "; ".join(risk_notes),
    }
