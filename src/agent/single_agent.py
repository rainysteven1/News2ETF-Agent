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
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from src.agent.client import LLMClient
from src.agent.prompts import researcher_prompt, tool_descriptions, trader_prompt
from src.agent.state import AgentState, TradeDecision
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

    # Build clean research context — only researcher thinking (AIMessage) and raw evidence (ToolMessage)
    msg_contents = []
    for m in state.get("messages", []):
        if not hasattr(m, "content") or not m.content:
            continue
        # ToolMessage content = raw evidence from tools
        if isinstance(m, ToolMessage):
            msg_contents.append(f"[{m.name}] {m.content[:300]}")
        # AIMessage content = researcher thinking (skip tool call artifacts)
        elif isinstance(m, AIMessage):
            if not getattr(m, "tool_calls", None):
                msg_contents.append(m.content)
    research_summary = "\n\n".join(msg_contents)

    holdings_str = (
        "\n".join(f"  - {ind}: {w:.3f}" for ind, w in state.get("last_week_holdings", {}).items() if w > 0)
        or "  (empty)"
    )

    prompt = trader_prompt(
        date=state["date"],
        research_summary=research_summary,
        last_week_pnl=state.get("last_week_pnl", 0.0),
        holdings=holdings_str,
        max_weight=config.agent.max_weight_per_industry,
        max_total=config.agent.max_total_weight,
    )

    # Try structured output first (OpenAI response_format)
    decisions: list[TradeDecision] = []
    try:
        result = client.chat_structured(
            system_prompt="你是一名专业的 ETF 基金经理，负责最终仓位拍板。严格输出 JSON。",
            user_prompt=prompt,
            response_model={
                "type": "json_schema",
                "json_schema": {
                    "name": "WeeklyTradePlan",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decisions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "industry": {"type": "string"},
                                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                        "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["industry", "action", "weight", "reason"],
                                },
                            },
                            "market_outlook": {"type": "string"},
                        },
                        "required": ["decisions"],
                    },
                },
            },
        )
        data = json.loads(result) if isinstance(result, str) else result
        decisions = [TradeDecision(**d) for d in data.get("decisions", [])]
    except Exception:
        # Fallback: plain JSON
        try:
            text = client.chat("", prompt).strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            data = json.loads(text)
            decisions = [TradeDecision(**d) for d in data] if isinstance(data, list) else []
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

    research_summary = "\n\n".join(
        f"[{m.name}] {m.content[:300]}" if isinstance(m, ToolMessage) else m.content
        for m in messages
        if hasattr(m, "content")
        and m.content
        and (isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)))
    )

    holdings_str = (
        "\n".join(f"  - {ind}: {w:.3f}" for ind, w in state.get("last_week_holdings", {}).items() if w > 0)
        or "  (empty)"
    )

    warning_block = f"\n\n## Risk Guard Warning\n{risk_warning}\n" if risk_warning else ""

    prompt = trader_prompt(
        date=state["date"],
        research_summary=research_summary,
        last_week_pnl=state.get("last_week_pnl", 0.0),
        holdings=holdings_str,
        max_weight=config.agent.max_weight_per_industry,
        max_total=config.agent.max_total_weight,
    )
    prompt = warning_block + prompt

    decisions: list[TradeDecision] = []
    try:
        result = client.chat_structured(
            system_prompt="你是一名专业的 ETF 基金经理，负责最终仓位拍板。严格输出 JSON。",
            user_prompt=prompt,
            response_model={
                "type": "json_schema",
                "json_schema": {
                    "name": "WeeklyTradePlan",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "decisions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "industry": {"type": "string"},
                                        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                        "weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                        "reason": {"type": "string"},
                                    },
                                    "required": ["industry", "action", "weight", "reason"],
                                },
                            },
                            "market_outlook": {"type": "string"},
                        },
                        "required": ["decisions"],
                    },
                },
            },
        )
        data = json.loads(result) if isinstance(result, str) else result
        decisions = [TradeDecision(**d) for d in data.get("decisions", [])]
    except Exception:
        try:
            text = client.chat("", prompt).strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])
            data = json.loads(text)
            decisions = [TradeDecision(**d) for d in data] if isinstance(data, list) else []
        except Exception:
            decisions = []

    return {
        "decisions": decisions,
        "is_risk_passed": False,
        "retry_count": state.get("retry_count", 0),
    }


# ─── Risk Check Node ──────────────────────────────────────────────────────────


def risk_check_node(state: AgentState, config: AgentRootConfig, mapper=None) -> dict:
    """Risk guard: validate weight constraints, mirror positions, Beta penalties, and handle empty decisions.

    - Empty decisions → default to all-hold (no position changes).
    - Weight/size violations → retry to trader for soft correction.
    - Mirror positions (same correlation cluster, both heavily weighted) → retry to trader.
    - Beta penalty: if last_week_pnl < 0, no new very_high Beta buys → retry to trader.
    - If passed → end.
    """
    from loguru import logger
    decisions = state.get("decisions", [])
    last_week_pnl: float = state.get("last_week_pnl", 0.0)
    last_holdings: dict[str, float] = state.get("last_week_holdings", {})

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

    errors = []

    # ── Basic weight constraints ───────────────────────────────────────────────
    for d in decisions:
        if d.action == "buy" and d.weight > config.agent.max_weight_per_industry:
            errors.append(f"[{d.industry}] weight {d.weight:.3f} > max {config.agent.max_weight_per_industry}")

    total = sum(d.weight for d in decisions if d.action == "buy")
    if total > config.agent.max_total_weight:
        errors.append(f"Total weight {total:.3f} > max {config.agent.max_total_weight}")

    # ── Beta penalty: no new very_high Beta buys when losing week ──────────────
    if last_week_pnl < 0 and mapper is not None:
        for d in decisions:
            if d.action != "buy" or d.weight == 0:
                continue
            # Only check industries that are NOT already held (new positions only)
            if d.industry in last_holdings:
                continue
            beta = mapper.small_cat_beta(d.industry)
            if beta == "very_high":
                errors.append(
                    f"[Beta Penalty] last_week_pnl={last_week_pnl:.2%} < 0: "
                    f"cannot ADD new very_high Beta position '{d.industry}'. "
                    f"Either skip this or reduce to hold."
                )

    # ── Mirror position check via correlation clusters ─────────────────────────
    if mapper is not None:
        cluster_groups: dict[str, list[str]] = {}
        for d in decisions:
            if d.action == "buy" and d.weight >= 0.15:
                cluster = mapper.small_cat_cluster(d.industry)
                if cluster not in cluster_groups:
                    cluster_groups[cluster] = []
                cluster_groups[cluster].append(d.industry)

        for cluster, industries in cluster_groups.items():
            if cluster == "unknown" or len(industries) < 2:
                continue
            industries_str = ", ".join(industries)
            errors.append(
                f"[Mirror Conflict] correlation_cluster='{cluster}': {industries_str} "
                f"are mirrors — do NOT both hold at high weight. Reduce one to < 0.15."
            )

    if errors:
        warning = (
            "[RISK GUARD] Your decision violated constraints:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n请重新输出一个符合约束的 JSON 数组。"
        )
        retry_count = state.get("retry_count", 0) + 1
        last_error = "; ".join(errors)

        if retry_count >= 3:
            logger.warning(
                "[RISK GUARD] Retry limit reached for week {} — accepting as-is.",
                state.get("date", "unknown"),
            )
            return {"is_risk_passed": True, "decisions": decisions, "retry_count": retry_count}

        return {
            "is_risk_passed": False,
            "retry_count": retry_count,
            "last_error": last_error,
            "messages": [AIMessage(content=warning)],
        }

    return {"is_risk_passed": True, "retry_count": state.get("retry_count", 0)}
