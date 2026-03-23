"""LangGraph ReAct workflow — proper conditional edges topology.

Topology 2.0:
    agent ──should_continue──→ tools ──→ agent (loop)
                    │
                    └──→ finalize ──→ risk_check ──┬─→ END
                                                  │
                     risk_should_retry:           │
                       weight error → trader_retry ├─→ agent (retry)
                       logic error  → agent (retry)
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent import tools as agent_tools
from src.agent.single_agent import (
    agent_node,
    decide_node,
    risk_check_node,
    risk_should_retry,
    should_continue,
    tools_node,
    trader_retry_node,
)
from src.agent.state import AgentState
from src.config import AgentRootConfig
from src.utils.industry_map import IndustryMapper


def build_workflow(config: AgentRootConfig) -> CompiledStateGraph:
    """Build the LangGraph ReAct workflow."""
    bound_tools = [
        agent_tools.read_market_news,
        agent_tools.compute_ml_signals,
        agent_tools.check_last_week_pnl,
        agent_tools.retrieve_history,
    ]

    # IndustryMapper for risk guard: mirror position detection + beta metadata
    # best_etf cache goes to data/best_etf_by_index.parquet (next to existing file)
    mapper = IndustryMapper(
        dict_path=config.data.industry_dict,
        etf_info=config.data.etf_info,
        best_etf_path=config.data.etf_info.parent.parent / "best_etf_by_index.parquet",
    )

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("agent", lambda s: agent_node(s, config, bound_tools))
    workflow.add_node("tools", lambda s: tools_node(s, config))
    workflow.add_node("finalize", lambda s: decide_node(s, config))
    workflow.add_node("trader_retry", lambda s: trader_retry_node(s, config))
    workflow.add_node("risk_check", partial(risk_check_node, config=config, mapper=mapper))

    # Entry point
    workflow.set_entry_point("agent")

    # agent → (should_continue) → tools OR finalize
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "finalize": "finalize",
        },
    )

    # tools → loop back to agent
    workflow.add_edge("tools", "agent")

    # finalize → risk_check
    workflow.add_edge("finalize", "risk_check")

    # risk_check → (risk_should_retry) → trader_retry OR agent OR end
    workflow.add_conditional_edges(
        "risk_check",
        risk_should_retry,
        {
            "trader": "trader_retry",
            "researcher": "agent",
            "end": END,
        },
    )

    return workflow.compile()
