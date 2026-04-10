from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from src.agent.single_agent import _apply_level1_risk_rules, _langchain_to_openai_message, agent_node, tools_node
from src.agent.state import MetaSectorPlan


def test_tools_node_preserves_tool_call_id_and_args_for_dict_tool_calls() -> None:
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "name": "demo_tool",
                        "args": {"ticker": "SPY"},
                    }
                ],
            )
        ],
        "observations": {},
        "decision_context": {},
        "tcn_sequence": {},
    }

    class DummyTool:
        def invoke(self, payload: dict) -> str:
            assert payload == {"ticker": "SPY"}
            return "ok"

    with patch("src.agent.tools.TOOL_REGISTRY", {"demo_tool": DummyTool()}):
        result = tools_node(state, config=None)

    assert len(result["messages"]) == 1
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call_123"
    assert tool_msg.name == "demo_tool"
    assert tool_msg.content == "ok"


def test_langchain_to_openai_message_preserves_assistant_tool_calls() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_123",
                "name": "demo_tool",
                "args": {"ticker": "SPY"},
            }
        ],
    )

    result = _langchain_to_openai_message(message)

    assert result == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "demo_tool",
                    "arguments": '{"ticker": "SPY"}',
                },
            }
        ],
    }


def test_agent_node_does_not_pass_none_tool_calls() -> None:
    state = {
        "date": "2024-01-01",
        "messages": [],
        "observations": {},
        "loop_step": 0,
        "last_week_pnl": 0.0,
        "last_week_holdings": {},
    }

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def chat_with_messages(self, messages: list[dict], tools: list | None = None) -> dict:
            return {"content": "done", "tool_calls": []}

    with patch("src.agent.single_agent.LLMClient", DummyClient):
        result = agent_node(state, config=type("C", (), {"agent": type("A", (), {"llm_model": "x", "llm_temperature": 0.0})()})(), bound_tools=[])

    message = result["messages"][-1]
    assert isinstance(message, AIMessage)
    assert message.content == "done"
    assert message.tool_calls == []


def test_apply_level1_risk_rules_zeroes_hold_with_positive_weight_without_position() -> None:
    level1_plan = [
        MetaSectorPlan(meta_sector="消费文娱", action="hold", weight=0.05, reason="satellite"),
    ]
    state = {
        "last_week_pnl": 0.0,
        "last_week_holdings": {},
        "forbidden_sectors": {},
    }
    config = type("C", (), {"agent": type("A", (), {"max_weight_per_industry": 0.3, "max_total_weight": 1.0})()})()

    adjusted, notes = _apply_level1_risk_rules(level1_plan, state=state, config=config, mapper=None)

    assert adjusted[0].action == "hold"
    assert adjusted[0].weight == 0.0
    assert "INVALID_HOLD_NO_POSITION" in adjusted[0].reason
    assert any("no existing position" in note for note in notes)
