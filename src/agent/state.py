from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class TradeDecision(BaseModel):
    industry: str
    action: str  # "buy", "sell", "hold"
    weight: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class AgentState(TypedDict):
    """LangGraph runtime state — TypedDict + add_messages for automatic message history."""

    # ── 核心对话流 (自动累加) ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 静态环境上下文 (每个周初初始化一次) ──
    date: str
    last_week_pnl: float
    last_week_holdings: dict

    # ── 动态业务数据 (由 Node 更新，避免 token 爆炸) ──
    observations: dict[str, Any]

    # ── 最终产出 ──
    decisions: list[TradeDecision]

    # ── 运行控制 ──
    is_risk_passed: bool
    retry_count: int
    last_error: str  # set by risk_check_node to route trader vs researcher retry
    loop_step: int  # ReAct loop counter — incremented each time agent_node returns, caps at 6
