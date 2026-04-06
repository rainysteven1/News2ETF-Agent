from enum import Enum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SectorStatus(Enum):
    """Sector trading status - normal or in forbidden zone."""

    NORMAL = "normal"
    FORBIDDEN_ZONE = "forbidden"


class MetaSectorPlan(BaseModel):
    """Level 1 plan: single meta sector trading plan."""

    meta_sector: str = Field(description="元板块名称，如'科技成长'、'高端制造'")
    action: str = Field(description="交易动作：buy/sell/hold")
    weight: float = Field(ge=0.0, le=1.0, description="仓位权重")
    reason: str = Field(default="", description="决策理由")


class ETFSelections(BaseModel):
    """Level 2 plan: ETF selection for a meta sector."""

    meta_sector: str = Field(description="元板块名称")
    selected_indices: list[str] = Field(
        default_factory=list, description="选择的跟踪指数列表"
    )
    selected_etf: str = Field(default="", description="最终选择的ETF代码和名称")


class TradeDecision(BaseModel):
    """Legacy trade decision schema (kept for backward compatibility)."""

    industry: str
    action: str  # "buy", "sell", "hold"
    weight: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    selected_indices: list[str] = Field(default_factory=list)
    selected_etf: str = ""
    # New fields for meta sector based decisions
    level1_plan: list[MetaSectorPlan] = Field(default_factory=list)
    level2_plan: list[ETFSelections] = Field(default_factory=list)
    sector_status: dict[str, SectorStatus] = Field(default_factory=dict)


class AgentState(TypedDict):
    """LangGraph runtime state — TypedDict + add_messages for automatic message history."""

    # ── 核心对话流 (自动累加) ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 静态环境上下文 (每个周初初始化一次) ──
    date: str
    last_week_pnl: float
    last_week_holdings: dict
    last_week_returns: dict

    # ── 动态业务数据 (由 Node 更新，避免 token 爆炸) ──
    observations: dict[str, Any]

    # ── 最终产出 ──
    decisions: list[TradeDecision]

    # ── 运行控制 ──
    is_risk_passed: bool
    retry_count: int
    last_error: str
    loop_step: int

    # ── 新增字段 (forbidden zone & decision context) ──
    forbidden_sectors: dict[str, str] = Field(
        default_factory=dict,
        description="处于禁闭期的板块，key=板块名，value=禁闭原因"
    )
    tcn_sequence: dict[str, list[float]] = Field(
        default_factory=dict,
        description="TCN输出的8元板块动量序列，key=元板块名，value=最近5天动量列表"
    )
    decision_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent决策时的完整上下文，包含A/B/C/D/E五类特征"
    )
    last_guardrail_events: list[dict] = Field(
        default_factory=list,
        description="最近触发的guardrail事件列表"
    )
