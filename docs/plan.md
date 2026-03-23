# Plan: 新闻量化交易 Agent 系统

## Context

用户决定**完全重启项目**，抛弃原有的：
- `src/` - FastAPI 后端 + 分类 pipeline
- `finbert/` - FinBERT ML 训练
- `agent/` - 旧的 feat/agent 分支

新项目的目标：**训练一个像人类操盘手的 Agent，直接看原始新闻+ETF走势决定买哪只ETF**。

核心变化：从"新闻分类系统"变成"端到端量化交易 Agent"。

---

## 最终项目结构

```
News2ETF-Engine/
├── src/                          # 所有代码统一放这里
│   ├── agent/                    # Agent 核心
│   │   ├── single_agent.py       # 单 Agent ReAct 实现
│   │   ├── state.py              # Agent State 定义
│   │   ├── prompts.py            # ReAct 引导 prompt
│   │   └── workflow.py           # LangGraph workflow
│   ├── skills/                   # LangChain Tools/Skills
│   │   ├── __init__.py
│   │   ├── market_news.py        # read_market_news
│   │   ├── ml_signals.py         # compute_ml_signals
│   │   ├── last_week_pnl.py      # check_last_week_pnl
│   │   ├── history_retrieval.py  # retrieve_history
│   │   ├── decision.py           # decide_positions
│   │   └── trade_execute.py      # execute_trade
│   ├── backtest/                 # 回测引擎
│   │   ├── engine.py             # 周粒度 WalkForward
│   │   ├── portfolio.py          # 持仓+收益计算
│   │   └── metrics.py            # 指标计算
│   ├── signals/                  # ML 信号层
│   │   ├── raw_scorer.py         # CPU/GPU 自适应
│   │   ├── knowledge_retrieval.py # TF-IDF 相似检索
│   │   └── weekly_returns.py     # 每周收益计算
│   └── utils/                    # 工具函数
│       ├── device.py              # CUDA/CPU 检测
│       ├── sentiment_cpu.py       # 关键词情感规则
│       ├── price_features.py       # ETF 价格特征
│       └── industry_map.py         # 行业→ETF 映射
├── data/                         # 原始数据（保留）
│   ├── converted/                 # tushare_news_*.parquet
│   └── 主题ETF历史量价.parquet     # ETF 价格
├── docs/
├── pyproject.toml
└── README.md
```

---

## 核心架构

### 周粒度 Walk-Forward + ReAct Agent

```
每周收盘后:
┌─────────────────────────────────────────────────────────────┐
│  Week N 输入                                               │
│  - 原始新闻 (tushare_news_*.parquet)                       │
│  - ETF 价格 (主题ETF历史量价.parquet)                       │
│  - ★ 上周持仓 + 收益率 (行为记忆)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  ReAct Agent (LLM 大脑)                                     │
│                                                             │
│  LLM: "先看新闻"  → read_market_news()                     │
│  LLM: "再看指标"  → compute_ml_signals()                    │
│  LLM: "看上周盈亏" → check_last_week_pnl()                 │
│  LLM: "查历史案例" → retrieve_history()                     │
│  LLM: "做决策"    → decide_positions()                     │
│                 → execute_trade()                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  回测引擎 → 计算周收益 → 存 weekly_returns.parquet          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                        Week N+1
```

---

## 关键设计决策

### 1. 完全新建，不基于旧代码
feat/agent 分支的设计思路保留，但代码完全重写。

### 2. GPU 解耦
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
无 GPU 时用规则替代 ML 模型，本地能测试。

### 3. Skill/Tool 化
每个能力是 `@tool` 装饰的 LangChain function，LLM 需要什么就 call 什么。

### 4. 训练/测试分离
```
训练期: 2021-01 ~ 2022-06  (调参用)
测试期: 2022-07 ~ 2023-12  (最终评估)
```

### 5. 行为金融记忆
Agent 决策时知道：上周持仓、收益率、赚钱/亏损状态 → 模拟"跟涨杀跌"心理。

---

## 实现步骤

### Step 1: 项目骨架 + GPU 解耦 (2h)
- 创建 `src/` 目录结构
- `src/utils/device.py` - CUDA/CPU 自动检测
- `src/utils/sentiment_cpu.py` - 关键词情感规则
- `src/utils/price_features.py` - ETF 价格特征
- `src/utils/industry_map.py` - 行业→ETF 映射
- **验证**：`python -c "from src.utils import *; print('OK')"` 无 GPU 也能过

### Step 2: 回测引擎骨架 (2h)
- `src/backtest/engine.py` - 周粒度 WalkForwardEngine
- `src/backtest/portfolio.py` - 持仓+周度收益
- `src/backtest/metrics.py` - Sharpe/Calmar/最大回撤
- **验证**：`python -c "from src.backtest import *; print('OK')"`

### Step 3: 行为记忆层 (1h)
- `src/signals/weekly_returns.py`
- 计算每周持仓行业的收益率
- **验证**：能读写 `data/weekly_returns.parquet`

### Step 4: 知识检索层 (3h)
- `src/signals/knowledge_retrieval.py`
- TF-IDF 对历史新闻建索引
- 给定日期返回相似历史案例
- **验证**：能返回相似新闻+当时市场反应

### Step 5: ML 信号层（GPU 可选） (2h)
- `src/signals/raw_scorer.py`
- CPU 模式：规则替代
- GPU 模式：LSTM/IForest/LightGBM
- **验证**：CPU 模式出结果

### Step 6: Skill 定义 (2h)
- `src/skills/market_news.py`
- `src/skills/ml_signals.py`
- `src/skills/last_week_pnl.py`
- `src/skills/history_retrieval.py`
- `src/skills/decision.py`
- `src/skills/trade_execute.py`
- **验证**：`from src.skills import *; print('OK')`

### Step 7: Agent ReAct 实现 (3h)
- `src/agent/state.py` - State 定义
- `src/agent/prompts.py` - ReAct 引导
- `src/agent/single_agent.py` - ReAct 循环
- `src/agent/workflow.py` - LangGraph workflow
- **验证**：`python -m src.agent.decide --week 2023-06-15` 单周能跑

### Step 8: 端到端回测 (2h)
- 完整周粒度回测
- 训练期/测试期分离
- **验证**：Sharpe、Calmar、最大回撤指标

---

## 验证方法

### 本地（无 GPU）
```bash
# 基础验证
python -c "from src.skills import *; from src.signals.raw_scorer import *; print('OK')"

# 单周 debug
python -m src.agent.decide --week 2023-06-15

# CPU 模式回测
python -m src.agent.backtest --train-end 2022-06-30 --test-start 2022-07-01 --end 2022-12-31
```

### 服务器（有 GPU）
```bash
# 全量训练期回测
python -m src.agent.backtest --train-end 2022-06-30

# 全量测试期回测
python -m src.agent.backtest --train-end 2022-06-30 --test-start 2022-07-01
```

### 指标标准

| 指标 | 合格 |
|---|---|
| Sharpe Ratio | > 1.0 |
| 最大回撤 | < 15% |
| Calmar Ratio | > 0.5 |
| 胜率 | > 50% |
| 总收益 | > Buy & Hold |

---

## 优势

1. **简洁**：完全新建，没有旧代码包袱
2. **端到端**：原始新闻+ETF → Agent 决策
3. **行为金融**：模拟人类"跟涨杀跌"心理
4. **GPU/CPU 自适应**：本地开发，服务器加速
5. **可解释**：每步 Skill 调用都有记录
6. **防过拟合**：训练/测试分离 + 周粒度 Walk-Forward
