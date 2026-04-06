# Plan Update: 三阶段训练架构 + 双模式决策系统

## Context

原 plan.md 存在以下核心问题：

1. **数据划分不完整** - 只有 train/test，没有为 Agent 单独设立"决策学习期"
2. **Agent 特征不足** - 只有信号输出，缺少市场状态特征和历史回溯特征
3. **Agent 训练方法缺失** - 只说"重构 decide_node"，没说如何让 LLM 学习决策逻辑
4. **冷启动问题未处理** - Agent 首周无历史持仓/收益反馈
5. **交易成本未嵌入** - guardrail 提了 0.1% 但未在特征层面体现
6. **决策频率模糊** - 隐含每日决策，但 A 股周频调仓更合理

---

## 核心架构：双模式决策

```
每周一（Agent 决策日）
    │
    ▼
┌─────────────────────────────────────────────┐
│  Agent 层（Level 1 + Level 2）               │
│  输入: TCN 日频序列（过去 5 天）+ 新闻摘要    │
│  输出: 本周仓位配置计划（8 元板块权重）       │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  周内持仓期（Hold）                         │
│  每日 Guardrail 监控                        │
│  触发条件 → 紧急平仓                        │
└─────────────────────────────────────────────┘
```

**为什么周决策更合理**：
- 新闻情绪（尤其政策类）有 3-5 天横向发酵期，日频调仓是噪音
- TCN 动量在日频波动极大，周频（5 日均值）形成更清晰趋势线
- LLMs 每周一次决策能站在更宏观视角看"本周核心叙事"

**为什么需要日监控**：
- 持仓期间可能出现突发利空（>5% 止损线、监管黑天鹅）
- Guardrail 只做"紧急退出"，不参与正常仓位调整

---

## 三阶段数据划分（核心改动）

```
时间线：2023-01 ───────────────────────────────────── 2026-03
        │←────────────────────────→│←───────────────→│
        Phase 1 (60%)               Phase 2 (20%)     Phase 3 (20%)
        模型训练集                   Agent 训练/验证集  最终回测集
```

| 阶段 | 时间范围 | 用途 | 目标 |
|------|---------|------|------|
| **Phase 1** | 2023-01 ~ 2024-09（60%）| 训练 TCN + LightGBM | 模型学会识别"每日情感"→"未来 5 日收益"的映射 |
| **Phase 2** | 2024-10 ~ 2025-06（20%）| Agent 决策学习（Dry Run） | Agent 每周一做深度复盘，学习如何根据信号序列做判断 |
| **Phase 3** | 2025-07 ~ 2026-03（20%）| 最终回测 | 模型和 Agent 都没见过，验证泛化能力 |

**关键设计**：
- Phase 2 首 2 周 = **热身期（Warm-up）**，Agent 只观察不交易
- Phase 2 末 4 周 = **Agent 验证集**（调优 Prompt，但不用来训练）

---

## Phase 1 → Phase 2 数据传递

Phase 1 产出的模型（TCN + LightGBM）在 Phase 2 期间**冻结权重**，只做推理：

```python
# Phase 2 每周流程
for each Monday in Phase 2:
    # 1-6. 模型推理（每日执行，结果缓存）
    sub_sentiment = aggregate(news, by_sub_category)           # 47 细分
    meta_sentiment = weighted_vote(sub_sentiment)               # 8 元板块
    tcn_momentum = tcn_model.predict(meta_sentiment)           # 8 维动量
    lgbm_signal = lgbm_model.predict(tcn_momentum, market_state) # 8 维综合信号
    iforest_heat = iforest_model.predict(meta_sentiment)       # 8 维热度异常

    # 7. 构成本周 Agent 输入（TCN 日频序列 + 新闻摘要）
    #    TCN 序列: shape (5, 8) = 过去 5 个交易日的 8 维动量
    agent_input = build_agent_features(tcn_daily_sequence, lgbm_signal, ...)

    # 8. Agent 决策（每周一执行）
    weekly_plan = agent_llm.decide(agent_input)

    # 9. 执行并记录
    execute(weekly_plan)

    # 10. 周内每日 Guardrail 监控
    for each weekday in this_week:
        guardrail_signal = check_daily_guardrail(positions, news, prices)
        if guardrail_signal.triggered:
            emergency_exit(guardrail_signal.targets)

    # 11. 周末记录决策日志（用于 Prompt 调优）
    log_decision(weekly_plan, actual_outcome, guardrail_events)
```

**关键设计**：
- TCN 输入是 **5 天序列**（不是周聚合），让 LLM 看到趋势线
- Agent **每周一决策一次**，周内不调整
- Guardrail **每日检查**，只触发紧急退出，不做主动加仓

---

## TCN 输入序列结构（47 维时序，每行 6 通道）

每行（每个时间步）包含以下 6 个特征通道：

| 通道 | 字段 | 说明 |
|------|------|------|
| 1 | `sentiment_ema` | EMA 平滑后的情感"存量"，$\alpha=0.2$，防止信号断崖 |
| 2 | `sentiment_acceleration` | 情感变化的变化率（爆发点检测） |
| 3 | `sentiment_std` | 1 小时内情感标准差（共识 vs 多空博弈） |
| 4 | `log_news_count` | $\log(\text{news\_count} + 1)$，消除长尾，捕捉爆发 |
| 5 | `event_type_embedding` | 事件类型 One-hot：政策/业绩/技术/其他 |
| 6 | `sentiment_vs_price_residual` | 情感 Z-score - 价格 Z-score，背离度检测 |

**EMA 平滑公式**（防止信号断崖）：
```
S_now = 0.2 * S_new + 0.8 * S_yesterday
```

**TCN 标签构造**：
```
目标: T+1 ~ T+5 日的元板块价格动量（均值）
标签 = clip((meta_sentiment[t+5] - meta_sentiment[t]) / (|meta_sentiment[t]| + 1e-9), -1, 1)
```

**TCN 架构：扇入式 47→8 映射（Critical）**：
```
输入：  (batch, 47, time_steps, 6)   # 47 个细分行业 × 6 通道
        ↓
中间层：TCN stack（kernel_size=3, dilation=[1,2,4,8]）
        ↓
展平层：Flatten(47 * hidden) → Linear(47 * hidden, 128)
        ↓
输出层：Linear(128, 8)              # 8 个元板块动量
```

**为什么用扇入（Fan-in）结构**：
- 强迫模型在预测"科技成长"时，必须同时看到"半导体材料+光刻机+消费电子+..."的联合信号
- 比先聚类再进 TCN 更有利于保留原始信号的微小差异
- 跨行业关联被显式学习，而不是被固定权重叠加掩盖

**Sentiment-Price Divergence**（博弈特征）：
```
sent_p_divergence = sentiment_zscore_5d - return_zscore_5d
> 0: 情绪跑赢价格 → 潜在补涨机会
< 0: 价格已兑现情绪 → 谨防利好出尽
```

**⚠️ 防止信息渗透（Critical）**：
- `sentiment_ma_6h` 在计算 $T$ 时刻时，只包含 $T-1$ 分钟之前的新闻
- **绝不**在特征里包含"当日收盘价"（当日收盘价只给 Agent 观察用，不是模型特征）
- **隔夜新闻标记**：`is_overnight_news` — 20:00-08:00 的新闻归到次日开盘前处理
- 每日 `build_agent_features` 在 **8:30** 执行，汇总隔夜情绪，作为开盘决策依据

---

## Agent 特征工程（周决策输入）

### 每周一更新一次的特征

**A. TCN 日频序列（核心输入）**
```
tcn_sequence[5, 8]: 过去 5 个交易日 × 8 元板块动量分数
  例如：
  科技成长: [0.20, 0.28, 0.35, 0.42, 0.51]  ← 持续上行
  金融周期: [0.10, 0.05, -0.10, -0.15, -0.20] ← 持续下行
```
让 LLM 看到**趋势线**，而不是单点值。

**B. 本周新闻摘要（每周聚合）**
```
每元板块 top-1 条最重要新闻（按置信度 × 动量 加权），格式：
### 科技成长
- [positive] 0.92 | 半导体国产替代加速... | 关键词: 降息, 银行, 存款
- [negative] 0.78 | 美国扩大芯片出口管制... | 关键词: 半导体, 出口管制
```

**C. 市场状态（每周一更新）**
```
- vol_percentile: 市场整体波动率分位（过去 52 周）
- iforest_heat[8]: 上周热度异常
- price_momentum_1w[8]: 近 1 周价格动量（用于判断是否"已充分定价"）
- sentiment_entropy[8]: 各元板块情感混乱程度（标准差/熵）
- news_dominance_ratio[8]: 该板块新闻量 / 全市场新闻量（聚光灯效应）
```

**D. 持仓状态（每周一更新）**
```
- position_size[8]: 当前各元板块持仓权重
- agent_performance_1w[8]: 上周 Agent 在各板块的决策收益率
- agent_performance_4w[8]: 近 4 周累计收益率
```

**E. 量价博弈特征（Agent 专属）**
```
sent_p_divergence[8]: 情绪 Z-score - 价格 Z-score
> 0: 情绪跑赢价格 → 预期差买点
< 0: 价格已兑现情绪 → 利好出尽/情绪滞后
```

### 决策逻辑检查清单（让 LLM 每周自审）

```
1. 【趋势判断】过去 5 天，哪些板块动量在持续改善？哪些在恶化？
2. 【预期差识别】TCN 情感极好（> 0.6）但价格已连涨 3 天 → "情绪已充分定价，谨防利好出尽"
3. 【关联性回避】要加仓的板块，和现有持仓是否同 correlation_cluster？
4. 【波动率自适应】当前 vol_percentile > 0.8 → "市场高波动，降低 Beta 暴露"
5. 【止损回顾】上周亏损的板块，信号是否已经反转？还是应该止损？
```

---

## Agent 训练方法（周频 Dry Run 模式）

### 核心思想
Agent 的目标不是**预测价格**，而是优化**风险调整收益（夏普比率）**。
Agent 每周一做一次**深度复盘**，而不是每日决策。

### 周频决策 Prompt 模板

```python
SYSTEM_PROMPT = """
你是行业轮动交易 Agent，运行于 A 股市场。
你的决策频率：每周一（周一开盘前）做一次仓位计划，周内不主动调仓。

## 你的输入（每周一更新）
1. TCN 动量日频序列：过去 5 个交易日 8 个元板块的动量变化
2. 综合信号：LightGBM 给出的方向预测 + 信号强度
3. 热度异常：IForest 检测到的新闻量异常
4. 本周核心新闻摘要：每元板块 top-1 条（最重要的）
5. 当前持仓状态：各元板块权重 + 上周收益率

## 你的决策逻辑（每周复盘）
在给出本周仓位计划前，先思考：
1. 【趋势判断】过去 5 天，哪些板块动量在持续改善？哪些在恶化？
2. 【预期差识别】TCN 分数很高（>0.6）但价格已连涨 3 天吗？→ 谨防利好出尽
3. 【关联性回避】本周要加仓的板块，和现有持仓是否同 correlation_cluster？
4. 【波动率自适应】当前市场波动率分位如何？高波动时降低 Beta 暴露
5. 【新闻叙事】本周最重要的宏观/政策叙事是什么？哪个板块最受益/最受损？

## 输出格式
仅输出以下 JSON（不输出分析过程）：
{
  "level1_plan": [...],  // 8 个元板块的仓位调整计划
  "level2_plan": [...],  // 每个元板块的 ETF 推荐
  "reasoning_summary": "..."  // 上述 5 点思考的简要总结（用于日志分析）
}
"""
```

### Few-shot Prompt 调优流程

```
Step 1: Phase 2 每周记录决策日志
    decision_log = [
        {monday_date, agent_input, decision, weekly_return, guardrail_events,
         tcn_prediction_error: {  # 新增：让 Agent 学会"质疑模型"
           meta_sector: "科技成长",
           tcn_predicted: 0.85,    # TCN 给出的动量分数
           actual_return: -0.04,   # 实际收益
           divergence: 0.89,       # |predicted - actual|
           root_cause_guess: "利好出尽 / 黑天鹅 / 关联板块拖累 / ..."  # LLM 复盘时自己填
         },
         ...
        ]

Step 2: 标注"决策质量"（每周评估一次）
    for each week in Phase 2:
        if weekly_return > 0 AND signal_alignment > 0.3:
            quality = "good"
        elif weekly_return < -2% AND signal_alignment < -0.3:
            quality = "good"  # 信号错误但方向判断对（不可避免）
        else:
            quality = "poor"  # 信号强但实际亏损 → 需要反思

Step 3: 提取 good/bad patterns（用于更新 Few-shot）
    good_patterns = [
        "科技成长：TCN 动量持续上行 3 周，新闻出现国产替代政策 → 建议加仓 15%",
        "红利/中特估：市场波动率分位 > 0.8 → 建议标配，作为防御",
    ]
    bad_patterns = [
        "高端制造：信号极强但上周已涨 8% → 本周追高被套（教训：预期差）",
        "金融地产：买了地产 + 建筑，同 correlation_cluster → 风险重叠亏损",
    ]

Step 4: 将 patterns 注入 Prompt（每 4 周更新一次）
```

### 决策逻辑检查清单（让 LLM 自我审查）

每次决策前，LLM 需要检查：

1. **预期差识别**：TCN 情感极好（> 0.6）但价格已连涨 3 天 → "情绪已充分定价，谨防利好出尽"
2. **关联性回避**：已持有某板块，当另一个同 correlation_cluster 板块出现信号时 → "风险重叠，谨慎加仓"
3. **波动率自适应**：当前 vol_percentile > 0.8 → "市场高波动，降低 Beta 暴露"
4. **趋势延续性**：price_momentum_4w 持续上行 → "趋势强劲，可适度追涨"
5. **止损检查**：recent_drawdown_3d < -5% → "超跌，观察是否见底"
6. **模型质疑**（Phase 2 复盘）：本周 TCN_Prediction_Error 较大的板块 → "这个板块上周信号失效，是因为突发黑天鹅，还是模型过热？本周围绕该板块的决策是否需要更保守？"

---

## 交易成本约束

在 Agent 决策逻辑中嵌入：

```
最小操作阈值：|weight_change| < 0.05 → 降为 HOLD（手续费不划算）

双边摩擦：0.1%（买入 0.05% + 卖出 0.05%）

真实组合权重 = Agent 建议权重 × min(1.0, 0.10 / |weight_change|)
# 例：建议加仓 3%，实际只能加 3% × (0.10/3%) = 10% → 触发最小阈值，降为 HOLD
```

---

## Guardrail 规则（周决策 + 日监控）

### 周一决策时（Weekly Guardrail）
| 规则 | 说明 |
|------|------|
| 单行业权重上限 30% | `rule_engine.py` |
| 总权重上限 100% | `rule_engine.py` |
| Beta 惩罚 | very_high beta ×0.7；high beta ×0.85 |
| Mirror 检查 | 同 correlation_cluster 两个板块不能同时 ≥ 15% |
| 亏损保护 | 上周 return < 0 时禁止新建 very_high beta 仓位 |
| 最小操作阈值 | \|weight_change\| < 5% → 降为 HOLD |

### 周内每日监控（Daily Guardrail）
| 触发条件 | 动作 |
|---------|------|
| 单只 ETF 当日跌幅 > 5% | 紧急平仓该 ETF |
| 单只 ETF 当日跌幅 > 3% 且 Beta = very_high | 紧急平仓该 ETF |
| 突发重大利空新闻（置信度 > 0.95 且 sentiment < -0.8）| 紧急平仓相关板块 |
| 全市场波动率单日飙升 > 3σ | 降仓至 50%，等周末复盘 |

**注意**：日 Guardrail 只做"紧急退出"，不参与正常仓位调整。

---

## 跨行业传导：滞后相关性替代 GNN

**为什么不用 GNN**：
- GNN 训练极其不稳定，A 股行业联动（如：煤炭→火电→绿电）的逻辑随政策频繁变动
- 周频/日频样本量有限，GNN 容易过拟合

**替代方案：手动特征交叉（滞后相关性）**

在 LightGBM 输入中，给每个元板块增加 **Global_Leader_Sentiment** 特征：

| 元板块 | 传导信号来源 | 逻辑 |
|--------|-------------|------|
| 高端制造 | 有色金属/能源情感 | 上游资源涨价 → 制造成本压力 |
| 科技成长 | 半导体情感 | 硬件→软件联动，景气度传导 |
| 大消费 | 医药健康情感 | 防御板块情感对消费有领先性 |
| 红利/中特估 | 银行情感 | 利率预期传导链 |
| 金融地产 | 建筑情感 | 基建链前端 |
| 周期资源 | 石油石化情感 | 全球定价锚定 |
| 智能网联 | 新能源情感 | 能源转型主线 |
| 区域经济 | 基建情感 | 政策驱动型传导 |

**计算方式**：
```python
# 滞后相关性：取前 5 日的领导板块情感均值作为从属板块的输入
global_leader_sentiment[sector][t] = mean(sentiment[leader_sector][t-5:t])
```

**嵌入位置**：LightGBM 输入 X 的新增通道，随 TCN 输出 + 市场状态特征一同输入。

---

## XAI 模块：SHAP 可解释性分析

**目标**：拆解黑盒 LightGBM，增强金融决策透明度

### SHAP 分析流程

```python
import shap

# LightGBM 训练完成后
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_lgbm_test)  # shape: (n_samples, n_features, n_classes)

# 每周决策时：解释为什么看好/看空某板块
shap_summary = {
    "科技成长": {
        "shap_values": shap_values[week_idx, :, sector_idx],
        "top_positive_features": ["sentiment_acceleration", "iforest_heat"],
        "top_negative_features": ["price_momentum_1w", "sentiment_std"]
    }
}
```

### 特征贡献分解

| 特征通道 | SHAP 贡献解读 |
|---------|--------------|
| `sentiment_ema` | 情感存量贡献，正向→加仓，负向→减仓 |
| `sentiment_acceleration` | 爆发点识别，贡献大→趋势加速确认 |
| `sentiment_std` | 分歧度，高→多空博弈预警 |
| `sentiment_vs_price_residual` | 预期差，>0→补涨机会，<0→利好出尽 |
| `global_leader_sentiment` | 跨行业传导贡献，验证逻辑链 |
| `market_beta` | Beta敏感度，泥沙俱下行情中权重降低 |

### 回测报告嵌入

每期回测报告生成以下 SHAP 可视化：

1. **SHAP Summary Plot**：展示所有特征对 LightGBM 预测的平均绝对贡献
2. **SHAP Force Plot（每周）**：解释本周决策的核心驱动因素
3. **SHAP Dependence Plot**：展示 `sentiment_acceleration` 与预测输出的非线性关系
4. **特征重要性时序图**：展示过去 4 周哪些特征持续贡献Alpha

### 实现步骤

```python
# Phase 1 末新增
- [ ] 新建 `trainer/src/utils/signals_xai.py`：SHAP 分析模块
  - `compute_shap_values(model, X_test)` → shap_values
  - `generate_summary_plot(shap_values, X_test, output_path)`
  - `generate_force_plot(shap_values, X_test, date, output_path)`
  - `export_shap_values(shap_values, dates)` → CSV for report
- [ ] 修改 `trainer/src/pipelines/train_signals.py`：训练完成后自动运行 SHAP 分析
- [ ] 修改 `src/backtest/engine.py`：在回测报告中嵌入 SHAP 可视化
```

**注意**：SHAP 只在 Phase 2/3 决策日志中展示，不参与实际交易决策。

---

## 市场Beta敏感度（Dynamic Beta）

**计算方式**：
```python
# 滚动 20 日行业 vs 沪深300（或中证全指）收益率相关性
market_beta[sector][t] = rolling_correlation(
    returns[sector][t-20:t],
    returns[index][t-20:t],
    window=20
)
```

**LightGBM 输入**：每元板块 1 维 Beta 值（标准化后）

**决策逻辑嵌入**：
| Beta 区间 | 市场状态 | LightGBM 信号权重调整 |
|-----------|---------|----------------------|
| < 0.3 | 结构性行情（行业分化） | 信号权重 ×1.0 |
| 0.3~0.6 | 平衡市场 | 信号权重 ×0.85 |
| > 0.6 | 泥沙俱下（系统性风险） | 信号权重 ×0.7，触发 Guardrail 降仓提示 |

**为什么需要这个特征**：
- A 股经常出现"大盘崩了再好的行业逻辑也带不动"的情况
- Beta 高的行业在 Guardrail 触发前就先对 LightGBM 信号做折扣

---

## 完整 LightGBM 输入特征清单（更新后共16维）

| # | 特征名 | 来源 | 说明 |
|---|--------|------|------|
| 1 | `delta_sentiment` | TCN输出 | 8元板块情感变化 |
| 2 | `news_count` | 新闻聚合 | 8元板块新闻数量 |
| 3 | `tcn_reg` | TCN回归输出 | 8维动量预测 |
| 4 | `tcn_cls` | TCN分类输出 | 8维方向预测 |
| 5 | `volume_ratio` | OHLCV | 量比（当日/均值） |
| 6 | `intraday_vol` | OHLCV | 日内波动率 |
| 7 | `iforest_heat` | IForest | 8维热度异常 |
| 8 | `price_momentum_1w` | 价格 | 近1周动量 |
| 9 | `price_momentum_4w` | 价格 | 近4周动量 |
| 10 | `sentiment_std` | 新闻聚合 | 情感分歧度 |
| 11 | `sentiment_vs_price_residual` | 计算 | 预期差特征 |
| 12 | `global_leader_sentiment` | 手动特征 | 跨行业传导特征 |
| 13 | `market_beta` | 价格计算 | Beta敏感度 |
| 14 | `vol_percentile` | 价格 | 52周波动率分位 |
| 15 | `sentiment_entropy` | 新闻聚合 | 情感熵 |
| 16 | `news_dominance_ratio` | 新闻聚合 | 聚光灯效应 |

**注意**：Phase 1 训练 TCN+IForest+LightGBM 时先不上 Beta/Lagged Correlation，等 Phase 2 数据积累后再补充；Phase 1 主要验证 TCN 的跨行业学习能力。

---

## Guardrail 规则（周决策 + 日监控）

### 周一决策时（Weekly Guardrail）

### Guardrail 优先级覆盖（Critical）

**规则冲突处理原则**：Daily Guardrail 的优先级 **必须高于** Agent Level 1 计划。一旦触发，直到下周一 Agent 重新决策前，该板块标记为 `FORBIDDEN_ZONE`。

```python
# FORBIDDEN_ZONE 状态机
class SectorStatus(Enum):
    NORMAL = "normal"
    FORBIDDEN_ZONE = "forbidden"   # 日 Guardrail 触发，平仓后封禁

# 日 Guardrail 触发后
if guardrail_triggered(sector):
    emergency_exit(sector)
    sector_status[sector] = SectorStatus.FORBIDDEN_ZONE
    forbidden_until[sector] = next_monday   # 下周一之前禁止重建

# Agent 周一决策时
for sector in FORBIDDEN_ZONE:
    # Agent 的 Level 1 计划中，该板块权重强制降为 0
    agent_plan[sector] = 0
    agent_plan[sector] += f"[FORBIDDEN_ZONE overridden: {reason}]"
```

**为什么这样设计**：
- 防止 Agent 在周内"反复申购"同一板块（情绪化操作）
- 突发利空需要 3-5 天消化期，强行持有只会放大亏损
- 保留 `reason` 字段供复盘：LLM 需要知道"为什么被禁止"，才能在下次避免类似情况

---

## 实现步骤更新

### Phase 1: 模型训练基础设施
- [x] `data/industry_dict.json`（已存在）
- [ ] 新建 `data/meta_sector_mapping.json`（47 细分 → 8 元板块 + 权重）
- [ ] 修改 `trainer/src/datasets/signals.py`：
  - 输入改为 47 维细分情感时序
  - 输出改为 8 维元板块动量标签
  - **新增 Phase 2 数据导出模式**：`export_phase2_dataset()` 导出每日特征用于 Agent 训练
- [ ] 训练 TCN（47 → 8 维，学习跨行业传导）
- [ ] 训练 IForest + LightGBM
- [ ] ONNX 导出

### Phase 2: Agent 训练系统
- [ ] 新建 `src/agent/features.py`：`build_agent_features()` 构建 A/B/C/D/E 五类特征
- [ ] 新建 `src/agent/decision_logger.py`：记录每周决策 + 实际收益 + Guardrail 事件
- [ ] 修改 `decide_node`：
  - 改为**每周一决策**模式（不是每日）
  - Few-shot Prompt 注入好的/坏的决策 pattern
  - 决策前先过"决策逻辑检查清单"
- [ ] 新建 `src/agent/daily_guardrail.py`：日频监控紧急退出逻辑
- [ ] Phase 2 Dry Run：运行 2024-10 ~ 2025-06 **每周**回测 + 日 Guardrail
- [ ] 分析 decision_log，提取 good/bad patterns，每 4 周更新一次 Prompt
- [ ] Phase 2 末 4 周验证集评估

### Phase 3: 最终回测（不变）
- [ ] 在 Phase 3 数据上运行完整 pipeline
- [ ] 计算夏普比率、最大回撤、胜率等指标
- [ ] 对比"Phase 2 调优 Prompt"vs"未调优 Prompt"的表现差异

---

## 关键文件改动

| 文件 | 改动 |
|------|------|
| `data/meta_sector_mapping.json` | **新建**：47 细分 → 8 元板块 + 权重 |
| `trainer/src/datasets/signals.py` | 重构 TCN 数据集（47 维输入，8 维输出，6 通道）；**新增 `export_phase2_dataset()`** |
| `src/agent/features.py` | **新建**：`build_agent_features()` 构建 A/B/C/D/E 五类特征 |
| `src/agent/decision_logger.py` | **新建**：记录每周决策 + 实际收益 + Guardrail 事件 + **TCN_Prediction_Error** |
| `src/agent/daily_guardrail.py` | **新建**：日频紧急退出监控逻辑 + **FORBIDDEN_ZONE 状态机** |
| `src/agent/state.py` | 新增 `SectorStatus` 枚举（NORMAL / FORBIDDEN_ZONE） |
| `src/agent/tools.py` | 新增 `build_decision_context`（调用 `features.py`）|
| `src/agent/single_agent.py` | 重构 `decide_node`：改为**每周一决策**模式 |
| `src/agent/rule_engine.py` | 新建 Guardrail 规则引擎（周决策 + 日监控双模式）|
| `src/agent/state.py` | `TradeDecision` 支持 `level1_plan`, `level2_plan` |
| `config/prompts/trader.md` | 更新 Prompt（加入 good/bad patterns，动态 Few-shot，周决策模式）|

---

## 验证方式

1. **Phase 2 决策质量追踪**：
   - 每周记录 `signal_alignment * actual_return` 的相关性
   - 预期：调优后 Prompt 的相关性 > 调优前

2. **Phase 3 回测指标**：
   - 夏普比率 > 1.0
   - 最大回撤 < 15%
   - 与等权基准对比：超额收益 > 5%

3. **Agent 冷启动验证**：
   - Phase 2 首 2 周 warm-up 期的决策分布
   - 预期：warm-up 期决策较保守（低权重、高 HOLD 比例）
