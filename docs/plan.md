# Plan: LLM 驱动的行业轮动 + ETF 智能选择

## Context

用户确认：**LLM 做真正的行业判断，不是规则引擎**。规则只做 Guardrail。

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

## 三阶段数据划分

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
| 1 | `sentiment_ema` | EMA 平滑后的情感"存量"，α=0.2，防止信号断崖 |
| 2 | `sentiment_acceleration` | 情感变化的变化率（爆发点检测） |
| 3 | `sentiment_std` | 1 小时内情感标准差（共识 vs 多空博弈） |
| 4 | `log_news_count` | log(news_count + 1)，消除长尾，捕捉爆发 |
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

**Sentiment-Price Divergence**（博弈特征）：
```
sent_p_divergence = sentiment_zscore_5d - return_zscore_5d
> 0: 情绪跑赢价格 → 潜在补涨机会
< 0: 价格已兑现情绪 → 谨防利好出尽
```

**防止信息渗透（Critical）**：
- `sentiment_ma_6h` 在计算 T 时刻时，只包含 T-1 分钟之前的新闻
- **绝不**在特征里包含"当日收盘价"（当日收盘价只给 Agent 观察用，不是模型特征）
- **隔夜新闻标记**：`is_overnight_news` — 20:00-08:00 的新闻归到次日开盘前处理
- 每日 `build_agent_features` 在 **8:30** 执行，汇总隔夜情绪，作为开盘决策依据

---

## 核心设计决策

### 1. 为什么"分而复合"比"直接分类"更强？

**本质：语义解耦 + 策略重组**

**第一步（8 → 47）：感知提取**
- BERT 模型在区分"半导体"和"软件"时需要非常具体的上下文特征
- 只分大类会丢失这些微小的语义差异
- 47 个细分让感知层尽可能保留原始信息

**第二步（47 → 8）：逻辑路由**
- 例子：有 3 条"光刻机"新闻（半导体）+ 2 条"大模型"新闻（AI）→ 各自信号都弱
- 在"科技成长"元板块汇聚 → 5 条强信号，噪声被自然平滑
- 类似经验丰富的基金经理：观察 47 个细分行业的风吹草动，得出"下周看好科技成长"的结论

### 2. TCN 设计核心修正

**输入**：47 维细分行业情感时序 `(batch, seq_len=5, 47)`
**输出**：8 维元板块动量分数 `(batch, 8)`

```
TCN 的角色：学习 47 个细分之间的跨行业传导模式

例如：油价上涨 → 化工(周期资源) → 物流(交通运输) → 消费品(大消费)
TCN 需要学会捕捉这类跨行业传导链。
```

**为什么不用聚合后的 8 维输入？**
- 聚合会丢失细分间的协方差信息
- TCN 的价值正是从高维噪声中提取共享因子
- 直接用 8 维输入会让 TCN 退化为简单的时间序列模型，失去"跨行业学习"能力

### 3. 情感聚合：加权投票制（非简单平均）

**公式**：
```
元板块情感(t) = Σ(细分情感_i(t) × weight_i) / Σ weight_i
```

**权重设计原则**：
- 核心驱动力权重高（0.3-0.4）
- 辅助/边缘分类权重低（0.1-0.2）
- 例：半导体(0.4) + AI(0.4) + 软件(0.2) → 科技成长

```
元板块 ← 细分                     推荐权重   理由
科技成长 ← 半导体/AI               0.35     核心驱动力
科技成长 ← 软件/信创/云计算         0.25     辅助特征
科技成长 ← 电子/通信设备            0.20     边缘平滑
高端制造 ← 新能源/军工              0.35     核心驱动力
高端制造 ← 机器人/航空航天          0.35     核心驱动力
高端制造 ← 机械设备                0.30     辅助特征
大消费 ← 食品饮料/医药健康           0.30     顺周期核心
大消费 ← 旅游/文娱                  0.25     消费辅助
大消费 ← 家电/零售                  0.25     消费辅助
红利/中特估 ← 央企/国企/银行         0.35     核心
红利/中特估 ← 能源/基建              0.35     核心
红利/中特估 ← ESG                   0.30     辅助
金融地产 ← 非银金融/地产             0.45     核心
金融地产 ← 建筑                     0.35     政策博弈
周期资源 ← 有色金属/化工             0.35     大宗核心
周期资源 ← 钢铁/煤炭                0.35     大宗核心
周期资源 ← 石油石化                 0.30     辅助
智能网联 ← 新能车/汽车零部件         0.40     硬件核心
智能网联 ← 物联网                  0.35     软件核心
智能网联 ← 电子/通信设备            0.25     边缘平滑
区域经济 ← 长三角/大湾区/成渝        各0.33   平均
```

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
        {monday_date, agent_input, decision, weekly_return, guardrail_events},
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

```
- [ ] 新建 `trainer/src/utils/signals_xai.py`：SHAP 分析模块
  - `compute_shap_values(model, X_test)` → shap_values
  - `generate_summary_plot(shap_values, X_test, output_path)`
  - `generate_force_plot(shap_values, X_test, date, output_path)`
  - `export_shap_values(shap_values, dates)` → CSV for report
- [ ] 修改 `trainer/src/pipelines/train_signals.py`：训练完成后自动运行 SHAP 分析
- [ ] 修改回测报告生成逻辑：嵌入 SHAP 可视化图表
```

**注意**：SHAP 只在 Phase 2/3 决策日志中展示，不参与实际交易决策。

---

## 数据规格

### Phase 1 输出（已有）
```
每条新闻: {datetime, major_category, sub_category, sentiment_score, confidence}
```

### Phase 2 TCN 数据结构

**输入 X**: `(samples, seq_len=5, n_sub_sectors=47)` — 47 个细分行业情感时序
**输出 Y**: `(samples, n_meta_sectors=8)` — 8 个元板块动量分数 [-1, 1]

**标签构造**：
```python
# 动量标签：T+1 ~ T+5 日元板块情感均值相对 T 时刻的变化
meta_sentiment[t+5] = Σ(sub_sentiment_i[t+5] × weight_i)
momentum = clip((meta_sentiment[t+5] - meta_sentiment[t]) / (|meta_sentiment[t]| + 1e-9), -1, 1)
```

### Phase 2 数据结构

**情感时序（47 → 8 聚合后）**：
```
日期 | 元板块 | 情感均值 | 情感std | 新闻数量 | 平均置信度
```

**TCN 输出**：8 元板块 × 动量分数 [-1, 1]

**Isolation Forest 输出**：8 元板块 × 热度异常 [0, 1]

**LightGBM 输出**：8 元板块 × 综合信号（方向预测）

### Phase 3 LLM 输入结构

```
## 元板块综合信号（LightGBM 输出）
- 科技成长: 综合=0.65, 动量=0.42, 热度=0.65, 趋势=1
- 金融周期: 综合=-0.30, 动量=-0.15, 热度=0.30, 趋势=0
- 消费价值: 综合=0.20, 动量=0.10, 热度=0.40, 趋势=1
...

## 历史动量轨迹（TCN，近4周）
- 科技成长: [0.20, 0.30, 0.35, 0.42]  ← 持续上行
- 金融周期: [0.10, 0.05, -0.10, -0.15] ← 持续下行
...

## 高置信度新闻摘要（每元板块 top 1 条）
### 科技成长
- [positive] 0.92 | 半导体国产替代加速... | 降息, 银行, 存款
- [negative] 0.78 | 美国扩大芯片出口管制... | 半导体, 出口管制
...

## 当前持仓
上周持仓: 科技成长 30%, 消费价值 25%
上周收益: +2.3%
已投权重: 55%
```

### LLM 决策输出格式

```json
{
  "level1_plan": [
    {
      "meta_sector": "科技成长",
      "action": "buy",
      "weight_change": 0.15,
      "reason": "动量持续上行4周，新闻高置信度正面，国产替代政策持续加码"
    },
    {
      "meta_sector": "金融周期",
      "action": "sell",
      "weight_change": -0.10,
      "reason": "动量连续下行，降息预期消化充分"
    },
    {
      "meta_sector": "消费价值",
      "action": "hold",
      "weight_change": 0.0,
      "reason": "动量平稳，新闻无显著方向"
    }
  ],
  "level2_plan": [
    {
      "meta_sector": "科技成长",
      "selected_etf_1": {"code": "159805", "name": "芯片ETF", "tracking_index": "中华半导体芯片指数", "aum": 45.2, "tracking_error": 0.02},
      "selected_etf_2": {"code": "512760", "name": "芯片ETF", "tracking_index": "费城半导体指数", "aum": 32.1, "tracking_error": 0.03}
    }
  ],
  "reasoning_summary": "科技成长动量持续4周上行，今日国产替代政策利好，符合预期差买入逻辑..."
}
```

---

## 8 元板块正式定义

| 元板块 | 代码 | 包含核心细分 | Beta | 驱动逻辑 |
|--------|------|-------------|------|---------|
| 科技成长 | Alpha Tech | 半导体、人工智能、计算机软件、云计算 | Very High | TMT 核心，对流动性和情绪最敏感 |
| 高端制造 | Hard Tech | 军工、新能源、机器人、航空航天 | Very High | 政策驱动 + 制造业景气度 |
| 大消费 | Consumption | 食品饮料、医药健康、旅游、文娱 | Medium | 顺周期指标，受内需和人口逻辑驱动 |
| 红利/中特估 | Value/SOE | 央企/国企、银行、能源、基建、ESG | Low | 避险属性，受利率环境和分红率影响 |
| 金融地产 | Financial | 非银金融、地产、建筑 | Low/Med | 强政策博弈，杠杆率和信用周期的风向标 |
| 周期资源 | Resources | 化工、有色金属、钢铁、煤炭 | Medium | 全球大宗商品价格 + 通胀预期 |
| 智能网联 | Smart Mobility | 新能车、物联网、汽车零部件 | High | 跨行业硬件+软件结合部 |
| 区域经济 | Regional | 长三角、大湾区、成渝等 | Medium | 宏观叙事，通常作为防守或特定政策观察点 |

---

## 实现步骤

### Phase 1: 模型训练基础设施
- [x] `data/industry_dict.json`（已存在）
- [ ] 新建 `data/meta_sector_mapping.json`（47 细分 → 8 元板块 + 权重）
- [ ] 修改 `trainer/src/datasets/signals.py`：
  - 输入改为 47 维细分情感时序，6 通道
  - 输出改为 8 维元板块动量标签
  - **新增 `export_phase2_dataset()`**：导出每日特征用于 Agent 训练
- [ ] 训练 TCN（47 → 8 维，学习跨行业传导）
- [ ] 训练 IForest + LightGBM
- [ ] 新增 `global_leader_sentiment` 跨行业传导特征（滞后相关性）
- [ ] 新增 `market_beta` Beta敏感度特征（滚动20日相关性）
- [ ] 新建 `trainer/src/utils/signals_xai.py`：SHAP 分析模块（训练完成后自动运行）
- [ ] ONNX 导出（TCN + IForest + LightGBM）

### Phase 2: Agent 训练系统
- [ ] 新建 `src/agent/features.py`：`build_agent_features()` 构建 A/B/C/D/E 五类特征
- [ ] 新建 `src/agent/decision_logger.py`：记录每周决策 + 实际收益 + Guardrail 事件
- [ ] 修改 `decide_node`：
  - 改为**每周一决策**模式
  - Few-shot Prompt 注入好的/坏的决策 pattern
  - 决策前先过"决策逻辑检查清单"
- [ ] 新建 `src/agent/daily_guardrail.py`：日频监控紧急退出逻辑
- [ ] Phase 2 Dry Run：运行 2024-10 ~ 2025-06 **每周**回测 + 日 Guardrail
- [ ] 分析 decision_log，提取 good/bad patterns，每 4 周更新一次 Prompt
- [ ] Phase 2 末 4 周验证集评估

### Phase 3: 最终回测
- [ ] 在 Phase 3 数据上运行完整 pipeline
- [ ] 计算夏普比率、最大回撤、胜率等指标
- [ ] 对比"Phase 2 调优 Prompt"vs"未调优 Prompt"的表现差异

---

## 关键文件改动

| 文件 | 改动 |
|------|------|
| `data/meta_sector_mapping.json` | **新建**：47 细分 → 8 元板块 + 权重 |
| `trainer/src/datasets/signals.py` | 重构 TCN 数据集（47 维输入，8 维输出，6 通道）；**新增 `export_phase2_dataset()`** |
| `trainer/src/utils/signals_xai.py` | **新建**：SHAP 分析模块（TreeExplainer、Summary/Force/Dependence Plot） |
| `src/agent/features.py` | **新建**：`build_agent_features()` 构建 A/B/C/D/E 五类特征 |
| `src/agent/decision_logger.py` | **新建**：记录每周决策 + 实际收益 + Guardrail 事件 + TCN_Prediction_Error |
| `src/agent/daily_guardrail.py` | **新建**：日频紧急退出监控逻辑 + FORBIDDEN_ZONE 状态机 |
| `src/agent/tools.py` | 新增 `build_decision_context`（调用 `features.py`）|
| `src/agent/single_agent.py` | 重构 `decide_node`：改为**每周一决策**模式 |
| `src/agent/rule_engine.py` | 新建 Guardrail 规则引擎（周决策 + 日监控双模式）|
| `src/agent/state.py` | `TradeDecision` 支持 `level1_plan`, `level2_plan`；新增 `SectorStatus` 枚举 |
| `config/prompts/trader.md` | 更新 Prompt（加入 good/bad patterns，动态 Few-shot，周决策模式）|

---

## 验证方式

### 指标 1：信号连续性
对比"半导体"细分情感曲线 vs "科技成长"元板块情感曲线。
- **预期**：元板块曲线更平滑，NaN 更少
- **意义**：加权聚合确实在平滑噪声

### 指标 2：IC (Information Coefficient) 提升
计算"元板块情感动量"与"该板块下所有 ETF 平均收益率"的相关性。
- **预期**：聚合后的 IC > 单一细分的 IC
- **意义**：降维聚合确实提取了更稳定的预测信号

### 指标 3：计算稳定性
观察 LightGBM 特征重要性排序。
- **预期**："元板块信号"能排进 Top 5
- **意义**：降维后的特征确实抓住了市场核心逻辑

### 指标 4：Phase 2 决策质量追踪
- 每周记录 `signal_alignment * actual_return` 的相关性
- 预期：调优后 Prompt 的相关性 > 调优前

### 指标 5：Phase 3 回测指标
- 夏普比率 > 1.0
- 最大回撤 < 15%
- 与等权基准对比：超额收益 > 5%

### 指标 6：Agent 冷启动验证
- Phase 2 首 2 周 warm-up 期的决策分布
- 预期：warm-up 期决策较保守（低权重、高 HOLD 比例）

### 指标 7：XAI 可解释性验证
- SHAP Summary Plot 特征重要性是否稳定（Top 3 特征不剧烈波动）
- 每周 Force Plot 决策逻辑是否与 Prompt 决策清单一致
- 预期：SHAP 贡献度 Top 特征与人工逻辑判断的准确率 > 70%
