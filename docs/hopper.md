# 实施计划：8 元板块决策架构收敛版

## Context

当前代码库已经同时包含训练链路和 Agent 链路的初步改造，因此这份文档不再按“新建 19 个文件”描述，而是聚焦三件事：

1. 明确最终目标架构
2. 冻结不会再反复变动的训练/特征规格
3. 给出可执行的落地顺序，避免一次改太多导致联调失控

本次改动的核心目标仍然不变：

- 决策层从“按细分行业交易”切换为“按 8 个元板块做 Level 1 决策，再做 Level 2 ETF 选择”
- 训练层从旧的 per-industry TCN 迁移到 fan-in TCN
- 解释层增加 SHAP，但 SHAP 只用于分析和报告，不参与交易

---

## 最终目标架构

### 训练层

- 输入来源仍是新闻情感聚合结果
- SetFit 负责 47 个细分类别
- TCNFanIn 负责把 47 个细分映射到 8 个元板块动量
- 8 个独立 LightGBM 负责每个元板块的二阶段判别
- IForest 负责新闻热度异常检测
- `export_phase2_dataset()` 负责批量导出 Agent 所需日频特征

### 决策层

- Level 1: Agent 输出 `level1_plan[]`，决定元板块的 buy/sell/hold 和权重
- Level 2: Agent 输出 `level2_plan[]`，决定该元板块下的指数和 ETF 选择
- `risk_check_node` 负责统一约束检查
- `daily_guardrail` 负责日内或日频带外风控覆盖

### 解释层

- 训练后运行 SHAP 分析
- 输出 summary plot、per-date force plot、`shap_values.csv`
- SHAP 不进入实时决策，只用于诊断 LightGBM 是否过度依赖 TCN 特征

---

## 冻结规格

### 1. TCN 输入输出

```text
输入:
  shape: (batch, seq_len=5, n_sub=47, channels=6)

6 通道:
  ch0: sentiment_ema
  ch1: sentiment_acceleration
  ch2: sentiment_std
  ch3: log_news_count
  ch4: event_type_score
  ch5: sentiment_vs_price_residual

输出:
  shape: (batch, 8)
  含义: 8 个元板块的动量预测分数
```

说明：

- `ch4` 不再写成 “4 维 one-hot 展平”，否则和总通道数 6 冲突
- `ch4 = event_type_score` 正式冻结为单通道加权事件分数：

```python
bucket_weight = {
    "policy_macro": 1.0,
    "earnings_fundamental": 0.6,
    "product_industry": 0.3,
    "risk_negative": -1.0,
}

event_type_score[sub][t] = clip(
    sum(bucket_weight[bucket_j] for news_j in day_t_of_sub) / (news_count[sub][t] + 1e-9),
    -1.0,
    1.0,
)
```

- 上游事件分类必须先映射到这 4 个 bucket，再进入 TCN；训练和推理不得各自使用不同映射。
- `ch5 = sentiment_vs_price_residual` 正式冻结为“前一交易日情感对当日收益解释偏差”：

```python
ret_1d[t] = close[t] / close[t-1] - 1
sent_lag[t] = sentiment_ema[t-1]

# beta 只用历史窗口估计，例如过去 60 个交易日
beta[t] = cov(sent_lag[t-60:t-1], ret_1d[t-59:t]) / (var(sent_lag[t-60:t-1]) + 1e-9)

sentiment_vs_price_residual[t] = zscore(
    ret_1d[t] - beta[t] * sent_lag[t]
)
```

- 时间对齐规则固定为：`T-1` 日情感解释 `T` 日收益，不允许在在线推理中使用 `T+1` 价格。
- 如交易日缺失，则沿用最近一个完整交易日窗口重算，不做未来值填充。

### 2. TCN 标签定义

TCN 标签正式冻结为：

```python
raw_momentum = (meta[t+5] - meta[t]) / (abs(meta[t]) + 1e-9)
raw_momentum = winsorize(raw_momentum, p1=1, p99=99)
z = (raw_momentum - mu) / (sigma + 1e-9)
target = tanh(z)
```

同时保留分类标签：

```python
y_cls = 1 if abs(raw_momentum) > 0.05 else 0
```

约束：

- 文档和实现都不再使用旧的 `clip(..., -1, 1)` 作为最终训练标签
- 如果后面要改阈值或 lookahead，只能在这一节更新

### 3. 元板块聚合

元板块情感定义冻结为加权聚合：

```python
meta_sentiment[sector][t] = sum(sub_sentiment_i[t] * weight_i) / sum(weight_i)
```

其中权重来自 `meta_sector_mapping.json`。

治理规则：

- `meta_sector_mapping.json` 是唯一权重来源，不允许在代码中再写一套隐式映射。
- 权重生成方式先冻结为“研究侧人工维护 + 版本化提交”，不是在线拟合。
- 默认按月检查，若出现细分类别新增、长期空桶或映射失衡，可触发临时更新。
- 映射权重发生变更后，必须重跑 Phase 1 训练产物，不能直接复用旧模型。

### 4. LightGBM 特征

LightGBM 保持 8 个独立模型，每个模型使用 16 维特征，但必须满足一个原则：

- 所有特征都必须在训练和真实推理时同时可得

因此，原计划中的 `tcn_residual = tcn_reg - target_value` 不可采用，因为它依赖真实标签，存在泄漏风险。

LightGBM 16 维特征冻结为：

```text
[0]  delta_sentiment_1w
[1]  delta_sentiment_2w
[2]  news_count
[3]  news_heat
[4]  tcn_reg
[5]  tcn_reg_delta
[6]  tcn_prediction_stability
[7]  news_count_std_5d
[8]  sentiment_volatility_5d
[9]  tcn_heat_interaction
[10] volume_ratio
[11] intraday_vol
[12] avg_price
[13] global_leader_sentiment
[14] market_beta
[15] sentiment_entropy
```

说明：

- `[6] tcn_prediction_stability` 用于替代原来的 `tcn_residual`
- `[6] tcn_prediction_stability` 正式冻结为：

```python
p = [tcn_reg[t-4], tcn_reg[t-3], tcn_reg[t-2], tcn_reg[t-1], tcn_reg[t]]
dir_consistency = abs(sum(sign(x) for x in p)) / 5.0
dispersion = std(p) / (mean(abs(x) for x in p) + 1e-9)
tcn_prediction_stability = clip(dir_consistency - 0.5 * dispersion, 0.0, 1.0)
```

- 该特征只允许使用历史 `tcn_reg` 序列，不得使用真实未来收益、真实标签或回看误差。
- 不允许直接使用任何依赖未来真实收益或未来标签的特征
- `[3] news_heat` 与 IForest 的关系正式冻结为：`news_heat` 直接取 IForest 输出的板块级异常热度分数，再做 252 交易日滚动 percentile 归一化到 `[0, 1]`。
- `[13] global_leader_sentiment` 正式冻结为“海外领导市场上一完整已收盘 session 的情感加权值”：

```python
leader_basket = {
    "SPY": 0.35,
    "QQQ": 0.30,
    "SOXX": 0.20,
    "TLT": 0.15,
}

global_leader_sentiment[t_cn] = sum(
    sentiment_last_closed_session[symbol][t_cn] * weight
    for symbol, weight in leader_basket.items()
)
```

- 时间对齐规则固定为：对中国市场交易日 `t_cn`，统一读取北京时间 `08:00` 之前最近一个“已完整收盘”的海外 session；盘后新增新闻一律计入下一个中国交易日。
- `leader_basket` 如需调整，必须在配置文件和文档中同步更新，并触发特征导出回归检查。

### 5. IsolationForest

IForest 的文档定义收敛为：

- 输入是按元板块组织的新闻热度时序特征
- 输出是“每个元板块一个异常热度分数”
- 实现上可以是 `8 个独立分数`，但文档不再强绑定某个具体张量 shape

也就是说，本质要求是：

- Agent 和 LightGBM 能拿到板块级 `news_heat`
- 具体是 1 个模型多列输出，还是 8 个模型各自产生分数，由实现决定

### 6. SHAP 使用边界

- SHAP 只用于训练后分析
- 不参与交易规则
- 重点检查 `tcn_reg`、`tcn_reg_delta`、`tcn_prediction_stability` 是否合计占比过高
- 若 TCN 相关特征贡献长期高于 `70%`，不直接判定模型失效，而是触发一次 ablation review：
  - 检查移除二阶段 TCN 相关特征后的性能变化
  - 检查非 TCN 特征是否仍提供独立增益
  - 只有在“高占比 + 无独立增益”同时成立时，才认定二阶段模型退化成 TCN 包装层

---

## 当前代码基线与目标差异

当前仓库里，`state.py`、`features.py`、`daily_guardrail.py`、`prompt_manager.py`、`meta_sector_map.py`、`xai.py` 等文件已经存在，因此本计划不再把它们视作“从零开始的新文件”。

真正要解决的是以下差异：

1. 训练侧还有旧的 per-industry 思路残留，需要彻底切到 fan-in TCN
2. 文档中关于标签、特征和 SHAP 的定义有前后冲突，需要统一
3. Agent 侧虽然已经出现 `level1_plan` / `level2_plan` 结构，但 schema、prompt、risk check、dry run 还没有形成稳定闭环
4. 某些增强模块已经存在雏形，但顺序上不应该早于训练闭环

---

## 实施顺序

### Phase 1: 先把训练闭环跑通

目标：

- fan-in TCN 可训练
- 8 个独立 LightGBM 可训练
- SHAP 可导出
- `export_phase2_dataset()` 可导出稳定特征

范围：

- `data/meta_sector_mapping.json`
- `src/utils/meta_sector_map.py`
- `trainer/src/models/signals.py`
- `trainer/src/datasets/signals.py`
- `trainer/src/pipelines/train_signals.py`
- `trainer/src/utils/signals_xai.py`

完成标准：

- `python -m trainer.main signals train` 可以完整跑通
- 能产出 TCN、LightGBM、IForest、SHAP 结果
- `agent_features.parquet` 的字段稳定，不含未来信息
- TCN 至少满足以下验收口径：
  - 验证集 loss 在连续 5 个 epoch 内不发散
  - 8 个元板块平均方向准确率 `>= 52%`，或平均 rank IC `> 0`
- LightGBM 至少满足以下验收口径：
  - 8 个模型平均 AUC `>= 0.55`，且不存在单模型长期接近随机猜测
  - 若任务定义不适合 AUC，则至少提供等价的 `balanced accuracy` 或 `PR-AUC` 验收口径
- SHAP 验收不只检查文件生成，还必须检查特征重要性分布是否可解释，并输出异常主导特征清单

### Phase 2: 再打通 Agent 决策闭环

目标：

- Agent 统一使用元板块级 schema
- 决策 prompt、risk check、workflow 一致
- Level 1 / Level 2 输出结构稳定

范围：

- `src/agent/state.py`
- `src/agent/prompts.py`
- `config/prompts/trader.md`
- `src/agent/tools.py`
- `src/agent/single_agent.py`
- `src/agent/workflow.py`
- `src/config.py`

完成标准：

- `decide_node` 输出稳定的 `level1_plan[]` 和 `level2_plan[]`
- `risk_check_node` 在元板块层面工作，不再依赖旧行业 schema
- 所有决策使用同一份 `decision_context`
- `level1_plan[]` 必须覆盖全部 8 个元板块；未入选板块允许权重为 `0`
- Level 2 只对 “通过 Level 1 且经一级风控后仍保留正向权重” 的元板块生成 ETF 选择
- `risk_check_node` 必须分两次介入：
  - 第一次在 Level 1 后，负责板块禁入、权重上限、总风险预算
  - 第二次在 Level 2 后，负责 ETF 集中度、重复暴露、最终组合约束
- `risk_check_node` 只能删减、降权或冻结风险暴露，不能新增未被 Agent 选中的方向

### Phase 3: 最后接入增强模块

目标：

- 用 dry run 积累决策日志
- 用日志回灌 prompt pattern
- 用日频 guardrail 管理禁闭期和覆盖逻辑

范围：

- `src/agent/features.py`
- `src/agent/decision_logger.py`
- `src/agent/daily_guardrail.py`
- `src/agent/rule_engine.py`
- `src/agent/prompt_manager.py`
- `scripts/run_phase2_dry_run.py`

完成标准：

- dry run 可生成 `decision_logs.jsonl`
- good/bad pattern 能从日志中稳定抽取
- `FORBIDDEN_ZONE` 触发、冷却、自动释放规则可验证
- dry run 不只看最终区间结果，必须至少按月做一次 checkpoint，检查日志完整性、guardrail 触发频率和异常决策样本

---

## 关键设计决策

1. 决策粒度切换到 8 元板块是本次改造的主线，ETF 选择是第二层，不再把细分行业直接暴露给交易决策层。

2. fan-in TCN 替代 `finetune_per_industry()`，原因不是“更先进”，而是它更符合现在的信号组织方式：47 个细分需要共同决定 8 个元板块，而不是先拆开再硬拼回去。

3. 标签采用 `Winsorize -> Z-score -> tanh`，因为这比简单 clip 更稳定，也更适合后续把 TCN 输出继续喂给 LightGBM。

4. LightGBM 必须坚持“在线可得特征”原则。任何依赖真实未来标签的特征，哪怕离线看起来有效，也一律视为泄漏。

5. SHAP 是诊断工具，不是决策工具。它负责回答“LightGBM 学到了什么”，不负责回答“现在该买什么”。

6. `daily_guardrail` 是覆盖层，不是第二个 Agent。它只能收紧或禁止风险暴露，不能替代周频主决策。

7. `daily_guardrail` 的触发条件先冻结为四类：异常波动、异常热度、数据缺失、禁闭期未结束。它允许的动作只有 `cap_weight`、`forbid_open`、`forbid_add`、`force_flat` 四类，其中 `force_flat` 只用于硬风险事件。

8. `daily_guardrail` 的冷却规则固定为：触发后进入 `FORBIDDEN_ZONE`，至少保留 3 个交易日；若期间再次触发则顺延，只有连续 3 个交易日无触发事件才自动释放。

9. 只有在 Phase 1 和 Phase 2 稳定之后，才值得做 `prompt_manager` 这类经验回灌增强。否则只是在不稳定输入上叠加不稳定策略。

---

## 验证方式

1. 训练验证

- 运行 `python -m trainer.main signals train`
- 检查 fan-in TCN loss 是否收敛
- 检查 8 个 LightGBM 是否都能产出评估结果
- 检查是否达到 Phase 1 中约定的最低质量门槛

2. 数据泄漏检查

- 核对 `export_phase2_dataset()` 的全部字段
- 确认没有使用未来价格、未来标签或真实未来收益构造特征

3. SHAP 检查

- 检查 `shap_values.csv`、summary plot、force plot 是否生成
- 检查 TCN 相关特征是否长期过度主导
- 若 TCN 相关特征占比持续高位，执行一次 ablation review，而不是直接按固定阈值判死

4. Agent 闭环检查

- 跑通一次周频决策链路
- 确认 `level1_plan[]`、`level2_plan[]`、`risk_check`、`workflow` 使用同一 schema

5. Dry run 检查

- 跑 `2024-10-01` 到 `2025-06-30` 的周频 dry run
- 检查 `decision_logs.jsonl`、guardrail 事件、质量标签是否完整
- 至少按月做 checkpoint，避免等到全区间结束才暴露系统性问题

---

## 明确不做

- 不让 SHAP 直接参与交易
- 不再继续维护旧的 per-industry 决策 schema 作为主路径
- 不使用任何带未来标签的信息构造 LightGBM 特征
- 不在训练闭环未稳定前优先做复杂的 prompt pattern 检索
- 旧的 per-industry 训练/决策代码本阶段只保留兼容读取，不再新增逻辑；待元板块主路径稳定后归档删除
