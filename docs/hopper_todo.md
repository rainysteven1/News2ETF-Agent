# `docs/hopper.md` / Signals-Agent 收敛 TODO

这份 TODO 不再围绕旧的 reviewer 文档缺口，而是直接服务当前主目标：

```text
signals train
  -> 导出可部署 ONNX bundle
signals infer
  -> 用固定 bundle 对历史 / 测试 / 未来日期做推理
agent
  -> 消费推理结果 + Memos + 新闻 + 市场/持仓上下文
backtest
  -> 只验证 Agent 基于这些在线可得上下文的决策表现
```

---

## P0：本轮必须落地

### 1. `signals train` 导出可部署 bundle

- [x] 导出 `tcn.onnx`
- [x] 尝试导出逐板块 `lgbm/*.onnx`
- [x] 尝试导出 `iforest.onnx`
- [x] 写出 `manifest.json`
- [x] 支持固定部署目录 `trainer/models/signals/latest`

当前实现：

- 训练结束后会导出 ONNX bundle
- 目录由 `signals.training.deploy_onnx_dir` 控制

---

### 2. 增加显式 `signals infer`

- [x] 增加 `python -m trainer.main signals infer`
- [x] 支持 `--bundle-dir`
- [x] 支持 `--output-path`
- [x] 支持 `--start-date / --end-date`
- [x] 用 ONNX bundle 生成 `agent_features.parquet`

当前实现：

- `signals infer` 已挂到 trainer CLI
- 推理入口为 `src/signals/signals_inference.py`

---

### 3. Agent 优先消费 signals 推理产物

- [x] `AgentFeatureBuilder` 优先读取 `agent_features.parquet`
- [x] 若缓存缺失，优先尝试用 `signals_onnx_dir` 自动生成
- [x] 增加 `ml_signal_snapshot`
- [x] `compute_ml_signals` 改为元板块信号口径

---

### 4. 把多源上下文真正接到 trader prompt

- [x] `build_decision_context` 输出结构化上下文
- [x] `tools_node` 将 `build_decision_context` 回填到 `state["decision_context"]`
- [x] trader prompt 增加：
  - `ml_signal_snapshot`
  - `historical_memory`
  - `good_patterns`
  - `bad_patterns`

---

### 5. Memos / 日志模式作为辅助上下文

- [x] `build_decision_context` 注入 PromptManager good/bad patterns
- [x] 若配置了 Memos API，则同时检索相似历史案例
- [x] 将这部分统一放入 `historical_memory`

---

## P1：下一轮继续收敛

### 6. 清理 Agent placeholder

- [x] `_get_sector_price()`
- [x] `market_state.volume_ratio`
- [x] `weekly_returns`
- [x] `agent_perf_1w / agent_perf_4w`

说明：

- `AgentFeatureBuilder` 现在会从 ETF 量价、backtest parquet、meta-sector ETF 映射中回填这些字段
- `agent_features` 缓存优先读 `data/agent_features.oof.parquet`

---

### 7. 让 backtest 完全切到元板块主路径

- [x] ETF 选择链路从旧 `industry` 口径完全迁移到 `8 元板块 -> Level 2 ETF`
- [x] 风控和回测报告统一用元板块 schema

当前实现：

- `Portfolio` 持仓改为 `meta_sector -> weight`
- 同时记录 `selected_etfs`
- 回测结果新增：
  - `meta_sector_contributions`
  - `meta_sector_returns`

---

### 8. walk-forward 从“评估”升级为“正式推理产物”

- [x] 导出 OOF `agent_features`
- [x] 明确区分 train / val / test / future inference
- [x] Backtest 默认优先读取 OOF / infer 产物，而不是训练期顺手导出的全历史特征

当前实现：

- `signals train` 导出：
  - `data/agent_features.parquet`
  - `data/agent_features.oof.parquet`
- `signals infer` 默认输出到 `data/agent_features.oof.parquet`
- Agent 默认优先读取 OOF / infer 特征缓存

---

## P2：建议补强

### 9. 校验 ONNX 和 Python 推理一致性

- [x] 对 TCN / LGBM / IForest 做数值回归检查
- [x] 写成脚本或测试，避免训练和推理口径再漂移

当前实现：

- 新增 `scripts/check_signals_onnx_consistency.py`
- 对参考 `agent_features` 与 ONNX 推理结果的重叠日期做列级数值比对
- 默认跳过 `lgbm_score_*` 这类训练期未写回的列

### 10. 决策上下文标准化

- [x] `build_decision_context` 同时产出 JSON 与人类可读摘要
- [x] researcher / trader prompt 统一引用同一份 schema

当前实现：

- `build_decision_context` 输出：
  - `schema_version`
  - 结构化 JSON
  - `human_summary`
- trader prompt 会优先拼接 `human_summary`

### 11. Agent 侧按日期增量刷新特征缓存

- [x] 不是每次都全量重跑
- [x] 支持 “只补到某个 date” 的缓存刷新

当前实现：

- `AgentFeatureBuilder._ensure_agent_feature_cache_upto(date)`
- 若指定日期超出当前缓存最大日期，则只增量补推到该日期

---

## 当前判断

当前主链已经从：

```text
训练后顺手导出一些历史特征
```

推进到：

```text
signals train -> ONNX bundle
signals infer -> 推理特征
agent -> 读取推理特征 + Memos + 其他上下文
```

当前 P1 / P2 已全部落地。

剩下更值得继续做的是：

1. 提升 `signals` 的 walk-forward / OOF 训练严谨度，而不是只补导出
2. 把 `daily_guardrail` 和日频行为日志里的 placeholder 继续清理掉
3. 继续强化 backtest 的 ETF 层归因和报告展示
