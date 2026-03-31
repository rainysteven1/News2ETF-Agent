# Plan: LLM 驱动的行业轮动 + ETF 智能选择

## Context

用户确认：**LLM 做真正的行业判断，不是规则引擎**。规则只做 Guardrail。

现有 pipeline 已完成：
- Raw新闻 → FinBERT+SetFit ONNX → 大类行业/细分行业/情感标签（已缓存）
- `get_onnx_predictions()` 提供每条新闻的三重标签

ML Pipeline（用户当前制作数据集中）：
```
情感时序 → TCN → 动量分数 [-1, 1]（主要信号）
新闻量时序 → Isolation Forest → 热度异常 [0, 1]
上述特征 → LightGBM → 综合信号
可选: GNN → 跨行业传导（油价→新能源）
```

**情感聚合粒度**：按 `sub_category`（47个细分），不是 `major_category`（8个大类）

**LLM 决策流程**（两层选择）：
1. 行业信号 + 新闻摘要 → LLM 判断"哪些行业值得配置 + 权重"
2. 行业 → LLM 选择 top 跟踪指数 → 再选具体 ETF（看规模/管理人/基准）

---

## LLM 决策输入设计

### 信息压缩策略

原始新闻 3000 条 → 每行业 top 3-5 条摘要（按置信度排序）

**摘要格式**（每条 ~50-80 tokens）：
```
[情感] 置信度 | 标题（50字内） | 关键词1, 关键词2
```

**LLM 完整输入结构**：
```
## 行业综合信号（LightGBM 输出）
- 大类A/细分A: 综合=0.65, 动量=0.42, 热度=0.65, 趋势=1
- 大类A/细分B: 综合=-0.30, 动量=-0.15, 热度=0.30, 趋势=0
...

## 历史动量轨迹（TCN，近4周）
- 大类A/细分A: [0.20, 0.30, 0.35, 0.42]  ← 持续上行
- 大类A/细分B: [0.10, 0.05, -0.10, -0.15] ← 持续下行
...

## 高置信度新闻摘要（每细分 top 3）
### 大类A/细分A（银行）
- [positive] 0.92 | 国有大行下调存款利率... | 降息, 银行, 存款
- [positive] 0.88 | 银行板块集体拉升... | 银行, 券商
...

### 大类A/细分B（半导体）
- [negative] 0.78 | 美国扩大芯片出口管制... | 半导体, 出口管制
...

## 可选 ETF 列表（该细分下所有 ETF）
| ETF代码 | 名称 | 跟踪指数 | 规模(亿) | 管理人 | 基准误差 |
|---------|------|----------|----------|--------|----------|
| 512800 | 华夏银行ETF | 中证银行指数 | 45.2 | 华夏基金 | 0.02% |
| 159887 | 银行ETF | 沪深300银行分指数 | 12.1 | 广发基金 | 0.05% |
...

## 当前持仓
上周持仓: 大类A/细分A 30%, 大类B/细分C 25%
上周收益: +2.3%
已投权重: 55%
```

---

## LLM 决策输出格式

```json
{
  "weekly_plan": [
    {
      "industry": "大类A/细分A",
      "action": "buy",
      "weight": 0.25,
      "selected_indices": ["中证银行指数"],
      "selected_etf": "512800 华夏银行ETF",
      "reason": "动量持续上行4周，新闻高置信度正面，存款降息利好持续"
    },
    {
      "industry": "大类A/细分B",
      "action": "sell",
      "weight": 0.15,
      "selected_indices": [],
      "selected_etf": null,
      "reason": "动量连续下行，出口管制压力持续"
    }
  ]
}
```

---

## Guardrail 规则（规则引擎校验，非 LLM 决策）

1. **单行业权重上限**: 30%
2. **总权重上限**: 100%
3. **Beta 惩罚**: very_high beta 仓位 ×0.7；high beta ×0.85
4. **Mirror 检查**: 同 correlation_cluster 的两个行业不能同时权重 ≥ 15%
5. **亏损保护**: 上周 return < 0 时，禁止新建 very_high beta 仓位
6. **最小操作阈值**: 权重变化 < 5% → 降为 HOLD（不值得手续费）

---

## 数据准备需求

### 1. 情感时序（TCN 输入）

```
日期 | 行业 | 情感均值 | 情感std | 新闻数量 | 平均置信度
```

**情感分数**：sentiment_score = 1/0/-1 (positive/neutral/negative) × l1_confidence

**TCN 输出**：动量分数 [-1, 1]，捕捉情感上升/下降趋势

### 2. 新闻量时序（Isolation Forest 输入）
```
日期 | 行业 | 当日新闻数量 | 近5日均值 | 近5日std
```

### 3. ETF 元信息表
```
ETF代码 | 名称 | 跟踪指数 | 规模 | 管理人 | 业绩基准 | 跟踪误差
```

### 4. 行业-指数-ETF 映射（IndustryMapper，已有）
- 大类 → 细分 → 跟踪指数 → best ETF

---

## 实现步骤

### Phase 1: 数据基础设施（用户当前在做的）

- [ ] 构建情感时序 DataFrame（按行业+日期聚合）
- [ ] 构建新闻量时序 DataFrame
- [ ] 训练 TCN 模型（动量信号）
- [ ] 训练 Isolation Forest（热度信号）
- [ ] 训练 LightGBM（综合信号）
- [ ] ONNX 导出并集成到 `compute_ml_signals`

### Phase 2: LLM 决策输入工具

- [ ] 新建 `build_decision_context(date)` 函数
  - 聚合行业信号（从 LightGBM 输出）
  - 拉取历史动量轨迹
  - 压缩新闻摘要（每行业 top 3）
  - 拉取候选 ETF 列表（按细分行业）
  - 拉取当前持仓

- [ ] 新增 `get_industry_top_news(date, industry, top_k=3)` 工具
  - 从 `get_onnx_predictions` 缓存按置信度排序取 top-k

- [ ] 新增 `get_etf_candidates(industry)` 工具
  - 从 IndustryMapper 获取该行业所有跟踪指数
  - 从 ETF info 表拉取每只 ETF 的规模/管理人/基准/误差

### Phase 3: decide_node 重构

- [ ] 修改 `decide_node` 调用 `build_decision_context()`
- [ ] Prompt 改为明确告诉 LLM：基于信号+摘要+趋势做判断
- [ ] 输出格式严格匹配 `TradeDecision` 模型

### Phase 4: Guardrail 集成

- [ ] 在 `risk_check_node` 前增加 `guardrail_node`
- [ ] 实现上述 6 条规则
- [ ] 违规时返回修改建议（而不是拒绝）

---

## 关键文件

| 文件 | 改动 |
|---|---|
| `src/agent/tools.py` | 新增 `get_industry_top_news`, `get_etf_candidates` |
| `src/agent/single_agent.py` | 重构 `decide_node`，调用新的决策上下文构建 |
| `src/agent/rule_engine.py` | 新建 Guardrail 规则引擎 |
| `src/agent/state.py` | 确认/扩展 `TradeDecision` 支持 `selected_indices`, `selected_etf` |
| `src/signals/raw_scorer.py` | 集成 TCN/IForest/LightGBM（Phase 1，重构） |
| `config/prompts/trader.md` | 更新 prompt 描述 LLM 决策逻辑 |

---

## 验证方式

1. 调用 `build_decision_context("2024-01-01")`，确认 token 量可控（< 5k）
2. LLM 基于该输入输出的决策格式正确
3. Guardrail 正确拦截违规决策
4. 完整 backtest 无 crash
