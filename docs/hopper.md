# 实现计划：三阶段训练架构 + 双模式决策系统

## Context

根据 `docs/plan-update.md` 和 `docs/plan.md`，实现完整的代码改动。当前代码库已有：
- 8 大类 → 47 细分的情感分类体系
- TCN + LightGBM + IForest 三模型 pipeline
- LangGraph ReAct Agent（agent_node → tools_node → decide_node → risk_check）
- ONNX 推理 pipeline（FinBERT + SetFit）
- Walk-forward 回测引擎

本次改动核心：**从"按细分行业决策"改为"按 8 元板块决策（Level 1）+ ETF 选择（Level 2）"**，以及新增 SHAP 可解释性分析。

---

## 三模型输入输出规格

### TCN（扇入式）

```
输入：
  shape: (batch, seq_len=5, n_sub=47, channels=6)
  来源：sentiment_df 按 sub_category 聚合后的 47 细分，每条新闻 → 6 通道特征

  6 通道定义（每行 = 每个时间步，每个细分）：
    ch0: sentiment_ema     — EMA 平滑情感存量，α=0.2
    ch1: sentiment_acceleration — 情感变化率的变化（爆发点）
    ch2: sentiment_std    — 1 小时间情感标准差（共识 vs 分歧）
    ch3: log_news_count  — log(news_count + 1)
    ch4: event_type_embedding — 事件类型 one-hot（政策/业绩/技术/其他→4维展平）
    ch5: sentiment_vs_price_residual — 情感 Z-score − 价格 Z-score

输出：
  shape: (batch, 8)
  含义：8 个元板块的动量预测分数 [-1, 1]
  训练标签：meta_momentum[t] = Σ(细分情感[t] × 权重) / Σ 权重
           目标 = clip((meta[t+5] − meta[t]) / |meta[t]|, −1, 1)
  损失函数：MSELoss(pred_reg, y_reg) + 0.01 × BCELoss(pred_cls, y_cls)
```

### LightGBM

```
训练方式：8 个独立 LightGBM（每个元板块独立训练，共用 16 维特征）

原因：8 个元板块的驱动逻辑差异极大（科技看流动性、消费看内需、金融看利率），
强制共享结构（multi-output）反而有害。独立训练更灵活，Phase 2 可单独评估每个
板块的 IC，分辨哪个板块信号最有效。

输入：
  shape: (N_sector, 16)
  每个 sector 独立训练：X_sector, y_sector

  16 维特征清单（修正后）：
    [0]  delta_sentiment_1w       — TCN 输出：8 元板块情感变化
    [1]  delta_sentiment_2w       — TCN 输出：8 元板块 2 周变化
    [2]  news_count               — 新闻数量
    [3]  news_heat                — IForest 热度
    [4]  tcn_reg                  — TCN 回归输出
    [5]  tcn_residual             — TCN 预测残差（替代 tcn_cls）
    [6]  tcn_reg_delta            — TCN 回归 delta
    [7]  news_count_std_5d        — 5 日新闻数量标准差
    [8]  sentiment_volatility_5d  — 5 日情感波动率
    [9]  tcn_heat_interaction     — tcn_reg × news_heat
    [10] volume_ratio             — 量比（当日/均值）
    [11] intraday_vol             — 日内波动率
    [12] avg_price                — 平均价格
    [13] global_leader_sentiment  — 跨行业传导特征
    [14] market_beta              — Beta 敏感度
    [15] sentiment_entropy         — 情感熵

输出：
  8 个独立模型 → 8 个 (N_sector, 1) 预测向量
  标签：next_dir = sign(sentiment[t+1] − sentiment[t])
  评估：每个 sector 独立计算 R²、IC
```

### IsolationForest

```
输入：
  shape: (n_periods, n_features)
  来源：新闻量时序特征
  特征：[news_count, news_heat, amount, roll_mean_amt, roll_std_amt] × 5 天窗口
      = 5 × 5 = 25 维（与现有相同）

输出：
  shape: (n_periods, 8) — 每元板块热度异常分数 [0, 1]
  含义：>0.5 = 新闻量异常高（可能存在热点轮动预警）
```

### 数据流向总图

```
原始新闻
   ↓
FinBERT ONNX → L1 类别 + 情感
   ↓
SetFit ONNX → 47 细分行业类别
   ↓
sentiment_df（47 细分 × 日期）
   ↓
┌─────────────────────────────────────┐
│         TCN Fan-in（47→8）           │
│  输入: (batch, 5, 47, 6)             │
│  输出: (batch, 8) 元板块动量           │
└─────────────────────────────────────┘
   ↓                                    ↓
┌──────────────┐              ┌─────────────────────────┐
│  IForest      │              │  LightGBM × 8（独立模型）│
│  热度异常检测   │              │  每个板块独立训练         │
│  → (n, 8)     │              │  共用 16 维特征          │
│               │              │  → (n, 8) 综合信号       │
└──────────────┘              └─────────────────────────┘
   ↓                                    ↓
   └──────────→ Agent 决策 ←─────────────┘
                   ↓
          ┌────────────────┐
          │  SHAP 分析      │
          │  → 回测报告可视化│
          └────────────────┘
```

---

## 现有 pipeline vs 新 pipeline（核心差异）

| | 现有（train.py） | 新（plan-update.md） |
|--|---|---|
| TCN 输入 | per-industry `(N, seq_len, 6)`，8行业独立 | **扇入式 `(N, 5, 47, 6)`**，单模型同时看47细分 |
| TCN 输出 | per-industry 回归头（每行业1个） | **单输出头 `(batch, 8)`**，8元板块动量 |
| TCN 微调 | `finetune_per_industry()` 逐行业微调 | **移除**（扇入式共享学习跨行业传导） |
| LightGBM | 13 维特征 | **16 维**（+global_leader_sentiment, +market_beta） |
| SHAP | 无 | **新增 xai.py** |
| Phase2 数据 | 无 | **新增 `export_phase2_dataset()`** |

---

## 实现顺序（19 个文件，按依赖排序）

### 第 1 步：`data/meta_sector_mapping.json`（新建）
- **来源**：`industry_dict.json` 的 47 细分 → 8 元板块 + 权重
- **关键结构**：
  - `meta_sectors`: 8 个元板块，每个含 `sub_categories[]` 和 `market_cap_weight`
  - `global_leader_map`: 跨行业传导映射

**市值权重设计**（替代简单等权）：

| 等级 | market_cap_weight | 示例 |
|------|-------------------|------|
| 核心驱动 | ×1.5 | 半导体/芯片（科技成长）、军工/国防（高端制造） |
| 重要辅助 | ×1.0 | 软件/信创（科技成长）、新能源/光伏（高端制造） |
| 边缘平滑 | ×0.5 | 打印机租赁（科技成长）、消费电子/家电（大消费） |

**标签构造中的权重应用**：
```python
meta_sentiment[sector][t] = Σ(sub_sentiment_i[t] × market_cap_weight_i) / Σ market_cap_weight_i
```
这使 TCN 输出的 8 维动量与 ETF 真实价格走势更贴合。

### 第 2 步：`src/utils/meta_sector_map.py`（新建）
- `load_meta_sector_mapping()` → 加载 JSON
- `sub_to_meta(sub: str) → str`: 细分 → 元板块
- `meta_to_subs(meta: str) → list[str]`: 元板块 → 细分列表
- `get_upstream_sentiment(sector: str) → list[str]`: **获取影响该 sector 的所有上游领导板块（多对多）**
  - 例：`get_upstream_sentiment("高端制造")` → `["有色金属/稀土", "能源/油气/资源"]`
  - 这是"跨行业传导"逻辑的核心：Agent 决策高端制造时，需同时看其上游板块过去 5 天情感

### 第 3 步：`src/config.py`（修改）
```python
class DataConfig(BaseModel):
    # 新增字段：
    meta_sector_mapping: Path = _ROOT / "data" / "meta_sector_mapping.json"
    output_agent_features: Path = _ROOT / "data" / "agent_features.parquet"
    output_logs: Path = _ROOT / "data" / "decision_logs.jsonl"
```

### 第 4 步：`src/agent/state.py`（修改）
```python
class SectorStatus(Enum):
    NORMAL = "normal"
    FORBIDDEN_ZONE = "forbidden"

class MetaSectorPlan(BaseModel):
    meta_sector: str
    action: str  # buy/sell/hold
    weight: float
    reason: str = ""

class ETFSelections(BaseModel):
    meta_sector: str
    selected_indices: list[str]
    selected_etf: str

class TradeDecision(BaseModel):
    # 保留现有字段
    level1_plan: list[MetaSectorPlan] = []
    level2_plan: list[ETFSelections] = []
    sector_status: dict[str, SectorStatus] = {}

class AgentState(TypedDict):
    # 新增：
    forbidden_sectors: dict[str, str]
    tcn_sequence: dict[str, list[float]]
    decision_context: dict[str, Any]
    last_guardrail_events: list[dict]
```

### 第 5 步：`trainer/signals/models.py`（修改）

**TCN 架构变更**：现有 `TCN(input_size=6)` → 新 `TCNFanIn`

```python
class SpatialDropout(nn.Module):
    """Spatial Dropout：对 47 维中的随机子集（所有 6 通道一起丢）

    防止 TCN 记住"某几条特定新闻"，强迫模型学习跨维度的共性模式。
    p=0.3 表示每 batch 丢弃 30% 的 47 维输入通道。
    """
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 47, 6)
        if not self.training:
            return x
        # mask shape: (batch, 1, 47, 1) — 对 47 维随机二值掩码
        mask = (torch.rand(x.shape[0], 1, x.shape[2], 1, device=x.device) > self.p)
        return x * mask / (1 - self.p)  # inverted dropout


class TCNFanIn(nn.Module):
    """Fan-in TCN: 输入 (batch, seq_len, 47, 6) → 输出 (batch, 8) 元板块动量

    架构：
      1. Spatial Dropout（防止过拟合）
      2. 空间压缩：47*6=282 → 128 → 64（MLP，可学习加权聚合）
      3. 时间建模：TCN stack（kernel_size=3, dilation=[1,2,4,8]）
      4. 输出头：Linear(hidden, 8) — 8 个元板块动量
    """

    def __init__(self, n_sub=47, n_meta=8, input_size=6,
                 hidden_size=64, num_layers=4, kernel_size=3, dropout=0.2,
                 spatial_dropout_p=0.3):  # 推荐 0.3-0.4
        self.spatial_dropout = SpatialDropout(p=spatial_dropout_p)
        # 空间压缩：47*6=282 → 64
        self.spatial_mlp = nn.Sequential(
            nn.Linear(n_sub * input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_size),
        )
        # 时间建模（复用现有 TCN）
        self.tcn = TCN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        # 输出头：8 维元板块动量（带可学习缩放因子）
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_meta),
        )
        self.scale = nn.Parameter(torch.ones(1))  # 可学习缩放因子：帮助 tanh 在±1附近梯度平缓时仍能收敛
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, 47, 6)
        x = self.spatial_dropout(x)  # Spatial Dropout
        batch, seq_len, n_sub, channels = x.shape
        x = x.reshape(batch, seq_len, n_sub * channels)  # (batch, seq_len, 282)
        x = self.spatial_mlp(x)  # (batch, seq_len, hidden_size)
        x = x.transpose(1, 2)  # (batch, hidden_size, seq_len)
        x = self.tcn(x)  # (batch, hidden_size, seq_len)
        x = x.mean(dim=2)  # (batch, hidden_size)
        # 可学习缩放因子 * tanh：解决 tanh 在±1 附近梯度饱和问题
        return torch.tanh(self.reg_head(x) * self.scale), torch.sigmoid(self.cls_head(x))
```

**训练时的 L2 正则化**：将 `weight_decay` 从 `1e-3` 提升到 `5e-3`（在 `train_tcn_fanin` 的 Adam 中设置）。

**Label 修正**（在 `build_sub_category_sequences` 中）：

```python
def build_sub_category_sequences(...):
    # 现有（有问题）：
    #   target = np.clip((meta[t+5] - meta[t]) / (np.abs(meta[t]) + 1e-9), -1, 1)

    # 修正：采用 Rank 变换 + Tanh 混合
    # Step 1: 计算原始 5 日动量
    raw_momentum = (meta[t+5] - meta[t]) / (np.abs(meta[t]) + 1e-9)

    # Step 2: Winsorize（修掉 1%/99% 分位以外的极端值）
    p1, p99 = np.percentile(all_raw, 1), np.percentile(all_raw, 99)
    raw_momentum = np.clip(raw_momentum, p1, p99)

    # Step 3: Z-score 标准化
    mu, sigma = np.mean(all_raw), np.std(all_raw) + 1e-9
    z = (raw_momentum - mu) / sigma

    # Step 4: Tanh 缩压（将 Z-score 映射到 [-1, 1]，极端值平滑）
    target = np.tanh(z)  # 梯度始终有界，不会爆炸

    # 同时保留原始分类标签（用于 BCE）
    y_cls = 1 if |raw_momentum| > 0.05 else 0  # 阈值可调
```

**ONNX 导出**：同前，不变。

---

**Label 修正说明**：

| 方案 | 优点 | 缺点 |
|------|------|------|
| Clip only | 简单 | 梯度在截断处突变，MSE 仍受极端值影响 |
| Rank 变换 | 梯度均匀，极端值友好 | 丢失幅度信息 |
| **Tanh(Z-score)** | 梯度有界，极端值平滑，保留方向+幅度感 | 需配合 Winsorize |
| Winsorize + Clip | 简单 | 截断处仍有梯度问题 |

**推荐 Tanh(Z-score)**：先 Winsorize 修掉极端值，再 Z-score 标准化，最后 `tanh()` 缩压到 [-1, 1]。训练收敛快且稳定。

### 第 6 步：`trainer/signals/dataset.py`（修改）

**D1. 新增 `build_sub_category_sequences()`**：
```python
def build_sub_category_sequences(
    sentiment_df: pl.DataFrame,
    meta_sector_map: dict,
    lookback_days: int = 5,
) -> tuple[X: (N, lookback, 47, 6), y: (N, 8), dates, sub_industries]:
    """构建扇入式 TCN 训练数据。

    输入：sentiment_df（含 47 细分，按 sub_category 聚合）
    输出：
      X: (N, 5, 47, 6) — 47 细分 × 6 通道
      y: (N, 8) — 8 元板块动量标签
      dates: 日期列表
      sub_industries: 47 细分名称列表

    6 通道（per timestep per 细分）：
      ch0: sentiment_ema（EMA α=0.2）
      ch1: sentiment_acceleration（变化率的变化）
      ch2: sentiment_std（情感标准差）
      ch3: log_news_count（log(news_count+1)）
      ch4: event_type_embedding（事件类型 one-hot → 1 维）
      ch5: sentiment_vs_price_residual（情感 Z-score - 价格 Z-score）

    标签：meta_momentum = Σ(细分情感 × 权重) / Σ 权重，
          目标 = clip((meta[t+5] - meta[t]) / |meta[t]|, -1, 1)
    """
```

**D2. 新增 `compute_global_leader_sentiment()`**：
```python
def compute_global_leader_sentiment(sentiment_df, meta_sector_map) -> pl.DataFrame:
    """跨行业传导特征：global_leader_sentiment[sector][t] = mean(sentiment[leader_sector][t-5:t])"""
```

**D3. 新增 `compute_market_beta()`**：
```python
def compute_market_beta(price_df, index_df, meta_sector_map, window=20) -> pl.DataFrame:
    """滚动 20 日 Beta：beta[sector][t] = rolling_correlation(returns[sector], index_returns)"""
```

**D4. 新增 `export_phase2_dataset()`（向量化批量推理）**：
```python
def export_phase2_dataset(
    sentiment_df, price_df, index_df, meta_sector_map,
    tcn_model, lgbm_models: dict, iforest_model, device, output_path: Path,
) -> None:
    """导出每日特征用于 Phase 2 Agent 训练。

    性能优化（关键）：3 年回测区间不能用 for date 循环推理（太慢）。
    改为向量化批量推理：

    1. X_all = build_sub_category_sequences() → (N, 5, 47, 6)
    2. with torch.no_grad(): tcn_out = tcn_model(X_all.to(device))  # 一次批量 (N, 8)
    3. for meta_sector in 8: lgbm_out = lgbm_models[meta].predict(X_sector)  # 8次独立推理
    4. iforest_out = iforest_model.predict(X_iforest)  # 一次
    5. Polars join: date + meta_sector + features → parquet

    这样导出时间从几小时缩短到几分钟。
    """
```

### 第 7 步：`trainer/signals/train.py`（修改）

**`run_training()` 改动对照**：

| | 现有代码 | 改动后 |
|--|---------|--------|
| 导入 | `build_sequences`, `train_tcn_pretrain` | 新增 `build_sub_category_sequences`, `compute_global_leader_sentiment`, `compute_market_beta`, `export_phase2_dataset` |
| Step A | `build_sequences()` per-industry | `build_sub_category_sequences()` 扇入式 |
| Step A | `train_tcn_pretrain(input_size=6)` | `train_tcn_fanin()`（新函数，输入282） |
| Step B | `finetune_per_industry()` | **删除** |
| Step C | `build_lgbm_features()` 单模型 13维 | **8 个独立 LightGBM，16 维** |
| 末 | `_export_all_onnx()` | 新增 Step F：SHAP分析；Step G：`export_phase2_dataset()` |

**新增 `train_tcn_fanin()` 函数**：
```python
def train_tcn_fanin(X, y_reg, y_cls, cfg, wb, device) -> TCNFanIn:
    model = TCNFanIn(n_sub=47, n_meta=8, input_size=6,
                     hidden_size=cfg.tcn.hidden_size,
                     num_layers=cfg.tcn.num_layers,
                     dropout=cfg.tcn.dropout,
                     spatial_dropout_p=0.3).to(device)
    # 训练循环与 train_tcn_pretrain 相同
    # loss = MSELoss(pred_reg, y_reg) + 0.01 * BCELoss(pred_cls, y_cls)
    return model
```

**LightGBM 8 个独立模型训练**：
```python
# Step C: 8 个独立 LightGBM（每个元板块单独训练）
lgbm_models = {}
lgbm_dir = checkpoint_dir / "lgbm"
lgbm_dir.mkdir(exist_ok=True)
for meta_sector in META_SECTORS:  # 8 个元板块
    X_sector = X_lgbm_by_sector[meta_sector]  # 16 维
    y_sector = y_lgbm_by_sector[meta_sector]
    model = train_lgbm_stacking(X_sector, y_sector, dates, signals_cfg, wb)
    lgbm_models[meta_sector] = model
    # 评估：R², IC per sector
    # 导出到 lgbm/科技成长.txt, lgbm/高端制造.txt, ...
    model.booster_.save_model(str(lgbm_dir / f"{meta_sector}.txt"))
```

**ONNX 导出目录结构**：
```
signals-{MMDD-HHMM}/
  tcn.onnx
  lgbm/
    科技成长.txt / .onnx
    高端制造.txt / .onnx
    ...
    主题策略.txt / .onnx
  iforest.onnx
  shap/
    shap_values.csv
    shap_summary.png
    force_plot_2024-10-07.html
    ...
```

**训练后验证**：运行 SHAP 分析，检查 tcn_reg 和 tcn_residual 的总权重。如果两者相加 > 70%，说明 LightGBM 过度依赖 TCN。

### 第 8 步：`trainer/signals/xai.py`（新建）

**LightGBM 特征冗余风险警告**：

当前 16 维中包含多个 TCN 来源特征：`[4]` tcn_reg、`[5]` tcn_cls、`[6]` tcn_reg_delta。这些高度相关（tcn_reg 和 tcn_cls 来自同一表示的线性头），可能导致 LightGBM 过度依赖 TCN 本身而忽视其他特征。

**修正方案**：

```python
# 方案 A（推荐）：用 tcn_residual 替换 tcn_cls
# tcn_residual = TCN 预测值 - 实际值（模型偏差）
# 让 LightGBM 专门学习"模型在什么时候会猜错"
#
# 方案 B：直接删掉 tcn_cls（牺牲分类信息，换特征正交性）
#
# 方案 C：先训练，检查 SHAP importance，如果 tcn_reg > 80% 权重，再做降维

# 修改 build_sub_category_sequences 中的标签构造：
# 在计算 tcn_reg 的同时，计算 tcn_residual
tcn_residual = tcn_reg.item() - target_value  # 预测偏差

# 修改 build_lgbm_features：
# 用 tcn_residual 替代 tcn_cls 位置
```

**LightGBM 16 维特征清单（修正后）**：

| # | 特征名 | 来源 | 说明 |
|---|--------|------|------|
| 0 | delta_sentiment_1w | TCN输出 | 8元板块情感变化 |
| 1 | delta_sentiment_2w | TCN输出 | 8元板块2周变化 |
| 2 | news_count | 新闻聚合 | 新闻数量 |
| 3 | news_heat | IForest | 新闻热度 |
| 4 | tcn_reg | TCN回归输出 | 8维动量预测 |
| 5 | **tcn_residual** | TCN残差 | TCN预测 - 实际（模型偏差）**替代 tcn_cls** |
| 6 | tcn_reg_delta | TCN输出 | TCN回归delta |
| 7 | news_count_std_5d | 统计 | 5日新闻数量标准差 |
| 8 | sentiment_volatility_5d | 统计 | 5日情感波动率 |
| 9 | tcn_heat_interaction | 交叉 | tcn_reg × news_heat |
| 10 | volume_ratio | OHLCV | 量比 |
| 11 | intraday_vol | OHLCV | 日内波动率 |
| 12 | avg_price | OHLCV | 平均价格 |
| 13 | global_leader_sentiment | 手工 | 跨行业传导 |
| 14 | market_beta | 价格计算 | Beta敏感度 |
| 15 | sentiment_entropy | 新闻聚合 | 情感熵 |

**训练后验证**：运行 SHAP 分析，检查 `tcn_reg` 和 `tcn_residual` 的总权重。如果两者相加 > 70%，说明 LightGBM 过度依赖 TCN，需要进一步正则化或降维。

```python
LIGHTGBM_16_FEATURE_NAMES = [
    "delta_sentiment_1w", "delta_sentiment_2w", "news_count", "news_heat",
    "tcn_reg", "tcn_cls", "tcn_reg_delta", "news_count_std_5d",
    "sentiment_volatility_5d", "tcn_heat_interaction",
    "volume_ratio", "intraday_vol", "avg_price",
    "global_leader_sentiment", "market_beta",
]

class SHAPAnalyzer:
    def __init__(self, lgbm_model, X_test):
        import shap
        self.lgbm_model = lgbm_model
        self.X_test = X_test
        self.explainer = shap.TreeExplainer(lgbm_model)
        self._shap_values = None

    def compute_shap_values(self):
        self._shap_values = self.explainer.shap_values(self.X_test)
        return self._shap_values

    def generate_summary_plot(self, output_path: Path):
        import matplotlib
        matplotlib.use('Agg')  # 防止 Debian 无 GUI 服务器报错
        import matplotlib.pyplot as plt
        import shap
        shap.summary_plot(self._shap_values, self.X_test,
                          feature_names=LIGHTGBM_16_FEATURE_NAMES, show=False)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def generate_force_plot(self, date: str, output_path: Path):
        import matplotlib
        matplotlib.use('Agg')
        import shap
        shap.force_plot(self.explainer.expected_value,
                        self._shap_values[0], self.X_test[0],
                        feature_names=LIGHTGBM_16_FEATURE_NAMES,
                        matplotlib=False,
                        out_file=output_path / f"force_plot_{date}.html")

    def export_shap_values(self, dates, output_path: Path):
        import pandas as pd
        df = pd.DataFrame(self._shap_values, columns=LIGHTGBM_16_FEATURE_NAMES)
        df["date"] = dates
        output_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path / "shap_values.csv", index=False)
        return df
```

**集成点**：`train.py` 的 `run_training()` 末尾调用：
```python
analyzer = SHAPAnalyzer(lgbm_model, X_lgbm[split:])
shap_values = analyzer.compute_shap_values()
analyzer.generate_summary_plot(checkpoint_dir / "shap_summary.png")
analyzer.export_shap_values(dates[split:], checkpoint_dir)
```

### 第 9 步：`src/agent/features.py`（新建）
```python
class AgentFeatureBuilder:
    def build_tcn_sequence(self, date, lookback=5) → dict[str, list[float]]  # Feature A
    def build_news_summary(self, date, top_k=1) → dict[str, list[str]]  # Feature B
    def build_market_state(self, date) → dict  # Feature C
    def build_position_state(self, current_holdings, weekly_returns, agent_perf_1w, agent_perf_4w) → dict  # Feature D
    def build_sent_p_divergence(self, date) → dict[str, float]  # Feature E
    def build_agent_features(self, date, current_holdings, current_time: str) → dict[str, Any]:
        """主入口。

        current_time: 决策时刻，如 "2024-10-07 08:30:00"（周一开盘前）
        防 Look-ahead Bias：
          - TCN序列：只取 current_time 之前的 5 个交易日
          - 新闻摘要：只取 current_time 之前（不含周一盘中）
          - 价格动量：必须用 last_close（上星期五收盘价）
        """
```

### 第 10 步：`src/agent/decision_logger.py`（新建）
```python
@dataclass
class TCNPredictionError:
    meta_sector, tcn_predicted, actual_return, divergence, root_cause_guess

@dataclass
class GuardrailEvent:
    date, meta_sector, trigger_type, etf_code, pnl_impact, reason

@dataclass
class DecisionRecord:
    monday_date, agent_input, level1_plan, level2_plan, weekly_return,
    guardrail_events, tcn_prediction_errors, reasoning_summary, quality_label

class DecisionLogger:
    def log_decision(self, record: DecisionRecord) → None  # JSONL append
    def compute_tcn_error(self, tcn_sequence, actual_returns) → list[TCNPredictionError]
    def extract_good_bad_patterns(self) → tuple[good: list, bad: list]
    def assign_quality_labels(self, weekly_return, signal_alignment) → str
```

### 第 11 步：`src/agent/daily_guardrail.py`（新建）
```python
# FORBIDDEN_ZONE 禁闭期规则（按触发原因）
COOLDOWN_RULES = {
    "DAILY_LOSS_5PCT": 3,         # 单日跌>5%，禁3个交易日
    "DAILY_LOSS_3PCT_HIGH_BETA": 2,  # 跌>3%+高Beta，禁2个交易日
    "BREAKING_NEWS": None,          # 黑天鹅利空，禁到下周一Agent重新评估
    "MARKET_VOL_SPIKE": 1,         # 市场波动率飙升，禁1个交易日
}

class FORBIDDEN_ZONEStateMachine:
    def mark_forbidden(self, sector, reason, trigger_type, current_date) → None
    def is_forbidden(self, sector, current_date) → bool
    def get_forbidden_sectors(self, current_date) → list[str]
    def cooldown_expired(self, sector, current_date) → bool  # 检查 cooldown 是否到期
    def auto_release(self, sector, current_date) → bool  # 到期自动释放

class DailyGuardrailMonitor:
    def check_guardrail_trigger(self, current_date, positions, etf_prices, news_df) → list[GuardrailSignal]
    def emergency_exit(self, signal, current_date) → dict  # 标记 FORBIDDEN_ZONE
    def apply_forbidden_zone(self, agent_plan, current_date) → (adjusted_plan, overrides)
```
**自愈机制**：触发后按 `COOLDOWN_RULES` 计算禁闭期。黑天鹅类（BREAKING_NEWS）需要下周一 Agent 主动重新评估后才能解除；其他类型到期自动释放。

### 第 12 步：`src/agent/rule_engine.py`（新建）
```python
class WeeklyRuleEngine:
    def apply_weekly_rules(
        self, level1_plan, last_week_pnl, last_week_holdings, last_week_returns
    ) -> (adjusted_plan, violations):
        # 规则 1-6：权重上限、Beta惩罚、Mirror检查、亏损保护、最小操作阈值
```

### 第 13 步：`src/agent/tools.py`（修改）
新增 tool：
```python
@tool
def build_decision_context(date: str) -> str:
    """调用 AgentFeatureBuilder 构建 A/B/C/D/E 特征集"""
```
加入 `TOOL_REGISTRY`，并加入 `bound_tools` 列表。

### 第 14 步：`config/prompts/trader.md`（修改）
重写 prompt 模板：
- 输入：TCN 日频序列(A)、新闻摘要(B)、市场状态(C)、持仓状态(D)、量价博弈(E)
- FORBIDDEN_ZONE 检查、Good/Bad patterns 注入槽位、决策逻辑检查清单（5项）
- 输出：`level1_plan[]`（8元板块）+ `level2_plan[]`（ETF选择）+ `reasoning_summary`

### 第 15 步：`src/agent/prompts.py`（修改）
更新 `trader_prompt()` 函数签名，新增参数：
`tcn_sequence`, `news_summary`, `market_state`, `position_state`, `sent_p_divergence`, `forbidden_sectors`, `good_patterns`, `bad_patterns`

### 第 16 步：`src/agent/single_agent.py`（修改）
- `decide_node` 重构：
  - 调用 `AgentFeatureBuilder.build_agent_features()` 内部构建特征
  - 使用新的 `WeeklyTradePlanV2` JSON schema（`level1_plan`/`level2_plan`）
  - 从 `DecisionLogger` 注入 good/bad patterns
  - 调用 `WeeklyRuleEngine.apply_weekly_rules()` 做规则检查
- `trader_retry_node` 适配新 schema

### 第 17 步：`src/agent/workflow.py`（修改）
- 将 `build_decision_context` 加入 `bound_tools`
- `risk_check_node` 支持 `MetaSectorPlan`（8元板块）
- 日 Guardrail 作为带外进程运行

### 第 18 步：`src/agent/prompt_manager.py`（新建）
```python
class PromptManager:
    """专门负责动态拼接 good/bad patterns 到 prompt 模板中（Few-shot 动态注入）。

    核心功能：
      1. 从 decision_logs.jsonl 提取"相似场景"的 Good/Bad Decision
      2. 根据当前市场状态（高波动/趋势/板块信号强弱）动态召回历史经验
      3. 将 patterns 注入 trader.md 模板，供 decide_node 使用
    """

    def recall_similar_decisions(
        self,
        current_context: dict,  # {vol_percentile, sector_signal, market_state, ...}
        n: int = 5,
    ) -> list[DecisionRecord]:
        """根据当前市场状态召回"相似场景"的 Good Decision。

        匹配规则：
          - vol_percentile 高 → 召回历史高波动时的 Good Decision
          - 某板块信号强 → 召回历史同板块信号强时的 Good Decision
          - 触发过 FORBIDDEN_ZONE → 召回类似触发后的正确处理
        """
        # 简单规则匹配，或用 cosine similarity（将 context embedding）

    def load_patterns_by_context(
        self, current_context: dict, n: int = 5
    ) -> tuple[good: list[str], bad: list[str]]:
        """根据当前 context 召回相似场景的 patterns。"""

    def inject_patterns(self, good: list[str], bad: list[str]) -> str:
        """将 patterns 注入 trader.md 模板，返回渲染后的完整 prompt。"""

    def update_prompt(self, current_context: dict) -> str:
        """每次决策前调用：context → 召回 patterns → 渲染 prompt。"""
```

### 第 19 步：`scripts/run_phase2_dry_run.py`（新建）
```python
"""Phase 2 干跑脚本：运行 2024-10 ~ 2025-06 每周回测，生成 decision_logs.jsonl。

这是训练 Agent 决策能力的"练兵场"。
"""
def run_phase2_dry_run(
    start_date: str = "2024-10-01",
    end_date: str = "2025-06-30",
) -> Path:
    """运行 Phase 2 Dry Run。

    流程：
      1. 加载 TCN + LightGBM + IForest ONNX 模型
      2. export_phase2_dataset() 导出 agent_features.parquet
      3. 每周一：Agent 决策 → 执行 → 记录 Guardrail 事件 → 周末计算收益
      4. 生成 decision_logs.jsonl

    输出：data/decision_logs.jsonl（DecisionLogger 格式）
    """
```

---

## 关键设计决策

1. **Backward Compatibility**：现有 ONNX pipeline（8 大类）完全不受影响，新 TCN 是独立模型（训练用），推理时用新的 `agent_features.parquet`
2. **扇入式 TCN 替代 finetune_per_industry**：单模型学跨行业传导，比逐行业微调更充分利用 47 细分的联合信号；Spatial Dropout + L2 正则化防过拟合
3. **Label 平滑**：Winsorize → Z-score → Tanh 缩压，防止 A 股 5 日收益率极端值导致 MSE 梯度爆炸
4. **8 个独立 LightGBM**：每个元板块独立训练，避免强制共享结构导致不同板块驱动逻辑相互干扰；tcn_residual 替代 tcn_cls 让模型学"什么时候会错"
5. **日 Guardrail 优先级**：在 `apply_forbidden_zone()` 中强制覆盖 Agent 的 Level1 计划；不同触发原因对应不同 cooldown 期，黑天鹅需下周一 Agent 主动评估
6. **SHAP 只展示不决策**：存入 `decision_logs.jsonl`，供回测报告展示，不参与交易
7. **Look-ahead Bias 防护**：`features.py` 强制 `current_time` 参数，所有特征只取当前时刻之前的数据
8. **向量化批量推理**：`export_phase2_dataset()` 一次性批量推理 TCN/LightGBM，不逐日期循环

---

## 验证方式

1. **单元测试**：features.py, decision_logger.py, daily_guardrail.py, rule_engine.py 独立测试
2. **Phase 1 训练**：`python -m trainer.main signals train`，验证扇入式 TCN 收敛（loss下降、IC提升）
3. **SHAP 可视化**：运行后检查 `shap_values.csv` 和 summary plot 生成
4. **Phase 2 Dry Run**：`python main.py backtest --start-date 2024-10-01 --end-date 2025-06-30`
5. **Guardrail 触发**：模拟单日跌>5% 数据，验证 FORBIDDEN_ZONE 状态机
