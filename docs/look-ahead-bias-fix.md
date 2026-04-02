# Look-Ahead Bias 修复方案

## 问题概述

量化交易中，模型在训练时使用了实盘不可获取的"未来信息"，导致回测曲线虚高，实盘失效。本文档逐一修复 signals 模块中的 look-ahead bias。

---

## 修复原则：T-1 可见性

所有特征必须对齐到 `T-1` 时刻：

- **TCN 序列**：预测目标是 `T` 时刻的情感动量，输入窗口为 `[T-seq_len, T-1]`
- **LightGBM**：预测目标是 `T+1` 方向，输入特征全部来自 `T` 及之前
- **OHLCV**：当日收盘后才有完整数据，盘中只用 `T-1` 的 OHLCV

---

## 一、LightGBM 特征修复

文件：[trainer/signals/dataset.py](trainer/signals/dataset.py)

### 问题 1：`delta_sentiment` 用的是当期情感

**原代码（line 562-565）：**
```python
delta1 = vals[i] - vals[i - 1]   # 用的是 vals[i]（T 时刻情感）
delta2 = vals[i] - vals[i - 2]   # 用的是 vals[i]（T 时刻情感）
news_count = nc[i]               # 当期新闻数
news_heat = nh[i]               # 当期热度
```

**问题**：在预测 `T+1` 方向时，用了 `T` 时刻的情感，而 `T` 时刻的情感本身包含了当天全天的新闻，这是结果预测结果。

**修复后：**
```python
delta1 = vals[i - 1] - vals[i - 2]   # T-1 vs T-2 的情感变化（历史已知）
delta2 = vals[i - 1] - vals[i - 3]   # T-1 vs T-3 的情感变化（历史已知）
news_count = nc[i - 1]               # T-1 的新闻数
news_heat = nh[i - 1]               # T-1 的热度
```

### 问题 2：OHLCV 特征用的是当期数据

**原代码（line 638-642）：**
```python
vr = vol_arr[i] / (vol_ma5[i] + 1e-9)   # 当期成交额比值
iv = (high_arr[i] - low_arr[i]) / close_arr[i]  # 当期日内波动率
ap = (high_arr[i] + low_arr[i] + close_arr[i] + open_arr[i]) / 4.0  # 当期均价
```

**问题**：`T` 时刻的 OHLCV 在 `T` 收盘后才能完全确定，盘中不可见。

**修复后：**
```python
vr = np.where(vol_arr[i - 1] > 0, vol_arr[i - 1] / (vol_ma5[i - 1] + 1e-9), 0.0)  # T-1
iv = (high_arr[i - 1] - low_arr[i - 1]) / max(close_arr[i - 1], 1e-9)           # T-1
ap = (high_arr[i - 1] + low_arr[i - 1] + close_arr[i - 1] + open_arr[i - 1]) / 4.0  # T-1
```

`vol_ma5[i - 1]` 的窗口也要改为从 `i-5` 到 `i-1`：
```python
vol_ma5 = np.array([np.mean(vol_arr[max(0, j - 4):j + 1]) for j in range(i - seq_len, i)])
# 改为
vol_ma5 = np.array([np.mean(vol_arr[max(0, j - 5):j]) for j in range(i - seq_len, i)])
# 即：计算 vol_arr[i-1] 的 MA5 时，不包含 vol_arr[i-1] 本身
```

---

## 二、TCN vol_delta 修复

文件：[trainer/signals/dataset.py](trainer/signals/dataset.py)

### 问题：`vol_delta` 用了当期成交量

**原代码（line 304，`build_tcn_sequences`）：**
```python
vol_delta = np.clip((vol_arr[i] - vol_arr[i - 1]) / (vol_arr[i - 1] + 1), -1, 1)
# vol_arr[i] 是当期成交量
```

**修复后：**
```python
vol_delta = np.clip((vol_arr[i - 1] - vol_arr[i - 2]) / (vol_arr[i - 2] + 1), -1, 1)
# 用 T-1 vs T-2 的成交量变化，不含当期
```

**原代码（line 336-340，`_build_per_industry_sequences`）：**
```python
vol_delta = np.clip(
    (vol_arr[i, idx_industry] - vol_arr[i - 1, idx_industry]) / (vol_arr[i - 1, idx_industry] + 1),
    -1, 1,
)
```

**修复后：**
```python
vol_delta = np.clip(
    (vol_arr[i - 1, idx_industry] - vol_arr[i - 2, idx_industry]) / (vol_arr[i - 2, idx_industry] + 1),
    -1, 1,
)
```

---

## 三、TCN 序列窗口是否包含预测时刻的 volume_ratio？

### 问题描述

在 `build_sequences`（line 478-484）中：

```python
vol_ma5[i] = np.mean(vol_arr[max(0, i - 4):i + 1])  # 窗口包含 vol_arr[i]
volume_ratio[i] = vol_arr[i] / (vol_ma5[i] + 1e-9)  # 用 vol_arr[i]
```

这里 `vol_arr[i]` 在序列窗口 `i:i+seq_len` 内，但 TCN 的**预测目标是** `i+seq_len` 时刻的情感动量。

**语义分析**：如果把序列窗口理解为"从 `i` 到 `i+seq_len-1` 的历史"，则窗口内每个时刻 `j` 的 `volume_ratio[j]` 包含 `vol_arr[j]`——这对于 `j` 时刻是"未来数据"。但这个 bias 比 LightGBM 轻，因为 TCN 的 label 是 `i+seq_len` 时刻的情感变化，不是同周期的市场结果。

**推荐修复**：将 `volume_ratio` 的 MA5 窗口右端点收缩 1：

```python
# 修复后：MA5 窗口为 [i-4, i]（不含 vol_arr[i] 本身）
vol_ma5 = np.array([np.mean(vol_arr[max(0, j - 5):j]) for j in range(i, i + seq_len)])
volume_ratio = np.where(vol_arr[i:i + seq_len] > 0, vol_arr[i:i + seq_len] / (vol_ma5 + 1e-9), 0.0)
```

---

## 四、IsolationForest 数据泄漏检查

文件：[trainer/signals/dataset.py](trainer/signals/dataset.py) `build_isolation_forest_dataset`

```python
for i in range(len(periods)):
    vol_norm = vol_arr[i] / (vol_arr[i - 1] + 1)   # ← vol_arr[i] 当期
    sent_diff = sent_arr[i] - sent_arr[i - 1]       # ← sent_arr[i] 当期
    sent_cur = sent_arr[i]                           # ← sent_arr[i] 当期
```

**问题**：IsolationForest 用当期成交量和情感作为异常标签，但 IsolationForest 本身是**同时点**的判别，不存在"预测下一时刻"的问题。所以这里的数据泄漏相对轻——它不会导致"用未来信息预测未来"，而是"用当期的综合表现判断当期是否异常"。

**结论**：IsolationForest 部分可接受当前逻辑，无需修改。

---

## 五、新闻时间戳过滤（数据源层）

文件：`WeeklySignalDataset._load_raw`

当前代码没有对新闻的 `publish_time` 做任何 `< T` 过滤：

```python
df = pl.read_parquet(self.raw_path)
df = df.with_columns(
    pl.col("datetime").str.to_datetime(),  # ← 没有过滤：只要在 period 内的新闻全算
    ...
)
```

**问题**：在构建 `T` 时刻的情感时，如果用的是 `T` 日全天的新闻，那么下午 4 点发的新闻实际上不能用来预测当天开盘到收盘的涨跌。

**修复建议**（数据源层面，更彻底）：

```python
# 假设每条新闻有 publish_time，对齐到 period 后做可见性过滤
# weekly 模式：period 是 Mon-Sun，用上周六 15:00 到本周一 9:00 的新闻
# daily 模式：用前一天 15:00 到当天 9:00 的新闻（隔夜新闻）

if self.freq == "daily":
    # T 日 9:00 开盘，只用前一天 15:00 之后的新闻
    cutoff = pl.col("datetime").dt.truncate("1d")  # 当天 00:00
    df = df.filter(pl.col("datetime") < pl.lit(cutoff).cast(pl.Datetime))  # 只保留 < T 日的新闻
else:
    # weekly：T 周的情感只用 T-1 周六 15:00 到 T 周一 9:00 的新闻
    ...
```

---

## 六、修复验证：Shift-1 强制检查

在数据处理最后，增量加一道 `shift(1)` 强制解耦，防止未来函数：

```python
def build_safe_features(raw_df):
    # 1. 计算原始指标（基于当天）
    raw_df['vol_ratio_raw'] = raw_df['volume'] / raw_df['volume'].rolling(5).mean()
    raw_df['sent_mean_raw'] = raw_df['sentiment_score']

    # 2. 核心：全体下移一行 —— 这样 feature_df 中 i 行的数据，物理含义是 i-1 日
    feature_df = raw_df[['vol_ratio_raw', 'sentiment_mean_raw', ...]].shift(1)

    # 3. label 保持不动：用 T-1 的特征预测 T 的表现
    feature_df['label'] = raw_df['target_direction']

    return feature_df.dropna()
```

---

## 七、修复优先级汇总

| 优先级 | 模块 | 修复内容 |
|--------|------|---------|
| **P0（必改）** | LightGBM features | `delta1/delta2` 改为 `vals[i-1]/vals[i-2]`，`vr/iv/ap` 改为 `T-1` |
| **P0（必改）** | LightGBM `vol_ma5` | 窗口改为 `[j-5, j)` 不含当期 |
| **P1（高优）** | TCN `vol_delta` | 改为 `vol_arr[i-1] - vol_arr[i-2]` |
| **P1（高优）** | 新闻时间戳 | 增加 `< T` 的可见性过滤 |
| **P2（可选）** | TCN `volume_ratio` | MA5 窗口右端收缩 1 格 |
| **P3（可不做）** | IsolationForest | 当前逻辑可接受 |

---

## 八、修复后预期

- LightGBM 回测曲线大概率**显著下降**（因为不再"偷看"当天市场结果）
- 下降不是模型变差，是**变真实**
- 只有通过 T-1 校验的策略才具备实盘资格
- 建议修复后重新跑完整的 train/val/test 分段回测，对比修复前后的 Sharpe Ratio、夏普比等风险指标
