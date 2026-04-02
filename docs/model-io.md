# 模型输入输出说明

## 1. TCN (Temporal CNN) — 时序卷积网络

**用途**: 情感动量预测 + 异常检测

### 输入

```
(batch, seq_len=5, input_size=6)
```

| 通道 | 名称 | 说明 |
|------|------|------|
| ch0 | sentiment_mean | 行业日均情感加权均值，标准化到 [-1, 1] |
| ch1 | sentiment_std | 行业日均情感标准差 |
| ch2 | news_count | 当日新闻数量 |
| ch3 | avg_confidence | 当日情感置信度均值 |
| ch4 | volume_ratio | 成交额比值 = Volume / MA(Volume, 5)，上限 100 |
| ch5 | intraday_vol | 日内波动率 = (High - Low) / Close |

### 输出

| 头 | 形状 | 激活函数 | 说明 |
|----|------|----------|------|
| reg_output | (batch, 1) | tanh | 情感动量分数，范围 [-1, 1] |
| cls_output | (batch, 1) | sigmoid | 异常概率，范围 [0, 1] |

---

## 2. LightGBM — 堆叠模型

**用途**: 最终方向预测（信号生成）

### 输入

```
(N, 13)
```

| 特征索引 | 名称 | 说明 |
|----------|------|------|
| 0 | delta_sentiment_1w | 情感变化（当日 - 前1日） |
| 1 | delta_sentiment_2w | 情感变化（当日 - 前2日） |
| 2 | news_count | 当日新闻数量 |
| 3 | news_heat | 新闻热度 = news_count × avg_confidence |
| 4 | tcn_reg | TCN 回归头输出（动量分数） |
| 5 | tcn_cls | TCN 分类头输出（异常概率） |
| 6 | tcn_reg_delta | TCN 动量变化（当前 - 上一步） |
| 7 | news_count_std_5d | 5日新闻数量标准差 |
| 8 | sentiment_volatility_5d | 5日情感波动率（标准差） |
| 9 | tcn_reg × news_heat | TCN动量与新闻热度交互项 |
| 10 | volume_ratio | 成交额比值（同TCN ch4） |
| 11 | intraday_vol | 日内波动率（同TCN ch5） |
| 12 | avg_price | OHLCV平均价 = (High + Low + Close + Open) / 4 |

### 输出

```
方向预测: +1（情感上升）, -1（情感下降）, 0（持平）
```

---

## 3. IsolationForest — 异常检测

**用途**: 识别异常市场状态（用于特征工程，非直接信号）

### 输入

```
(N, seq_len=5, channels=5)
```

| 通道 | 名称 | 说明 |
|------|------|------|
| ch0 | news_count_seq | 5日新闻数量序列（归一化） |
| ch1 | news_heat_seq | 5日新闻热度序列 |
| ch2 | amount_seq | 5日ETF成交额序列 |
| ch3 | amount_roll_mean_seq | 5日成交额滚动均值 |
| ch4 | amount_roll_std_seq | 5日成交额滚动标准差 |

### 输出

| 值 | 含义 |
|----|------|
| +1 | 异常（anomaly） |
| -1 | 正常（normal） |

---

## 数据流向

```
原始新闻数据
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  WeeklySignalDataset                                 │
│  - 聚合情感: sentiment_mean, sentiment_std, avg_conf │
│  - 关联OHLCV: amount, high, low, close, open, volume│
│  - 计算return: (close-open)/open                      │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ build_sequences │
         │ (N, 5, 6) TCN输入 │
         └───────┬───────┘
                 │
         ┌───────▼───────┐
         │  TCN 预训练    │
         │  (混合全行业)  │
         └───────┬───────┘
                 │
         ┌───────▼───────┐
         │  TCN 微调     │
         │  (按行业)     │
         └───────┬───────┘
                 │
         ┌───────▼──────────┐
         │ build_lgbm_features │
         │ (N, 13) LGBM输入  │
         └───────┬──────────┘
                 │
         ┌───────▼───────┐
         │  LightGBM     │
         │  方向预测     │
         └───────────────┘
```
