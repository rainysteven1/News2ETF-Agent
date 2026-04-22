# News2ETF-Agent

News2ETF-Agent 是一个围绕“新闻 -> 情绪/行业标签 -> 信号模型 -> LLM Agent -> ETF 回测”构建的周频 ETF 轮动系统。

它的目标不是只训练一个模型，而是把以下几件事串成一条可复现、可部署、可评估的完整链路：

1. 用 ONNX 分类器对新闻做一级行业 / 二级子行业 / 情绪标注。
2. 用 `signals` 模块把新闻与行情转成可部署的元板块信号。
3. 让 Agent 在只使用“当下可得信息”的前提下做板块配置与 ETF 选择。
4. 用严格区分训练期与推理期的方式进行回测，避免 look-ahead bias。

如果你只想快速上手，先看本文档；如果你要完整理解系统设计、数据契约和运行规范，请直接看：

- 标准说明设计文档：`docs/system-design-spec.md`

## 你会在这个仓库里看到什么

系统主链路如下：

```text
raw news / labeled news
  -> predict major / predict sub
  -> signals train
  -> signals export-onnx
  -> signals infer
  -> agent decision_context
  -> weekly backtest
```

其中：

- `trainer/` 负责训练、导出、离线推理。
- `runtime/agent/` 负责运行 Agent、做单周决策、回测与可视化。
- `data/` 保存共享数据、映射表与导出的特征缓存。
- `docs/` 保存架构说明、策略设计、偏差修复与历史方案文档。

## 5 分钟快速上手

### 1. 环境准备

建议环境：

- Python 3.11+
- `uv`
- `just`
- 可用的 `.env`（用于 LLM / W&B / 其他运行时密钥）

安装依赖：

```bash
just cpu-sync
```

如果你本机需要 GPU 训练环境：

```bash
just gpu-sync
```

### 2. 查看当前运行所需产物是否齐全

如果你主要想直接跑 Agent / backtest：

```bash
just runtime-check-artifacts
```

这个命令会检查运行时最关键的两个输入是否已经就绪：

- `runtime/agent/data/inputs/sentiment_weekly.parquet`
- `runtime/agent/models/signals/final-3y/manifest.json`

如果这些产物还在 `trainer/` 下，而没有迁移到 `runtime/agent/`，可以执行：

```bash
just runtime-migrate-artifacts
```

### 3. 直接运行一次回测

如果运行时产物已经就绪，最短路径是直接跑周频回测：

```bash
python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
```

或者使用 `just`：

```bash
just backtest 2024-01-01 2024-12-31
```

### 4. 调试单周决策

如果你想看某一周 Agent 是如何做决策的：

```bash
python runtime/agent/main.py decide --week 2024-06-03
```

或：

```bash
just decide 2024-06-03
```

## 推荐使用路径

### 路径 A：我只想快速体验回测

按这个顺序：

```bash
just cpu-sync
just runtime-check-artifacts
just runtime-migrate-artifacts   # 仅在需要时
python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
```

适合：

- 想先确认项目能跑起来
- 想看 Agent 回测结果
- 暂时不关心训练细节

### 路径 B：我想完整跑一遍标准 pipeline

按这个顺序：

```bash
just signals-train-final-3y
just signals-export-onnx-final-3y
just signals-infer-2024
just backtest-2024
```

或者直接一条命令：

```bash
just signals-agent-pipeline-2024
```

适合：

- 想严格区分训练 / 推理 / 回测
- 想复现实验标准流程
- 想生成新的 `agent_features.oof.parquet`

### 路径 C：我想先做开发集验证，再做最终评估

开发阶段推荐的 4 年切分是 `2 + 1 + 1`：

- 2021-2022：训练
- 2023：验证 / OOF
- 2024：留作 Agent + Backtest holdout

对应命令：

```bash
just signals-train-dev-2y1y
just signals-export-onnx-dev-2y1y
```

模型确认后，再使用最终 `3 + 1` 切分：

```bash
just signals-train-final-3y
just signals-export-onnx-final-3y
just signals-infer-2024
just backtest-2024
```

## 仓库结构

```text
README.md
justfile
pyproject.toml
config/prompts/                 # Prompt 模板

data/                          # 共享数据、映射表、导出的 agent feature

docs/                          # 架构、计划、偏差修复、策略说明文档

trainer/                       # 训练 / 导出 / 离线推理 CLI 与数据集
  main.py
  config.toml
  src/

runtime/agent/                 # 运行时 Agent、backtest、可视化
  main.py
  config.toml
  src/
```

## 常用命令速查

### 依赖与环境

```bash
just cpu-sync
just gpu-sync
```

### 训练与导出

```bash
python -m trainer.main signals train
python -m trainer.main signals export-onnx --checkpoint-dir ./trainer/checkpoints/signals/final-3y --bundle-dir ./trainer/models/signals/final-3y
python -m trainer.main signals infer --bundle-dir ./trainer/models/signals/final-3y --output-path ./data/agent_features.oof.parquet --start-date 2024-01-01 --end-date 2024-12-31
```

### 运行 Agent / Backtest

```bash
python runtime/agent/main.py decide --week 2024-06-03
python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
python runtime/agent/main.py diagnose-backtest --run-id bt_example
python runtime/agent/main.py visualize-backtest --run-id bt_example
```

### Docker

```bash
just docker-build-runtime
just docker-backtest 2024-01-01 2024-12-31
just docker-backtest-run bt_demo 2024-01-01 2024-12-31
just docker-backtest-2024
```

## 文档阅读顺序

如果你是第一次接触这个项目，推荐按下面顺序阅读：

1. `README.md`
2. `docs/system-design-spec.md`
3. `runtime/agent/README.md`

## 文档优先级说明

为了避免历史文档之间的口径不一致，建议按以下优先级理解项目：

1. 代码与配置：`trainer/`、`runtime/agent/`、`trainer/config.toml`、`runtime/agent/config.toml`
2. 标准说明设计文档：`docs/system-design-spec.md`

也就是说：

- 如果其他说明和当前代码冲突，以代码和标准设计文档为准。
- 如果你准备修改 `signals`、`agent`、`backtest` 中任一模块，建议同时更新 `docs/system-design-spec.md`。

## 一句话理解当前标准主链

当前推荐、也是最安全的使用方式是：

```text
signals train -> export ONNX bundle -> infer held-out features -> agent/backtest only consume infer outputs
```

这条主链的核心目标只有一个：

> 让 Agent 的输入尽可能接近真实线上可获得的信息，避免训练期信息污染回测结果。
