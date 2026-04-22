# News2ETF-Agent 标准说明设计文档

## 1. 文档目的

这份文档是当前仓库关于 `predict -> signals -> infer -> agent -> backtest` 主链路的标准说明文档，目标是把 `docs/` 中已有的多个设计草稿、计划文档、偏差修复说明和运行约束，统一成一份可以直接指导开发、运行、评审和交接的设计规范。

它解决的核心问题不是“某个模型怎么训练”，而是以下四件事如何在一个系统里正确衔接：

1. 训练阶段到底产出什么。
2. 推理阶段到底允许消费什么。
3. Agent 决策阶段到底基于什么上下文。
4. 回测阶段到底验证的是什么能力。

这份文档的定位是“当前标准口径”，而不是历史思路集合。因此：

- 它已经吸收仓库里旧版架构草稿、计划文档与偏差修复说明中仍然有效的内容。
- 它会显式解决这些文档之间的口径差异。
- 它会以代码与配置现状为最终校准依据。

---

## 2. 文档适用范围

本说明文档覆盖：

- 新闻标注与离线预测链路（`predict major` / `predict sub`）
- `signals` 训练、导出与推理链路
- Agent 决策上下文构建
- 周频回测与可视化
- 数据切分、产物落盘、运行命令、风控约束
- 防止 look-ahead bias 的基础规则

本说明文档不细化以下内容：

- Prompt 具体措辞与提示词修辞细节
- 某家 LLM 供应商的调用策略
- 所有实验性策略分支
- 每一种研究类工具的探索性用法

这些属于实现层、实验层或运维层的延伸内容，不是这份文档的主契约。

---

## 3. 文档优先级与单一真相源

为了避免历史文档之间互相打架，当前项目采用如下优先级：

### 3.1 第一优先级：代码与配置

以下内容是最终真实口径：

- `trainer/main.py`
- `trainer/config.toml`
- `trainer/src/**`
- `runtime/agent/main.py`
- `runtime/agent/config.toml`
- `runtime/agent/src/**`
- `data/label_stats.json`
- `data/meta_sector_mapping.json`

### 3.2 第二优先级：本标准说明文档

本文件的作用是把代码中已经稳定的事实组织成一份可读的、系统性的规范文档。

### 3.3 归档原则

历史设计稿已经合并进本文件，`docs/` 目录不再保留重复、过期或口径冲突的设计文档。

如果你需要理解历史演进，请以 git 历史记录为准；如果你需要理解当前标准，请以代码和本标准说明文档为准。

---

## 4. 项目一句话定义

News2ETF-Agent 是一个面向中文财经新闻与主题 ETF 的周频多阶段决策系统：它先把新闻映射成结构化标签，再把新闻与行情压缩成元板块信号，随后由 Agent 在多源上下文下生成配置方案，最终通过只消费推理期输入的周频回测来验证策略有效性。

---

## 5. 核心设计目标

当前标准设计有五个不可偏离的目标：

### 5.1 部署优先，而不是训练优先

训练阶段必须为推理阶段服务。

也就是说：

- 不是只要训练指标好就行；
- 训练结果必须能够导出为稳定的 ONNX bundle；
- 运行时必须能在没有训练态 Python 对象的前提下完成推理。

### 5.2 Agent 输入必须接近真实线上环境

Agent 和 backtest 默认应优先消费：

- `agent_features.oof.parquet`，或
- 固定 ONNX bundle 推理出的特征

而不是训练阶段顺手生成的全历史特征或训练标签。

### 5.3 回测验证的是“决策系统”，不是单模型离线指标

回测要回答的问题是：

- 当 Agent 只拿到推理期可见信息时，能否做出合理的板块配置和 ETF 选择？

回测不直接回答：

- TCN 单独有多强；
- LightGBM 单独有多强；
- 训练 loss 是否最低。

### 5.4 数据契约必须稳定

这意味着：

- canonical 子行业维度不能因样本年份变化而动态缩水；
- 元板块映射不能在代码里偷偷存在多份逻辑；
- 推理产物字段不能随意漂移。

### 5.5 明确防止未来函数与信息泄漏

任何特征、上下文或回测流程，只要让决策阶段看到了训练期或未来时点才能知道的信息，都属于架构错误，而不是“小问题”。

---

## 6. 术语表

### 6.1 Major Category

一级行业分类。由 `predict major` 阶段负责。

### 6.2 Sub Category

二级子行业分类。由 `predict sub` 阶段负责。

### 6.3 Sentiment

对新闻情绪的分类结果，通常为 positive / neutral / negative。

### 6.4 Canonical Sub-Category Space

由 `data/label_stats.json` 定义的、训练和推理共享的稳定子行业空间。当前代码口径下，`by_sub_category` 的数量为 46。

### 6.5 Meta Sector

由 `data/meta_sector_mapping.json` 定义的 8 个元板块，是 Agent 一级配置和回测归因的核心维度。当前包括：

- `科技成长`
- `高端制造`
- `消费文娱`
- `医药健康`
- `资源材料`
- `金融地产`
- `基础设施/公共`
- `主题策略`

### 6.6 OOF

Out-of-fold / walk-forward 口径下的推理特征产物，用于更接近真实推理条件的验证。

### 6.7 ONNX Bundle

`signals export-onnx` 导出的部署包，至少包含：

- `tcn.onnx`
- `manifest.json`

并尽量包含：

- `lgbm/*.onnx`
- `iforest.onnx`

### 6.8 Agent Feature Cache

提供给 Agent 直接消费的结构化特征 parquet。当前主要有两个：

- `data/agent_features.parquet`
- `data/agent_features.oof.parquet`

其中 OOF / infer 版本优先级更高。

---

## 7. 总体架构

### 7.1 标准主链

```text
原始新闻 / 已标注新闻
  -> predict major
  -> predict sub
  -> trainer/data/labeled/signals/raw.parquet
  -> signals train
  -> checkpoint + full-history features + OOF features
  -> signals export-onnx
  -> signals infer
  -> data/agent_features.oof.parquet
  -> runtime agent build_decision_context
  -> level1_plan + level2_plan
  -> weekly backtest + diagnostics + visualization
```

### 7.2 各层职责

#### A. 预测标注层

负责把新闻数据变成可被 `signals` 模块消费的结构化标签：

- 一级行业
- 二级子行业
- 情绪
- 置信度

#### B. 信号建模层

负责把新闻情绪与行情组合成元板块层面的结构化信号：

- TCN 时序输出
- LightGBM 元板块得分
- 热度异常
- 其他稳定性或残差特征

#### C. 推理交付层

负责把训练期产物转成运行期稳定输入：

- ONNX deployment bundle
- 指定时间段的 `agent_features.oof.parquet`

#### D. Agent 决策层

负责在多源上下文下生成：

- Level 1 元板块配置
- Level 2 ETF 选择
- 结构化理由说明

#### E. 回测评估层

负责验证 Agent 在只拿到推理期可见信息时的真实表现，并输出：

- 收益曲线
- 周收益
- 回撤
- 元板块贡献
- 选中的 ETF
- 可视化报告

---

## 8. 系统目录与模块分工

### 8.1 `trainer/`

主要负责训练和离线推理：

- `trainer/main.py`：统一 CLI 入口
- `trainer/config.toml`：训练配置
- `trainer/src/datasets/`：数据集构建
- `trainer/src/models/`：模型定义与导出
- `trainer/src/pipelines/`：训练 / 预测 / 推理流程

### 8.2 `runtime/agent/`

主要负责运行时决策与回测：

- `runtime/agent/main.py`：运行时 CLI
- `runtime/agent/config.toml`：运行时配置
- `runtime/agent/src/agent/`：Agent、工具、状态、规则、日志
- `runtime/agent/src/backtest/`：回测引擎、指标、诊断、可视化
- `runtime/agent/src/signals/`：运行时 ONNX 推理与辅助逻辑
- `runtime/agent/src/utils/`：新闻、ETF、映射、量价特征工具

### 8.3 `data/`

共享数据层，主要包含：

- `data/label_stats.json`
- `data/meta_sector_mapping.json`
- `data/industry_dict.json`
- `data/agent_features.parquet`
- `data/agent_features.oof.parquet`
- 行情、ETF 信息与其他转换后的 parquet 数据

### 8.4 `docs/`

文档层，保存：

- 现行主架构说明
- 策略设计说明
- 偏差修复原则
- 历史计划与演进记录

---

## 9. 数据流详细说明

## 9.1 原始新闻到标签化新闻

原始新闻先经过两级离线推理：

1. `predict major`
2. `predict sub`

形成可供 `signals` 使用的标签数据。该阶段产物通常位于 `trainer/data/labeled/signals/` 下。

这一层的目标是把非结构化文本压缩成后续时序建模所需的最小表达：

- 行业标签
- 子行业标签
- 情绪标签
- 置信度
- 时间戳

## 9.2 标签化新闻到 `raw.parquet`

`signals` 的训练输入标准路径由 `trainer/config.toml` 指定：

- `signals.dataset.raw_data_path = ./trainer/data/labeled/signals/raw.parquet`

如果这个文件不存在，当前系统允许从：

- `trainer/data/labeled/signals/raw/.raw_sub_monthly_checkpoints/`

自动重建，但前提是：

- 月份序列连续
- 每个 parquet 都可读
- 中间没有缺月

这是一个显式设计：

- 训练应该对数据缺口“失败得很早”，
- 不能悄悄用残缺数据训练。

## 9.3 `raw.parquet` 到 `signals train`

`signals train` 的主要职责：

- 构建 TCN 数据集
- 构建 LightGBM 特征
- 构建异常热度检测特征
- 执行 walk-forward / OOF 训练逻辑
- 导出 checkpoint
- 写出 Agent 可用特征缓存

当前训练后会输出：

- `tcn_fanin.pt`
- `lgbm/*.txt`
- `iforest_model.pkl`
- `signals_checkpoint.json`
- `latest.txt`
- `data/agent_features.parquet`
- `data/agent_features.oof.parquet`

## 9.4 `signals train` 到 ONNX bundle

`signals export-onnx` 的目标是把训练态模型转成运行态可部署产物。

部署包目录标准内容：

- `tcn.onnx`
- `manifest.json`
- `lgbm/*.onnx`（如果导出成功）
- `iforest.onnx`（如果导出成功）

最终 bundle 目录通常位于：

- `trainer/models/signals/dev-2y1y/`
- `trainer/models/signals/final-3y/`

## 9.5 ONNX bundle 到 held-out / future inference

`signals infer` 使用固定 bundle，在给定日期区间内生成运行态特征：

- 输入：ONNX bundle + sentiment / labeled news 数据
- 输出：`agent_features.oof.parquet`

这一步的目的非常关键：

- 把“训练阶段的模型”与“回测阶段消费的特征”硬性分离开；
- 避免 Agent 直接读取训练过程中的便利数据。

## 9.6 `agent_features.oof.parquet` 到 Agent 决策

运行时的 `AgentFeatureBuilder` 优先读取：

1. `output_agent_features_oof`
2. `output_agent_features`

如果缓存缺失，运行时才会尝试使用 signals ONNX bundle 进行补推。

这保证了：

- 有现成 held-out 特征时，优先走最稳定、最可复现的路径；
- 只有在确实缺失时，才退回运行时推理。

## 9.7 Agent 到 Backtest

Agent 不再直接以“细行业列表”作为主输出，而是以“元板块配置 + ETF 选择”作为标准输出：

- `level1_plan[]`：8 个元板块的动作与权重
- `level2_plan[]`：在允许买入的元板块内选择 ETF

回测则按这些输出计算：

- 周收益
- 净值
- drawdown
- `meta_sector_returns`
- `meta_sector_contributions`
- `selected_etfs`

---

## 10. 核心数据契约

### 10.1 Canonical 子行业空间契约

当前 canonical 子行业空间来自 `data/label_stats.json` 中的 `by_sub_category`。

标准约束：

- 子行业维度是固定空间，不因年份减少而减少。
- 某一年的缺失类别可以是全零列，但不能直接删维度。
- 训练、导出、推理必须共享同一套子行业顺序。

这样做的原因是：

- TCN 输入 shape 稳定；
- ONNX 导出 shape 稳定；
- 运行时 schema 稳定；
- 不同年份之间可直接比较。

### 10.2 元板块映射契约

元板块映射来自 `data/meta_sector_mapping.json`。

标准约束：

- 元板块定义只允许在这里集中维护；
- 代码中不应该再有一套平行、隐藏、无法同步的映射；
- Agent、signals、backtest 的元板块口径必须一致。

当前标准元板块数量为 8。

### 10.3 运行时路径默认值契约

运行时默认路径由 `runtime/agent/src/config.py` 自动填充，典型包括：

- `output_sentiment` -> `runtime/agent/data/inputs/sentiment_weekly.parquet`
- `output_backtest` -> `runtime/agent/data/backtest_results.parquet`
- `output_backtest_metrics` -> `runtime/agent/data/backtest_metrics.parquet`
- `output_agent_features` -> `data/agent_features.parquet`
- `output_agent_features_oof` -> `data/agent_features.oof.parquet`
- `signals_onnx_dir` -> `runtime/agent/models/signals/final-3y`

这意味着：

- `trainer/` 与 `runtime/agent/` 之间存在一个“训练产物迁移到运行产物”的边界；
- `just runtime-migrate-artifacts` 是官方提供的迁移路径之一。

### 10.4 Agent Feature Cache 契约

标准 feature cache 分为两类：

#### `data/agent_features.parquet`

用途：

- 全历史调试
- schema 检查
- 快速 smoke test

不建议把它当成 held-out 评估的默认输入。

#### `data/agent_features.oof.parquet`

用途：

- held-out 验证
- 未来日期推理
- Agent 运行时特征缓存
- Backtest 输入

这是当前推荐、优先级最高的标准输入。

---

## 11. 模型设计契约

### 11.1 新闻标注模型

项目中存在 major / sub 两级 ONNX 标注链路，用于把原始新闻转为结构化标签。

在标准主链中，这些模型负责“文本理解前置层”，而不是最终投资决策。

它们的职责是：

- 提供稳定标签空间
- 保证后续 `signals` 能在结构化数据上建模
- 在 runtime 必要时对原始新闻做 fallback 标注

### 11.2 Signals 主模型组合

`signals` 当前是组合式建模，而不是单一模型：

- TCN：学习跨子行业的时序模式
- LightGBM：做元板块级综合打分
- IsolationForest：检测新闻热度异常 / 异常状态

三者共同为 Agent 提供结构化、板块级的可消费信号。

### 11.3 TCN 契约

按照当前代码与训练配置的标准口径：

```text
input shape:
  (batch, seq_len=10, n_sub=46, channels=6)

output shape:
  (batch, 8)
```

6 个输入通道为：

- `sentiment_ema`
- `sentiment_acceleration`
- `sentiment_std`
- `log_news_count`
- `event_type_score`
- `sentiment_vs_price_residual`

输出为 8 个元板块的前瞻性分数。

需要特别说明的是：

- 历史计划文档中曾出现过“47 维”“5 天 TCN 输入”等表述；
- 当前标准实现以 `label_stats.json` 与训练配置为准，即 canonical 子行业数 46、训练时 `sequence_length = 10`；
- Agent 决策层消费的 `tcn_sequence` 是从运行态特征中回看 5 个交易日的摘要序列，这与训练时 TCN 底层窗口长度不是一个概念。

### 11.4 TCN 目标契约

当前默认目标模式来自 `trainer/config.toml`：

- `target_mode = "meta_excess_return"`

标准含义是：

- TCN 及后续综合信号以元板块超额收益方向为主目标；
- 不再把“未来情绪变化本身”当成主任务目标。

### 11.5 LightGBM 契约

LightGBM 当前保持“每个元板块一个模型”的结构。

它的硬约束是：

- 所有训练期使用的特征，都必须在运行时推理阶段可重建；
- 不允许依赖 target 派生字段；
- 不允许依赖只有训练态才存在的中间变量。

### 11.6 IsolationForest 契约

IsolationForest 的角色是：

- 检测新闻热度异常
- 输出稳定的“异常新闻状态”信号

它不是主方向预测模型，也不应被直接当作交易主引擎。

### 11.7 SHAP 契约

SHAP 仅用于离线解释和检查，不直接进入线上决策闭环。

标准用途包括：

- 识别 LightGBM 主导特征
- 分析是否过度依赖某类 TCN 派生特征
- 支持 ablation 与诊断

不允许用途包括：

- 直接把 SHAP 结论硬编码成交易规则并替代推理输出

---

## 12. Agent 设计契约

### 12.1 Agent 的定位

Agent 不是一个“看到新闻就直接回答该买什么”的一次性问答器，而是一个多源上下文融合后的周频决策器。

它至少要整合：

- 模型信号
- 新闻摘要
- 市场状态
- 当前持仓与近期绩效
- 历史记忆 / Memos
- 风险规则

### 12.2 Agent 输出分层

当前标准输出分为两层：

#### Level 1：元板块配置

决定：

- 哪些元板块可以买 / 持有 / 卖出
- 每个元板块建议权重是多少

#### Level 2：ETF 选择

决定：

- 每个被允许配置的元板块具体买哪只 ETF

这样做的原因是：

- 板块判断与 ETF 选择是两个不同问题；
- 先做元板块判断更稳定、更可解释；
- 回测归因也更清晰。

### 12.3 决策上下文构建

`build_decision_context()` 是当前 Agent 上下文构建的标准入口。

它会统一产出：

- 结构化 JSON
- `schema_version`
- `human_summary`

其中 `human_summary` 用于提升 Prompt 可读性，但不改变底层数据契约。

### 12.4 决策上下文的五类核心特征

当前 Agent 特征构造遵循 A/B/C/D/E 五类体系。

#### Feature A：时序模型特征

包括：

- `tcn_sequence`
- `ml_signal_snapshot`
- 残差与稳定性类字段
- 运行期板块快照

作用：

- 给 Agent 一个结构化的模型先验；
- 减少纯文本阅读的主观波动。

#### Feature B：新闻摘要

包括：

- 周内新闻摘要
- 板块 top news
- major/sub/sentiment 结果

作用：

- 把原始新闻转成可供推理消费的结构化语义背景。

#### Feature C：市场状态

包括：

- 市场收益
- 波动率
- `volume_ratio`
- 市场状态标签

作用：

- 告诉 Agent 当前适合进攻还是防守。

#### Feature D：持仓与绩效状态

包括：

- 当前持仓
- 周收益
- `agent_perf_1w`
- `agent_perf_4w`
- top holdings

作用：

- 防止 Agent 与组合现实状态脱节；
- 让系统具备“行为闭环”能力。

#### Feature E：情绪-价格背离

包括：

- `sent_p_divergence`

作用：

- 给 Agent 一个“预期差”视角；
- 避免既只追价格，也只追情绪。

### 12.5 Agent 运行机制

当前 Agent 采用研究与决策分离的方式：

- Researcher：负责信息收集与工具调用
- Trader：负责生成最终方案
- Rule Engine：负责周级规则约束
- Daily Guardrail：负责周内紧急退出与禁闭区管理

这意味着：

- 模型可以提出一个高层判断；
- 规则系统仍然有权做保护性修正；
- 风控层不替代研究层，但限制其输出边界。

---

## 13. 风控与执行约束

### 13.1 Weekly Rule Engine

当前周级规则引擎至少覆盖以下约束：

- 单板块权重上限
- 组合总仓位上限
- 上周亏损后的风险收缩
- 过度集中或镜像暴露检查
- 最小交易阈值

其目标不是“生成 alpha”，而是：

- 把 Agent 输出修正到可执行范围；
- 避免明显不合理的极端配置。

### 13.2 Daily Guardrail

Daily Guardrail 负责周内的保护性机制，当前主要包括：

- forbidden zone 管理
- cooldown 周期
- 极端波动触发
- 数据缺失风险
- 硬风险事件处理

它的定位是：

- 周内只做防守，不做主动择时增强；
- 避免把“周频配置器”变成“日内频繁交易器”。

### 13.3 风控层和 Agent 的关系

标准关系是：

- Agent 负责形成投资观点；
- Rule Engine 负责周级约束；
- Guardrail 负责异常情形下的保护；
- Backtest 负责验证这一整套机制的组合行为。

---

## 14. Backtest 契约

### 14.1 Backtest 验证对象

标准 backtest 验证的是整个决策系统：

```text
OOF / infer features
  + news
  + market state
  + holdings
  + memory
  -> agent decision
  -> rule adjustment
  -> weekly portfolio result
```

因此它不应被解释为：

- 训练集上模型拟合得有多好；
- 某个单模型 standalone 的收益水平。

### 14.2 持仓口径

当前回测主口径已经切换为元板块：

- holdings：`meta_sector -> weight`
- 明确保留 `selected_etfs`
- 元板块收益和贡献被单独记录

### 14.3 回测输出

标准输出包括：

- `runtime/agent/data/backtest_results.parquet`
- `runtime/agent/data/backtest_metrics.parquet`
- `runtime/agent/checkpoints/{run_id}/visualizations/report.html`
- `runtime/agent/checkpoints/{run_id}/visualizations/summary.json`

在条件满足时，还会生成：

- 各类 Plotly HTML 图
- PNG 图像
- W&B 媒体日志

### 14.4 结果解释边界

使用 backtest 结果时应明确：

- 它是系统级评估，不是模型级评估；
- 它依赖输入数据、ETF 映射、成本设置、Prompt 与风控约束；
- 它更适合回答“当前整条链能否稳定工作”，而不是回答“纯模型理论最优是多少”。

---

## 15. 防止 Look-Ahead Bias 的标准规则

项目中的 look-ahead bias 修复原则应被视为本系统的硬约束，核心可以归纳为一句话：

> 决策时刻只能看到在该时刻之前真实可获得的信息。

### 15.1 通用规则

- 训练特征不能包含未来收益信息。
- 盘中决策不能使用当日收盘后才知道的字段。
- Agent 不能直接消费训练标签。
- 回测不能用最终训练阶段已经见过的日期做“最终泛化评估”。

### 15.2 Signals 特征规则

重点包括：

- LightGBM 的差分、成交量比率、日内波动等字段必须回退到 `T-1` 可见口径。
- 如果某个特征只能在收盘后确定，则它不能被同一交易日盘中决策使用。
- 新闻时间戳过滤必须遵循“可见性”而不是“自然日全量”。

### 15.3 Agent 特征规则

`AgentFeatureBuilder` 明确要求只构建当前时间之前可得的特征，这条原则必须保持不变。

### 15.4 Backtest 规则

Backtest 的 held-out 日期，不应出现在最终训练窗口内。

---

## 16. 推荐数据切分策略

假设 4 年原始历史覆盖 `2021-01-01` 到 `2024-12-31`，当前标准推荐两种切分方式。

### 16.1 开发验证切分：`2 + 1 + 1`

用于调试模型与流水线：

- 2021-01-01 ~ 2022-12-31：训练
- 2023-01-01 ~ 2023-12-31：验证 / OOF
- 2024-01-01 ~ 2024-12-31：完全保留的 holdout

对应命令：

```bash
just signals-train-dev-2y1y
just signals-export-onnx-dev-2y1y
```

### 16.2 最终评估切分：`3 + 1`

用于确定最终 Agent / backtest 评估：

- 2021-01-01 ~ 2023-12-31：最终训练
- 2024-01-01 ~ 2024-12-31：纯推理 + Agent + backtest

对应命令：

```bash
just signals-train-final-3y
just signals-export-onnx-final-3y
just signals-infer-2024
just backtest-2024
```

### 16.3 为什么不能 4 年全训再回测同一段

因为那样会把“模型已经见过的数据”当作“泛化评估数据”，使回测结果失去解释力。

---

## 17. 标准运行命令

以下命令来自 `justfile` 与两个 CLI 入口，是当前推荐 runbook。

### 17.1 环境准备

CPU：

```bash
just cpu-sync
```

GPU：

```bash
just gpu-sync
```

### 17.2 新闻标签推理

```bash
python -m trainer.main predict major
python -m trainer.main predict sub --sub-shard-workers 4 --sub-major-workers 8
python -m trainer.main predict all
```

### 17.3 Signals 训练与导出

```bash
python -m trainer.main signals train
python -m trainer.main signals export-onnx --checkpoint-dir ./trainer/checkpoints/signals/final-3y --bundle-dir ./trainer/models/signals/final-3y
python -m trainer.main signals infer --bundle-dir ./trainer/models/signals/final-3y --output-path ./data/agent_features.oof.parquet --start-date 2024-01-01 --end-date 2024-12-31
```

### 17.4 `just` 快捷命令

```bash
just signals-train-dev-2y1y
just signals-train-final-3y
just signals-export-onnx-dev-2y1y
just signals-export-onnx-final-3y
just signals-infer-2024
just backtest-2024
just signals-agent-pipeline-2024
```

### 17.5 Runtime / Backtest

```bash
python runtime/agent/main.py decide --week 2024-06-03
python runtime/agent/main.py backtest --start-date 2024-01-01 --end-date 2024-12-31
python runtime/agent/main.py diagnose-backtest --run-id bt_example
python runtime/agent/main.py visualize-backtest --run-id bt_example
```

### 17.6 Runtime 产物检查与迁移

```bash
just runtime-check-artifacts
just runtime-migrate-artifacts
```

### 17.7 Docker

```bash
just docker-build-runtime
just docker-backtest 2024-01-01 2024-12-31
just docker-backtest-run bt_demo 2024-01-01 2024-12-31
just docker-backtest-2024
```

---

## 18. 关键产物清单

### 18.1 训练期产物

- `trainer/checkpoints/signals/*/signals_checkpoint.json`
- `trainer/checkpoints/signals/*/tcn_fanin.pt`
- `trainer/checkpoints/signals/*/lgbm/*.txt`
- `trainer/checkpoints/signals/*/iforest_model.pkl`

### 18.2 部署期产物

- `trainer/models/signals/*/manifest.json`
- `trainer/models/signals/*/tcn.onnx`
- `trainer/models/signals/*/lgbm/*.onnx`
- `trainer/models/signals/*/iforest.onnx`

### 18.3 推理期产物

- `data/agent_features.parquet`
- `data/agent_features.oof.parquet`
- `runtime/agent/data/inputs/sentiment_weekly.parquet`

### 18.4 运行期产物

- `runtime/agent/data/backtest_results.parquet`
- `runtime/agent/data/backtest_metrics.parquet`
- `runtime/agent/checkpoints/{run_id}/`
- `runtime/agent/wandb/`

---

## 19. 验证与验收清单

### 19.1 数据层验收

- 月度 parquet 是否连续。
- `raw.parquet` 是否存在或可自动重建。
- `label_stats.json` 与当前 canonical 子行业维度是否一致。
- `meta_sector_mapping.json` 是否仍是唯一映射源。

### 19.2 训练层验收

- `signals train` 是否能无 shape mismatch 完成。
- 是否导出 checkpoint。
- 是否写出 OOF / infer 可用特征。
- walk-forward 指标是否正常，不存在异常虚高迹象。

### 19.3 推理层验收

- `signals infer` 是否只输出指定日期范围。
- `agent_features.oof.parquet` 是否成功生成。
- Runtime 是否优先读取 OOF / infer 特征。
- ONNX 推理与参考值是否保持在可接受误差范围内。

可用工具：

```bash
python scripts/check_signals_onnx_consistency.py --help
```

### 19.4 Agent / Backtest 验收

- `decision_context` 是否包含 model / news / market / position / memory 五大类信息。
- `human_summary` 是否存在。
- `level1_plan` 是否以元板块为主。
- 是否记录 `selected_etfs`。
- 回测结果中是否写出 `meta_sector_returns` 与 `meta_sector_contributions`。

---

## 20. 当前已知历史口径差异与统一结论

这一节非常重要，因为它直接解释了为什么需要这份标准文档。

### 20.1 “46 维”还是“47 维”

历史计划文档中出现过 47 维子行业表述，但当前标准实现以：

- `data/label_stats.json`
- 当前训练代码

为准，canonical 子行业数量是 46。

统一结论：

- 设计文档、模型说明、接口说明统一按 46 维描述。

### 20.2 “5 天 TCN”还是“10 天 TCN”

历史文档中“5 天”更多是在描述 Agent 看的最近趋势窗口；当前训练配置中：

- `signals.tcn.sequence_length = 10`

统一结论：

- TCN 底层训练窗口标准口径是 10；
- Agent 决策层消费的摘要序列通常回看最近 5 个交易日；
- 两者不冲突，但不能混为同一个概念。

### 20.3 元板块命名差异

历史文档里曾有“红利/中特估”“区域经济”“智能网联”等另一套候选命名；当前标准实现以 `data/meta_sector_mapping.json` 为准。

统一结论：

- 当前有效元板块必须使用映射文件中的 8 个名称。

### 20.4 “全历史特征”是否等价于“推理特征”

不是。

统一结论：

- `agent_features.parquet` 主要用于调试；
- `agent_features.oof.parquet` 才是 held-out / runtime / backtest 的优先输入。

---

## 21. 非协商规则

以下规则一旦被破坏，就说明改动在架构层面不成立：

1. 不要在最终评估中使用训练期已经见过的日期。
2. 不要让 Agent 直接消费训练标签或未来派生特征。
3. 不要因为某段年份数据较少就缩减 canonical 输入维度。
4. 不要在代码里维护第二套隐藏元板块映射。
5. 不要把全历史训练特征当作 held-out 推理特征的等价替代物。
6. 不要让风控层替代 Agent 决策逻辑本身，风控只负责保护和约束。

---

## 22. 推荐维护方式

如果未来你修改了以下任一模块：

- `trainer/src/datasets/signals.py`
- `trainer/src/models/signals.py`
- `trainer/src/pipelines/train_signals.py`
- `trainer/src/pipelines/infer_signals.py`
- `runtime/agent/src/agent/features.py`
- `runtime/agent/src/agent/tools.py`
- `runtime/agent/src/backtest/*`
- `data/label_stats.json`
- `data/meta_sector_mapping.json`

建议同时做三件事：

1. 更新本标准说明文档。
2. 运行最小可复现命令验证主链未断。
3. 明确说明这次变更是否改变了训练 / 推理 / Agent / backtest 之间的契约。

---

## 23. 标准结论

当前项目最重要的架构结论可以浓缩成一句话：

> News2ETF-Agent 的标准主链不是“训练一个模型然后直接回测”，而是“训练可部署信号模型 -> 导出固定推理产物 -> 让 Agent 只消费推理期上下文 -> 用周频回测验证整个决策系统”。

如果你只记住一条规则，请记住这一条。
