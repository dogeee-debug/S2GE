# HopQA 基线配方

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a>
</p>

本目录记录外部 G-Retriever 和 LLaGA 基线使用的匹配 HopQA 适配配方。S2GE
核心发布版不内置第三方基线仓库、模型权重、checkpoint 或生成输出。这些文件
公开训练与评估设置，以便核验基线比较的公平性。

## 公平性范围

主比较控制以下评估条件：

- 使用相同的 Core-HopQA 领域：`dblp`、`biomedical`、`goodreads` 和 `pubmed`，
  均使用 `train2000_val200_test1000` 变体。
- 使用相同任务筛选：最短跳数 / hop-count 标签。
- 在可用时使用相同 seeds：`0`、`1` 和 `12138`。
- 使用相同的原生生成 StrictEM 评估和首个整数诊断审计。
- 使用同一数量级的适配预算：S2GE、G-Retriever 和 LLaGA 配方均训练 12 epochs。

这些配方不会将基线接口改造成与 S2GE 完全相同。G-Retriever 和 LLaGA 保留各自
的图—语言接口；S2GE 的结构化编码和对齐损失仍是被检验的方法差异。

## 文件

- `gretriever_hopqa.yaml`：G-Retriever HopQA 适配超参数和领域特定安全覆盖项。
- `llaga_hopqa.yaml`：LLaGA HopQA 适配超参数和 checkpoint 评估策略。
- `main_table_runs.yaml`：主表基线覆盖所用的领域/seed 矩阵和结果文件命名。

## 外部仓库

复现这些基线时，将第三方仓库放在 S2GE 项目旁：

```text
baselines/
  G-Retriever-main/
  LLaGA-master/
```

使用与 S2GE 相同的 `*_hopqa_train2000_val200_test1000` 处理后划分，并将 HopQA
数据整理为各基线要求的目录结构。随后，在对应基线训练脚本中应用本目录 YAML
文件给出的超参数。
