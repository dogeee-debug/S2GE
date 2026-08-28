# 训练配置指南

<p align="center">
  <a href="CONFIGURATION.md">English</a> |
  <a href="CONFIGURATION.zh-CN.md">简体中文</a>
</p>

配置文件统一使用小写 snake case，并遵循
`<domain>_<purpose>_<variant>.yaml` 的命名方式。

## 论文复现配置

只有以下四个配置用于复现论文的四领域主实验协议。它们均使用最多 12 个
epoch、early-stopping patience 6，以及固定的对齐损失权重
`align_lambda: 0.25`。

| 配置 | Core-HopQA 领域 |
| --- | --- |
| `dblp_main.yaml` | DBLP |
| `biomedical_main.yaml` | Biomedical |
| `goodreads_main.yaml` | GoodReads |
| `pubmed_main.yaml` | PubMed |

`bf16`、`use_deepspeed`、worker 数量和梯度累积等硬件相关字段可以根据目标
机器调整。修改模型、采样、接口、损失函数或训练日程字段，将改变论文实验协议。

## 诊断配置

- `dblp_intervention_no_graph.yaml`：单 seed 的无图输入干预。
- `dblp_intervention_shuffled_graph.yaml`：单 seed 的图 token 随机打乱干预。

这些配置用于分析，不属于多 seed 主结果协议。

## 工具配置

- `dblp_smoke.yaml`：用于检查依赖、数据和模型连接的一轮烟雾测试。

烟雾测试成功只表示本地安装和代码链路连接正确，不代表复现了论文分数。

## 保留的实验性配置

为了研究扩展和历史追踪，发布版保留以下文件：

- `dblp_full.yaml`
- `dblp_ablation_no_alignment.yaml`
- `dblp_ablation_no_query_syntax.yaml`
- `dblp_ablation_random_order.yaml`
- `dblp_experimental_query_syntax_variant.yaml`
- `dblp_experimental_token_shuffle_variant.yaml`

这些配置包含开发阶段设置，例如较短的训练日程或可选的子 epoch 监控。每个
YAML 顶部均有明确标记。

其中两个保留变体需要特别注意：

- `dblp_experimental_query_syntax_variant.yaml` 关闭的开关与
  `dblp_ablation_no_query_syntax.yaml` 相同；保留它是为了来源追踪，它并不是
  “关闭 query-aware sampling”的实现。
- `dblp_experimental_token_shuffle_variant.yaml` 启用的 token shuffle 开关与
  `dblp_ablation_random_order.yaml` 相同；它不是独立验证过的 role-perception
  消融。

## 可选 Trainer 控制项

CLI 保留了动态对齐权重、对齐学习率补偿、轴动量重置和子 epoch 监控等选择性
研究开关。它们默认关闭，且所有 `*_main.yaml` 配置都没有启用。具体可查看
`src/s2ge/cli/common.py` 和 `src/s2ge/engine/trainer.py` 中标记为 experimental
的代码块。
