# S2GE 核心源码

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <strong>已被 EMNLP 2026 主会录用</strong>
</p>

<p align="center">
  <img src="assets/S2GE-Icon.png" alt="S2GE 项目图标" width="720">
</p>

本发布包包含 S2GE 图—语言实验的可运行源码和训练配置。发布包只包含代码与
配置；数据集、模型权重、checkpoint 和生成结果需要在解压后自行放置到本地。

## 目录结构

- `src/s2ge/`：数据加载、图采样、模型训练和评估的 Python 包。
- `configs/train/`：GRBench 训练、干预和消融配置。
- `scripts/preprocess/`：数据构建工具，包括完整的
  Planetoid/PubMed 到 Core-HopQA 流程。
- `baseline_recipes/`：G-Retriever 和 LLaGA 外部基线的匹配 HopQA 配方。
- `docs/`：作者说明和训练配置指南，均提供英文与简体中文版本。
- `datasets_processed/`：本地处理后数据的建议位置，需要自行创建。
- `models/`：本地 LLM 权重的建议位置，需要自行创建。
- `outputs/`：checkpoint、预测结果和日志的默认输出目录。

## 环境安装

推荐 Python 3.10。正式训练面向 Linux 和带 CUDA 的 PyTorch 环境。Windows
适合数据预处理和 CPU 烟雾测试；原生 DeepSpeed 训练建议使用 Linux 或 WSL。

正式训练前，请先安装与机器上的 PyTorch/CUDA 版本严格匹配的 PyTorch 和 PyG
扩展 wheel，尤其是 `torch-scatter`，再安装其余依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

只进行预处理时，可以使用较小的依赖集合：

```bash
pip install -r requirements-preprocess.txt
pip install -e .
```

也可以从仓库根目录设置包路径：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

离线运行时，将模型权重放在 `models/` 下，或令 `MODEL_ROOT` 指向包含 Hugging
Face 模型文件的目录。默认模型键 `llama3_8b_instruct` 对应
`meta-llama/Meta-Llama-3-8B-Instruct`；也可以通过 `--llm-model-path` 指定。

## 数据集与模型下载

以下 Hugging Face 页面均已在发布时核验可以访问。S2GE 领域名与原始 GRBench
命名的对应关系为：DBLP 对应 `computer_science`，Biomedical 对应
`healthcare`，GoodReads 对应 `literature`。

| S2GE 领域 | Hugging Face 来源 | 需要获取的内容 |
| --- | --- | --- |
| DBLP | [GRBench QA](https://huggingface.co/datasets/yuyangbai/GRBench-copy) 和 [DBLP 图分片](https://huggingface.co/datasets/YF0808/GRBench_dblp_shards) | `computer_science` QA 记录以及 DBLP `graph.json`/manifest |
| Biomedical | [GRBench QA](https://huggingface.co/datasets/yuyangbai/GRBench-copy) 和 [Biomedical 图分片](https://huggingface.co/datasets/YF0808/GRBench_biomedical_shards) | `healthcare` QA 记录以及 Biomedical `graph.json`/manifest |
| GoodReads | [GRBench QA](https://huggingface.co/datasets/yuyangbai/GRBench-copy) 和 [GoodReads 图分片](https://huggingface.co/datasets/YF0808/GRBench_goodreads_shards) | `literature` QA 记录以及 GoodReads `graph.json`/manifest |
| PubMed | [Hugging Face 上的 NCBI PubMed](https://huggingface.co/datasets/ncbi/pubmed) | 此处可获取 PubMed 记录，但论文使用 Planetoid/PyG PubMed 引文图；必须使用下文脚本重建精确图结构，不能直接用该文本语料替代。 |

图分片仓库体积较大，请使用 Git LFS 或 `hf download`，不要通过浏览器直接下载：

```bash
python -m pip install -U huggingface_hub
hf download yuyangbai/GRBench-copy --repo-type dataset --local-dir downloads/GRBench-QA
hf download YF0808/GRBench_dblp_shards graph.json manifest.json \
  --repo-type dataset --local-dir downloads/GRBench-DBLP
```

S2GE 使用两个 Hugging Face 模型仓库：

- 解码器：[meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。
  该仓库受许可控制，需要先在 Hugging Face 页面接受 Meta 的协议。
- 文本编码器：[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)。

获得 LLaMA 权限后下载：

```bash
hf auth login
hf download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir models/Meta-Llama-3-8B-Instruct
```

加载器会搜索 `models/llama3_8b_instruct` 和
`models/Meta-Llama-3-8B-Instruct`，也可以通过 `--llm-model-path` 提供明确路径。

## 数据目录

获取 GRBench 图和 QA 文件后，将其处理成加载器需要的结构：

```text
datasets_processed/
  GRBENCH/
    processed_data/
      dblp/
        data.json
        graph.pt
        nodes.csv
        edges.csv
      biomedical/
      goodreads/
      pubmed/
```

HopQA 变体使用相同结构，但目录名改为完整领域名，例如
`dblp_hopqa_train2000_val200_test1000`。

也可以将代码指向外部数据目录：

```bash
export DATA_ROOT=/path/to/datasets_processed
export GRBENCH_ROOT=/path/to/datasets_processed/GRBENCH
```

## 数据预处理

将原始 GRBench 文件放入本地目录后运行：

```bash
python -m s2ge.cli.preprocess \
  --dataset grbench \
  --grbench-root /path/to/GRBENCH \
  --domain dblp \
  --grbench-feature-mode light
```

只验证已经处理的数据：

```bash
python -m s2ge.cli.preprocess \
  --dataset grbench \
  --grbench-root /path/to/GRBENCH \
  --domain dblp \
  --validate-only
```

### 重建 PubMed Core-HopQA

发布版包含重建 PubMed 领域的全部代码。当 `--pyg-root` 中没有缓存时，第一个
命令会通过 PyG 下载 Planetoid/PubMed：

```bash
python scripts/preprocess/build_grbench_pubmed_domain.py \
  --pyg-root datasets_raw/pyg_pubmed \
  --grbench-root GRBENCH

python -m s2ge.cli.preprocess \
  --dataset grbench \
  --grbench-root GRBENCH \
  --domain pubmed \
  --grbench-feature-mode text \
  --grbench-embed-dim 384 \
  --grbench-embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --grbench-cache-embeddings \
  --graph-only

python scripts/preprocess/generate_grbench_generic_hop_qa.py \
  --grbench-root GRBENCH \
  --source-domain pubmed \
  --target-domain pubmed_hopqa_train2000_val200_test1000 \
  --num-samples 3200 \
  --min-hop 1 \
  --max-hop 5 \
  --max-attempts 300000 \
  --seed 0

python scripts/preprocess/write_grbench_explicit_split.py \
  --grbench-root GRBENCH \
  --domain pubmed_hopqa_train2000_val200_test1000 \
  --train-count 2000 \
  --val-count 200 \
  --test-count 1000 \
  --seed 0

python -m s2ge.cli.preprocess \
  --dataset grbench \
  --grbench-root GRBENCH \
  --domain pubmed_hopqa_train2000_val200_test1000 \
  --validate-only
```

只检查依赖时，可在第二个命令中使用 `--grbench-feature-mode light`。由于 QA 记录
由后续命令生成，需要保留 `--graph-only`。论文配置使用 384 维文本嵌入。

## 训练

`src/s2ge/engine/trainer.py` 是训练编排层，负责数据加载器、模型、优化器和可选
DeepSpeed runtime 的构建，以及梯度累积、验证、checkpoint 选择、早停、最终测试
和断点恢复。实验性控制项保留用于研究扩展，但均有明确标记，并在论文主配置中
关闭。

快速烟雾测试：

```bash
python -m s2ge.cli.train \
  --config configs/train/dblp_smoke.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/smoke \
  --no-use-deepspeed
```

DBLP 主实验：

```bash
deepspeed --module s2ge.cli.train \
  --config configs/train/dblp_main.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/main
```

其他论文领域使用相同命令结构，并选择对应配置：

- `configs/train/biomedical_main.yaml`
- `configs/train/goodreads_main.yaml`
- `configs/train/pubmed_main.yaml`

PubMed 示例：

```bash
deepspeed --module s2ge.cli.train \
  --config configs/train/pubmed_main.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/pubmed_main
```

论文配置启用 DeepSpeed，并使用内置 stage-0 配置以保持发布版优化器和梯度累积
语义。等价的本地 PyTorch 路径是把 `deepspeed --module` 替换为 `python -m`，
并追加 `--no-use-deepspeed`。调整 `batch_size`、`grad_steps`、`bf16` 或
`max_txt_len` 可能改变有效优化 batch 或输入预算。

### Camera-ready 方法与代码对应关系

- **Query-aware sampling**：启用 HSGS，在邻域扩展前将查询端点放到 seed 选择前部。
- **Role-based perception**：query-syntax 排序暴露源点、目标点、最短路走廊和邻近
  角色；主配置在投影到解码器空间后加入角色与距离残差。
- **Adjacency-based alignment**：将投影后节点 token 的余弦相似度与归一化采样
  子图邻接关系匹配，固定 `align_lambda: 0.25`。

四个主配置都使用 LLaMA-3-8B-Instruct、bf16、关闭 gradient checkpointing、
最多 12 epochs、patience 6 和 node-token 模式。MoCo、dual-view 对比学习、
token corruption、随机 token shuffle 和动态对齐权重均关闭。

## 评估

评估输出目录中的最佳 checkpoint：

```bash
python -m s2ge.cli.eval \
  --config configs/train/dblp_main.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/main
```

评估指定 checkpoint：

```bash
python -m s2ge.cli.eval \
  --config configs/train/dblp_main.yaml \
  --checkpoint-path /path/to/checkpoint.pt \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT"
```

## 配置指南

四个 `*_main.yaml` 是论文复现配置。诊断、烟雾、旧版和实验性配置均保留，但已
明确标记。选择配置前请阅读[训练配置指南](docs/CONFIGURATION.zh-CN.md)。

配置文件名遵循 `<domain>_<purpose>_<variant>.yaml`，统一使用小写 snake case。
对齐权重等实现细节保留在 YAML 内，不编码到文件名中。

常用覆盖项：

```bash
--grbench-domain pubmed
--seed 1
--offline
--llm-model-path /path/to/local/model
--output-root outputs/pubmed_seed1
```

## 基线配方

`baseline_recipes/` 包含用于公平性核验的外部基线配方，但不内置第三方仓库。
请将仓库放在 `baselines/G-Retriever-main` 和 `baselines/LLaGA-master`，准备相同的
`*_hopqa_train2000_val200_test1000` 划分，并使用发布的 YAML 设置。

## 测试

先安装面向 CPU 的测试依赖：

```bash
python -m pip install -r requirements-test.txt
```

`torch-scatter` 必须与 PyTorch 和 CUDA 构建匹配。以 PyTorch 2.10 CPU 为例：

```bash
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
```

运行测试：

```bash
python -m pytest -q tests
```

测试会直接覆盖 `gensim`、编译后的 `torch-scatter` 扩展、GRBench 预处理和
VPA-AEGF 单层前向路径。

## 说明

源码包不包含数据集或模型权重。除非专门制作独立发布包，否则请将大型文件、
checkpoint 和生成输出保留在版本控制之外。

## 文档

- [作者说明](docs/AUTHORS_NOTES.zh-CN.md) | [Authors' notes (English)](docs/AUTHORS_NOTES.md)
- [训练配置指南](docs/CONFIGURATION.zh-CN.md) | [Training configuration guide (English)](docs/CONFIGURATION.md)
- [基线配方指南](baseline_recipes/README.zh-CN.md) | [Baseline recipe guide (English)](baseline_recipes/README.md)

如需了解 S2GE 背后的研究动机、接口视角和未来方向，请阅读[作者说明](docs/AUTHORS_NOTES.zh-CN.md)。

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。

Copyright 2026 The S2GE Authors.
