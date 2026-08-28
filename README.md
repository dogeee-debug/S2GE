# S2GE Core Source

[English](README.md) | [简体中文](README.zh-CN.md)

<p align="center">
  <img src="assets/S2GE-Icon.png" alt="S2GE project icon" width="720">
</p>

This package contains the runnable source code and training configurations for
S2GE graph-language experiments. It intentionally ships only code and config
files. Datasets, model weights, checkpoints, and generated outputs should be
placed locally after unpacking.

## Directory Layout

- `src/s2ge/`: Python package for data loading, graph sampling, model training, and evaluation.
- `configs/train/`: ready-to-edit GRBench training, intervention, and ablation configs.
- `scripts/preprocess/`: dataset-construction utilities, including the complete
  Planetoid/PubMed-to-Core-HopQA pipeline.
- `baseline_recipes/`: matched HopQA adaptation recipes for external G-Retriever
  and LLaGA baselines.
- `docs/`: author notes and the training-configuration guide, each available in
  English and Simplified Chinese.
- `datasets_processed/`: expected location for processed datasets; create this directory locally.
- `models/`: expected location for local LLM weights; create this directory locally.
- `outputs/`: default output directory for checkpoints, predictions, and logs.

## Environment

Python 3.10 is recommended. The production training path targets Linux with a
CUDA-enabled PyTorch installation; Windows is suitable for preprocessing and
CPU smoke tests, but native DeepSpeed training should use Linux or WSL.

For production training, first install PyTorch and the PyG extension wheels
(especially `torch-scatter`) matching the exact CUDA and PyTorch versions on the
machine. Then install the remaining release requirements. This ordering avoids
accidentally building an incompatible `torch-scatter` wheel:

Install the Python package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

For a preprocessing-only environment, the smaller dependency set is enough:

```bash
pip install -r requirements-preprocess.txt
pip install -e .
```

Set the package path from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

For offline runs, place model weights under `models/` or set `MODEL_ROOT` to a
directory containing the downloaded Hugging Face model files. The default model
key is `llama3_8b_instruct`, which resolves to
`meta-llama/Meta-Llama-3-8B-Instruct` unless `--llm-model-path` is supplied.

## Dataset and Model Downloads

The following Hugging Face pages were checked when this release was prepared.
The S2GE domain names map to the original GRBench names as follows: DBLP to
`computer_science`, Biomedical to `healthcare`, and GoodReads to `literature`.

| S2GE domain | Hugging Face source | What to obtain |
| --- | --- | --- |
| DBLP | [GRBench QA](https://huggingface.co/datasets/yuyangbai/GRBench-copy) and [DBLP graph shards](https://huggingface.co/datasets/YF0808/GRBench_dblp_shards) | `computer_science` QA records and the DBLP `graph.json`/manifest |
| Biomedical | [GRBench QA](https://huggingface.co/datasets/yuyangbai/GRBench-copy) and [Biomedical graph shards](https://huggingface.co/datasets/YF0808/GRBench_biomedical_shards) | `healthcare` QA records and the Biomedical `graph.json`/manifest |
| GoodReads | [GRBench QA](https://huggingface.co/datasets/yuyangbai/GRBench-copy) and [GoodReads graph shards](https://huggingface.co/datasets/YF0808/GRBench_goodreads_shards) | `literature` QA records and the GoodReads `graph.json`/manifest |
| PubMed | [NCBI PubMed on Hugging Face](https://huggingface.co/datasets/ncbi/pubmed) | PubMed records are available here, but the paper uses the Planetoid/PyG PubMed citation graph; reconstruct that exact graph with the included script below rather than substituting this text corpus. |

The graph-shard repositories are large. Use Git LFS or `hf download` instead of
downloading files through a browser. For example:

```bash
python -m pip install -U huggingface_hub
hf download yuyangbai/GRBench-copy --repo-type dataset --local-dir downloads/GRBench-QA
hf download YF0808/GRBench_dblp_shards graph.json manifest.json \
  --repo-type dataset --local-dir downloads/GRBench-DBLP
```

S2GE uses two Hugging Face model repositories:

- Decoder: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
  This repository is gated; accept Meta's license on Hugging Face before downloading.
- Text encoder: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Download the decoder after receiving access:

```bash
hf auth login
hf download meta-llama/Meta-Llama-3-8B-Instruct \
  --local-dir models/Meta-Llama-3-8B-Instruct
```

The loader searches both `models/llama3_8b_instruct` and
`models/Meta-Llama-3-8B-Instruct`. You may instead pass an explicit directory
with `--llm-model-path`.

## Data Layout

After obtaining the GRBench source graphs and QA files, preprocess them into the
layout expected by the loader:

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

For HopQA variants, use the same structure with the variant domain name, for
example `dblp_hopqa_train2000_val200_test1000`.

You can also point the code to an external dataset directory:

```bash
export DATA_ROOT=/path/to/datasets_processed
export GRBENCH_ROOT=/path/to/datasets_processed/GRBENCH
```

## Preprocess

Run preprocessing after placing the raw GRBench files in a local directory:

```bash
python -m s2ge.cli.preprocess \
  --dataset grbench \
  --grbench-root /path/to/GRBENCH \
  --domain dblp \
  --grbench-feature-mode light
```

Validate an existing processed dataset without rebuilding it:

```bash
python -m s2ge.cli.preprocess \
  --dataset grbench \
  --grbench-root /path/to/GRBENCH \
  --domain dblp \
  --validate-only
```

### Rebuild the PubMed Core-HopQA domain

The release includes every code step needed to reconstruct the PubMed domain.
The first command downloads Planetoid/PubMed through PyG when it is not already
cached under `--pyg-root`.

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

For a dependency-only smoke test, use `--grbench-feature-mode light` in the
second command. Keep `--graph-only` because QA records are generated by the
following command. The paper configuration uses 384-dimensional text embeddings.

## Train

`src/s2ge/engine/trainer.py` is the orchestration layer. It builds data loaders,
the model, optimizer, and optional DeepSpeed runtime; runs gradient accumulation,
validation, checkpoint selection, early stopping, and final test evaluation;
and supports checkpoint resume. Optional development controls are retained for
research extensions but are labelled in code and disabled by all paper main
configs.

Quick smoke run:

```bash
python -m s2ge.cli.train \
  --config configs/train/dblp_smoke.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/smoke \
  --no-use-deepspeed
```

DBLP main HopQA configuration:

```bash
deepspeed --module s2ge.cli.train \
  --config configs/train/dblp_main.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/main
```

For the other paper domains, select the matching config while keeping the same
command structure:

- `configs/train/biomedical_main.yaml`
- `configs/train/goodreads_main.yaml`
- `configs/train/pubmed_main.yaml`

PubMed example:

```bash
deepspeed --module s2ge.cli.train \
  --config configs/train/pubmed_main.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/pubmed_main
```

The paper configs enable DeepSpeed and use the built-in stage-0 configuration,
which preserves the released optimizer and gradient-accumulation semantics. To
run the equivalent local PyTorch path, replace `deepspeed --module` with
`python -m` and append `--no-use-deepspeed`. You may adjust `batch_size`,
`grad_steps`, `bf16`, and `max_txt_len` for the available GPU memory, but doing
so changes the effective optimization batch or input budget unless compensated.

### Camera-ready method-to-code mapping

- **Query-aware sampling:** HSGS is enabled and the question endpoints are
  forced to the front of seed selection before neighborhood expansion.
- **Role-based perception:** query-syntax ordering exposes source, destination,
  shortest-path-corridor, and proximity roles; the main configs apply the role
  and distance residuals after projection into the decoder space.
- **Adjacency-based alignment:** projected node-token cosine similarity is
  matched to normalized sampled-subgraph adjacency with fixed
  `align_lambda: 0.25`.

All four main configs use LLaMA-3-8B-Instruct, bf16, no gradient checkpointing,
a maximum of 12 epochs, patience 6, and node-token mode. MoCo, dual-view
contrastive learning, token corruption, random token shuffling, and dynamic
alignment weighting are disabled.

## Evaluate

Evaluate the best checkpoint saved under an output directory:

```bash
python -m s2ge.cli.eval \
  --config configs/train/dblp_main.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/main
```

Evaluate a specific checkpoint:

```bash
python -m s2ge.cli.eval \
  --config configs/train/dblp_main.yaml \
  --checkpoint-path /path/to/checkpoint.pt \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT"
```

## Config Guide

The four `*_main.yaml` files are the paper-reproduction configs. Diagnostic,
smoke, legacy, and experimental recipes are retained but explicitly labelled.
See the [training configuration guide](docs/CONFIGURATION.md) before selecting
a configuration.

Training config filenames follow `<domain>_<purpose>_<variant>.yaml`. All names
use lowercase snake case; implementation details such as alignment weights stay
inside the YAML rather than being encoded into the filename.

Common overrides:

```bash
--grbench-domain pubmed
--seed 1
--offline
--llm-model-path /path/to/local/model
--output-root outputs/pubmed_seed1
```

## Baseline Recipes

The release includes external-baseline recipes under `baseline_recipes/` for
fairness auditing. The third-party repositories are not vendored. Place them
under `baselines/G-Retriever-main` and `baselines/LLaGA-master`, prepare the
same `*_hopqa_train2000_val200_test1000` splits, and apply the settings in the
released recipe YAML files.

## Tests

Install the CPU-oriented test dependencies first:

```bash
python -m pip install -r requirements-test.txt
```

`torch-scatter` must match the installed PyTorch and CUDA build. Install its
wheel from the official PyG index; for example, for PyTorch 2.10 CPU:

```bash
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
```

Then run the included release tests:

```bash
python -m pytest -q tests
```

The suite directly exercises `gensim`, the compiled `torch-scatter` extension,
GRBench preprocessing, and the VPA-AEGF single-layer forward path.

## Notes

This source package does not include datasets or model weights. Keep large
files, checkpoints, and generated outputs outside version control unless you are
creating a separate release bundle for them.

## Documentation

- [Authors' notes](docs/AUTHORS_NOTES.md) | [作者说明（中文）](docs/AUTHORS_NOTES.zh-CN.md)
- [Training configuration guide](docs/CONFIGURATION.md) | [训练配置指南（中文）](docs/CONFIGURATION.zh-CN.md)
- [Baseline recipe guide](baseline_recipes/README.md) | [基线配方指南（中文）](baseline_recipes/README.zh-CN.md)

For the motivation, interface perspective, and future directions behind S2GE,
see the [authors' notes](docs/AUTHORS_NOTES.md).
