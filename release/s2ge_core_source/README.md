# S2GE Core Source

This package contains the runnable source code and training configurations for
S2GE graph-language experiments. It intentionally ships only code and config
files. Datasets, model weights, checkpoints, and generated outputs should be
placed locally after unpacking.

## Directory Layout

- `src/s2ge/`: Python package for data loading, graph sampling, model training, and evaluation.
- `configs/train/`: ready-to-edit GRBench training, intervention, and ablation configs.
- `datasets_processed/`: expected location for processed datasets; create this directory locally.
- `models/`: expected location for local LLM weights; create this directory locally.
- `outputs/`: default output directory for checkpoints, predictions, and logs.

## Environment

Use Python 3.10 or newer. Install PyTorch, PyG, Transformers, DeepSpeed, and the
other packages required by your CUDA environment.

Install the Python package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

Set the package path from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

For offline runs, place model weights under `models/` or set `MODEL_ROOT` to a
directory containing the downloaded Hugging Face model files. The default model
key is `llama3_8b_instruct`, which resolves to
`Meta-Llama-3-8B-Instruct` unless `--llm_model_path` is supplied.

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

## Train

Quick smoke run:

```bash
python -m s2ge.cli.train \
  --config configs/train/grbench.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/smoke
```

Main HopQA configuration:

```bash
python -m s2ge.cli.train \
  --config configs/train/grbench_main_align_fixed025.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/main
```

The configs are hardware-neutral. Adjust `batch_size`, `grad_steps`, `bf16`,
`use_deepspeed`, and `max_txt_len` for your GPU memory budget.

## Evaluate

Evaluate the best checkpoint saved under an output directory:

```bash
python -m s2ge.cli.eval \
  --config configs/train/grbench_main_align_fixed025.yaml \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT" \
  --output-root outputs/main
```

Evaluate a specific checkpoint:

```bash
python -m s2ge.cli.eval \
  --config configs/train/grbench_main_align_fixed025.yaml \
  --checkpoint-path /path/to/checkpoint.pt \
  --grbench-root "$GRBENCH_ROOT" \
  --model-root "$MODEL_ROOT"
```

## Config Guide

- `grbench.yaml`: minimal smoke-test configuration.
- `grbench_full.yaml`: longer full-training baseline.
- `grbench_main_align_fixed025.yaml`: main S2GE HopQA setting with fixed alignment weight.
- `grbench_intervention_no_graph.yaml`: no-graph intervention.
- `grbench_intervention_shuffled_graph.yaml`: shuffled-graph intervention.
- `grbench_ablation_*.yaml`: component ablations used for diagnostic comparisons.

Common overrides:

```bash
--grbench-domain pubmed
--seed 1
--offline
--llm-model-path /path/to/local/model
--output-root outputs/pubmed_seed1
```

## Notes

This source package does not include datasets or model weights. Keep large
files, checkpoints, and generated outputs outside version control unless you are
creating a separate release bundle for them.
