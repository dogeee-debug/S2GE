# HopQA Baseline Recipes

[English](README.md) | [简体中文](README.zh-CN.md)

This directory records the matched HopQA adaptation recipes used for the
external G-Retriever and LLaGA baselines. The S2GE core release does not vendor
third-party baseline repositories, model weights, checkpoints, or generated
outputs. These files expose the training/evaluation settings needed to audit
baseline fairness.

## Fairness Scope

The main comparison controls the evaluation setting:

- Same Core-HopQA domains: `dblp`, `biomedical`, `goodreads`, and `pubmed`
  variants with `train2000_val200_test1000`.
- Same task filter: shortest-hop / hop-count labels.
- Same seeds where available: `0`, `1`, and `12138`.
- Same native-generation strict evaluation and diagnostic first-integer audit.
- Same order-of-magnitude adaptation budget: 12 training epochs for S2GE,
  G-Retriever, and LLaGA recipes.

The recipes do not make the baseline interfaces identical to S2GE. G-Retriever
and LLaGA keep their own graph-language interfaces; S2GE's structured encoding
and alignment losses remain the method difference under test.

## Files

- `gretriever_hopqa.yaml`: G-Retriever HopQA adaptation hyperparameters and
  domain-specific safety overrides.
- `llaga_hopqa.yaml`: LLaGA HopQA adaptation hyperparameters and checkpoint
  evaluation policy.
- `main_table_runs.yaml`: domain/seed matrix and result-artifact naming used
  for the main-table baseline coverage.

## External Repositories

To reproduce these baselines, place the third-party repositories next to the
S2GE project:

```text
baselines/
  G-Retriever-main/
  LLaGA-master/
```

Prepare the HopQA data into each baseline's expected layout using the same
`*_hopqa_train2000_val200_test1000` processed splits used by S2GE. Then apply
the hyperparameters in the YAML files here when invoking the corresponding
baseline training scripts.
