# Training Configuration Guide

[English](CONFIGURATION.md) | [简体中文](CONFIGURATION.zh-CN.md)

Configuration filenames use lowercase snake case and follow
`<domain>_<purpose>_<variant>.yaml`.

## Paper Reproduction Configs

These are the only configs intended to reproduce the four-domain main training
protocol. They use a maximum of 12 epochs, early-stopping patience 6, and fixed
alignment weight `align_lambda: 0.25`.

| Config | Core-HopQA domain |
| --- | --- |
| `dblp_main.yaml` | DBLP |
| `biomedical_main.yaml` | Biomedical |
| `goodreads_main.yaml` | GoodReads |
| `pubmed_main.yaml` | PubMed |

Hardware fields such as `bf16`, `use_deepspeed`, worker counts, and gradient
accumulation may be adjusted to match the target machine. Changing the model,
sampling, interface, loss, or schedule fields changes the experimental protocol.

## Diagnostic Configs

- `dblp_intervention_no_graph.yaml`: single-seed no-graph intervention.
- `dblp_intervention_shuffled_graph.yaml`: single-seed shuffled-token
  intervention.

These recipes are analysis tools, not the multi-seed main-result protocol.

## Utility Config

- `dblp_smoke.yaml`: one-epoch dependency, data, and model-wiring smoke test.

A successful smoke run confirms that the local installation is connected
correctly; it does not reproduce a paper score.

## Retained Experimental Configs

The following files are preserved for research extensions and historical
traceability:

- `dblp_full.yaml`
- `dblp_ablation_no_alignment.yaml`
- `dblp_ablation_no_query_syntax.yaml`
- `dblp_ablation_random_order.yaml`
- `dblp_experimental_query_syntax_variant.yaml`
- `dblp_experimental_token_shuffle_variant.yaml`

They use development-time settings such as a shorter schedule or the optional
sub-epoch monitor. They are deliberately marked at the top of each YAML.

Two retained variants deserve special attention:

- `dblp_experimental_query_syntax_variant.yaml` disables the same implemented
  query-syntax switch as `dblp_ablation_no_query_syntax.yaml`; it is retained
  for provenance and is not a no-query-aware-sampling implementation.
- `dblp_experimental_token_shuffle_variant.yaml` enables the same token-shuffle
  switch as `dblp_ablation_random_order.yaml`; it is not a distinct validated
  role-perception ablation.

## Optional Trainer Controls

The CLI retains opt-in research controls for dynamic alignment weighting,
alignment learning-rate compensation, axis-momentum reset, and sub-epoch
monitoring. They default to disabled and are not enabled by any
`*_main.yaml` config. See the labelled experimental blocks in
`src/s2ge/cli/common.py` and `src/s2ge/engine/trainer.py`.
