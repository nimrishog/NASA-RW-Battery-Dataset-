# Event-Window Transformer for Battery SOH Estimation

This repository contains the data, code, model outputs, numerical results, and figures used in the associated manuscript on event-window Transformer state-of-health (SOH) estimation under randomized charge-discharge operation.

## Study scope

- NASA randomized-use cells RW9, RW10, and RW11
- Primary transfer split: RW9 and RW10 for model development; RW11 held out for testing
- Complete leave-one-cell-out (LOCO) Transformer evaluation across RW9-RW11
- Next-reference, previous-reference, charge-count interpolation, and cumulative-Ah interpolation targets
- Transformer window-length, feature-group, local dQ/dV, and reference-block-weighting analyses
- Three one-variable ridge baselines and a controlled vanilla recurrent neural network baseline
- Gradient SHAP global attribution and LIME local explanations
- Paired-seed, cross-cell, dependence, and attribution-redistribution analyses for local dQ/dV

## Repository layout

```text
data/
  raw_mat/                  RW9-RW11 source MATLAB files
  raw_extraction_summary/   extraction validation and reference-capacity summary
  modeling/                 event-feature table used by the models
scripts/
  extract_rw_raw_mat_pipeline.py
  event_data.py
  run_trend_baselines.py
  run_transformer_loco.py
  run_controlled_rnn.py
  run_transformer_ablations.py
  run_transformer_xai.py
  run_stage*.py
  build_manuscript_figures.py
results/
  primary_transformer/      primary RW9/RW10-to-RW11 outputs
  reviewer_validation/      LOCO, baseline, ablation, and dQ/dV results
  manuscript_figures/       figures cited in the manuscript
```

## Environment

Python 3.11 was used for the audited environment. Create a clean environment and install the pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduction order

Run commands from the repository root. The raw extraction step is optional when using the tracked event-feature table.

```bash
# 1. Optional raw MATLAB extraction and quality-control summaries
python scripts/extract_rw_raw_mat_pipeline.py

# 2. Primary Transformer and complete LOCO evaluation
python scripts/run_transformer_loco.py

# 3. Trend-only ridge baselines and controlled recurrent baseline
python scripts/run_trend_baselines.py
python scripts/run_controlled_rnn.py

# 4. Label, window-length, feature-group, dQ/dV, and weighting analyses
python scripts/run_transformer_ablations.py

# 5. Primary Gradient SHAP and LIME analyses
python scripts/run_transformer_xai.py

# 6. Paired-seed and cross-cell dQ/dV investigation
python scripts/run_stage2_paired_dq_dv_seeds.py
python scripts/run_dq_dv_loco_ablation.py
python scripts/run_stage4_global_shap_redistribution.py
python scripts/run_stage6_dq_dv_redundancy.py

# 7. Data-driven manuscript figures
python scripts/build_manuscript_figures.py

# 8. Fast integrity check against reported outputs
python scripts/validate_repository.py
```

The training scripts use fixed seeds, training-only clipping and standardization, SOH-stratified internal validation from development cells, inverse reference-block weighting where specified, weighted Huber loss, and validation-R2 checkpoint selection. A held-out cell is excluded from preprocessing, fitting, hyperparameter selection, and checkpoint selection.

## Primary reported results

For RW9/RW10 development and held-out RW11 testing:

| Evaluation scale | R2 | MAE | RMSE |
|---|---:|---:|---:|
| Event window | 0.9551 | 2.387 | 3.039 |
| Diagnostic checkpoint | 0.9664 | 2.069 | 2.535 |

MAE and RMSE are SOH percentage points. Machine-readable predictions, metrics, training histories, and fitted model states are under `results/`.

## Data availability

The source MATLAB files originate from the NASA Randomized Battery Usage dataset. The repository includes RW9-RW11 source files, extraction summaries, the processed event-feature table, model outputs, and figure source tables to support reproducibility. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for provenance, redistribution, and archival guidance.

## Licensing

The analysis code is released under the MIT License. The NASA source data and any third-party materials retain their original terms and are not relicensed by the code license.
