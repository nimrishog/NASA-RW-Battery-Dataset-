# Event-Window Transformer for Battery SOH Estimation

This repository contains only the data, code, figures, and numerical results used in the associated manuscript on event-window Transformer state-of-health (SOH) estimation under randomized charge-discharge operation.

## Study scope

- NASA randomized-use cells: RW9, RW10, and RW11
- Primary transfer split: RW9 and RW10 for model development; RW11 held out for testing
- Complete leave-one-cell-out Transformer evaluation across RW9-RW11
- Sparse-label comparisons: next reference, previous reference, charge-count interpolation, and cumulative-Ah-throughput interpolation
- Transformer window-length, feature-group, local dQ/dV, and block-weighting ablations
- Controlled vanilla recurrent neural network baseline
- Gradient SHAP global attribution and LIME local explanations

## Repository layout

```text
data/
  raw_mat/                 RW9-RW11 source MATLAB files
  raw_extraction_summary/  extraction validation and reference-capacity summary
  modeling/                RW9-RW11 event-feature table used by the models
scripts/
  extract_rw_raw_mat_pipeline.py
  event_data.py
  run_transformer_loco.py
  run_controlled_rnn.py
  run_transformer_ablations.py
  run_transformer_xai.py
results/
  primary_transformer/     primary RW9/RW10-to-RW11 outputs
  reviewer_validation/     LOCO, recurrent baseline, and ablation metrics
  manuscript_figures/      figures cited in the manuscript
```

## Modeling data

`data/modeling/event_features_rw9_rw11.csv` contains one row per extracted random-walk charge or discharge event. The model inputs are event descriptors derived from voltage, current, relative time, duration, throughput, energy, power, event direction, and local charge-voltage behavior. Capacity and SOH are retained only as diagnostic targets and are not model inputs.

The primary target assigns each operating event to the next available reference-discharge SOH measurement. Alternative target rules are constructed in `scripts/event_data.py` and evaluated by separate Transformer retraining.

## Reproducing the reported analyses

Create an environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the main analyses from the repository root:

```bash
python scripts/run_transformer_loco.py
python scripts/run_controlled_rnn.py
python scripts/run_transformer_ablations.py
python scripts/run_transformer_xai.py
```

The Transformer scripts use fixed seed 42, training-only feature clipping and standardization, SOH-stratified internal validation from the development cells, inverse reference-block weighting where specified, weighted Huber loss, and validation-R2 checkpoint selection. The held-out cell is excluded from preprocessing, fitting, and checkpoint selection.

## Primary reported results

For RW9/RW10 development and held-out RW11 testing:

| Evaluation scale | R2 | MAE | RMSE |
|---|---:|---:|---:|
| Event window | 0.9551 | 2.387 | 3.039 |
| Diagnostic checkpoint | 0.9664 | 2.069 | 2.535 |

MAE and RMSE are SOH percentage points. Complete machine-readable results are under `results/`.

## Data source

The raw files are from the NASA Randomized Battery Usage dataset. Users should cite the NASA dataset and the associated manuscript when reusing this workflow.
