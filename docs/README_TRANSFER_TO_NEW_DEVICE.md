# Battery Project Transfer Package

Transfer folder:

```text
C:\Users\nimri\Downloads\battery_project_transfer_20260429
```

Start here in the new session:

```text
README\README_battery_soh_handoff.md
```

This package collects the Markdown summary, Python scripts, data exports, model outputs, plots, and raw MATLAB files needed to continue the battery SOH work on another device.

## Important Transfer Note

Large files in this folder were collected using NTFS hardlinks to avoid duplicating many GB on the current C drive.

For normal use, this is fine:

```text
copy the entire battery_project_transfer_20260429 folder to the new device or external drive
```

When copied to another device, the files will be copied as normal files.

Approximate logical package size:

```text
about 12 GB
```

Make sure the new device or external drive has at least 15 GB free.

## Folder Contents

```text
README
```

Contains the project handoff Markdown and file inventory.

```text
code
```

Contains the Python scripts used for extraction, feature engineering, transformer training, XAI, plotting, and PowerPoint generation.

```text
data\raw_mat
```

Contains the original raw MATLAB files:

```text
RW9.mat
RW10.mat
RW11.mat
RW12.mat
```

```text
data\raw_mat_extraction_exports
```

Contains the full clean CSV exports from the raw MATLAB files, including:

```text
step_metadata.csv
long_samples.csv
charge_random_walk.csv
discharge_random_walk.csv
reference_charge.csv
reference_discharge.csv
reference_discharge_capacity.csv
rests.csv
step_chronology_map.csv
qc_issues.csv
```

```text
data\handoff_csv_data
```

Contains the practical RW9/RW10/RW11 CSVs used for handoff:

```text
charge samples with temperature
charge engineered features
analysis summaries
charge/discharge segment metadata
final transformer CSV outputs
```

```text
data\charge_discharge_segments_pkl
```

Contains extracted charge/discharge random-walk segment PKLs used by the transformer scripts.

```text
data\model_outputs
```

Contains key model-result folders:

```text
history_residual_patch_transformer
charge_discharge_rebuilt_same_features
charge_discharge_rebuilt_same_features_rolling_diff_fast6
charge_discharge_reltime_only_transformer_xai
charge_discharge_reltime_patch_transformer_xai
```

```text
figures
```

Contains the final training and prediction plots plus hierarchy diagrams.

```text
presentations
```

Contains PowerPoint summary files generated during the project.

## Final Best Model

The best model so far is:

```text
history_residual_patch_transformer
```

Split:

```text
Train: RW9 + RW10
Test: RW11
```

Inputs:

```text
100 rolling patch features
2 benchmark-history features:
  prev_reference_soh_percent
  prev_reference_soh_change_pp
```

Result:

```text
Event-level RW11 test R2 = 0.9717
Event-level RW11 test MAE = 1.85
Event-level RW11 test RMSE = 2.42
Checkpoint-aggregated RW11 test R2 = 0.9649
```

Important interpretation:

```text
This is a history-assisted model, not a pure sensor-only model.
It uses previous reference SOH as a degradation anchor.
```

## Files to Open First

1. Open:

```text
README\README_battery_soh_handoff.md
```

2. Then inspect:

```text
data\model_outputs\history_residual_patch_transformer\metrics.json
data\model_outputs\history_residual_patch_transformer\patch_feature_manifest.csv
data\model_outputs\history_residual_patch_transformer\training_history.csv
data\model_outputs\history_residual_patch_transformer\test_checkpoint_predictions.csv
```

3. Main script for the best model:

```text
code\run_history_residual_patch_transformer.py
```

