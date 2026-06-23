# NASA RW Battery Dataset SOH Modeling

This repository contains the validated NASA randomized-walk battery data workflow used for battery state-of-health (SOH) modeling.

The work covers:

- source `.mat` battery files
- verified Excel exports from the `.mat` files
- MATLAB extraction and validation scripts
- preprocessing and feature-engineering scripts
- transformer and baseline model scripts
- tested model outputs
- XAI outputs using permutation importance, SHAP-style attribution, and LIME
- final presentation/report assets

## Data

Validated battery files:

```text
data/raw_mat/RW9.mat
data/raw_mat/RW10.mat
data/raw_mat/RW11.mat
data/raw_mat/RW12.mat
```

Verified Excel exports:

```text
data/validated_excel/RW9.xlsx
data/validated_excel/RW10.xlsx
data/validated_excel/RW11.xlsx
data/validated_excel/RW12.xlsx
```

The Excel workbooks were validated against the original `.mat` files using SHA-256 hashes for metadata, step index, time, relativeTime, voltage, current, and temperature. See:

```text
data/validated_excel/VALIDATION_NOTE.md
```

## Main Modeling Split

The final charge+discharge top-20 transformer workflow used:

```text
Train/validation source: RW9 + RW10
Validation: 10% stratified windows from RW9 + RW10
Held-out test: RW11
RW12: unused in that final model
```

## Method Summary

1. Extract source `.mat` files into validated per-battery Excel workbooks and raw CSV summaries.
2. Compute reference-discharge capacity labels from controlled reference-discharge steps.
3. Convert capacity to SOH:

```text
SOH (%) = 100 * reference_discharge_capacity / initial_reference_discharge_capacity
```

4. Build charge/discharge random-walk event-level features from voltage, current, relative time, and event direction.
5. Train transformer models on chronological windows of random-walk events.
6. Evaluate on held-out RW11.
7. Interpret model behavior with global and local XAI outputs.

## Key Results

Final charge+discharge top-20 transformer, RW11 held-out test:

```text
Event-level R2: 0.9551
Event-level MAE: 2.387% SOH
Checkpoint R2: 0.9664
Checkpoint MAE: 2.069% SOH
```

History-residual patch transformer, RW11 held-out test:

```text
Event-level R2: 0.9717
Event-level MAE: 1.847% SOH
Checkpoint R2: 0.9649
Checkpoint MAE: 1.969% SOH
```

## Repository Layout

```text
data/
  raw_mat/
  validated_excel/
  raw_extraction_summary/
scripts/
docs/
results/
  analysis/
  final_charge_discharge_top20_xai/
  history_residual_patch_transformer/
  charge_discharge_model_comparisons/
  award/
```

## Large Files

Large data/model/report artifacts are tracked with Git LFS. The multi-GB raw extracted CSV files are not committed because they are reproducible from the `.mat` files using:

```text
scripts/extract_rw_raw_mat_pipeline.py
```

The extraction summary and validation reports are included under:

```text
data/raw_extraction_summary/
```

