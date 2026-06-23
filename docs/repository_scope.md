# Repository Scope

This repository includes validated source data, verified Excel exports, scripts, selected model outputs, XAI outputs, and presentation/report assets.

It intentionally excludes the multi-GB full raw CSV extraction files because they can be regenerated from the `.mat` files using:

```text
scripts/extract_rw_raw_mat_pipeline.py
```

Included validation artifacts:

```text
data/validated_excel/VALIDATION_NOTE.md
data/raw_extraction_summary/VALIDATION_REPORT.md
data/raw_extraction_summary/summary.json
data/raw_extraction_summary/reference_discharge_capacity.csv
```

Included final model/report folder:

```text
results/final_charge_discharge_top20_xai/
```

Included high-accuracy history-assisted transformer folder:

```text
results/history_residual_patch_transformer/
```

