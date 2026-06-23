# Battery SOH Modeling Handoff: RW9, RW10, RW11

Date: 2026-04-29

Working root used in the previous session:

```text
C:\Users\nimri\Downloads\BatteryData
```

Handoff folder:

```text
C:\Users\nimri\Downloads\battery_rw9_rw11_handoff_20260429
```

## 1. Main Objective

The goal was to estimate battery SOH/capacity from NASA Randomized Battery Usage data using transformer-based models.

The main target used in the final experiments was:

```text
SOH (%) = 100 * reference_discharge_capacity / initial_reference_discharge_capacity
```

The SOH label comes from periodic benchmark reference-discharge tests. Charge/discharge random-walk events are paired with the next benchmark reference-discharge SOH checkpoint.

## 2. Data Structure

The raw MATLAB files contain:

```text
data.step
```

Each step can include:

```text
comment
type
relativeTime
time
voltage
current
temperature
date
```

The important step categories are:

```text
charge (random walk)
discharge (random walk)
reference charge
reference discharge
rest
```

Hierarchy used for modeling:

```text
Battery file
  -> MATLAB step
    -> random-walk event / segment
      -> sample row: Voltage, Current, Temperature, relTime
        -> 30-row patch token
          -> transformer input sequence
```

## 3. Copied CSV Data

The handoff folder contains the practical CSVs for RW9, RW10, and RW11.

The multi-GB combined raw files were not copied because the drive had limited free space. These source files remain in:

```text
C:\Users\nimri\Downloads\BatteryData\raw_mat_extraction_exports
```

Copied data:

```text
csv_data\charge_samples_with_temperature
  RW9_charge_temp_datacapa.csv
  RW10_charge_temp_datacapa.csv
  RW11_charge_temp_datacapa.csv

csv_data\charge_engineered_features
  RW9_charge_engineered_features.csv
  RW10_charge_engineered_features.csv
  RW11_charge_engineered_features.csv

csv_data\analysis_summaries\RW9
  all_steps_summary.csv
  charge_segments_summary.csv
  reference_capacity_summary.csv

csv_data\analysis_summaries\RW10
  all_steps_summary.csv
  charge_segments_summary.csv
  reference_capacity_summary.csv

csv_data\analysis_summaries\RW11
  all_steps_summary.csv
  charge_segments_summary.csv
  reference_capacity_summary.csv

csv_data\charge_discharge_segment_metadata_RW9_RW10_RW11.csv
```

Final model outputs copied:

```text
csv_data\transformer_history_residual
  patch_feature_manifest.csv
  patch_feature_scaler.csv
  rolling_patch_feature_pairs.csv
  training_history.csv
  test_predictions.csv
  test_checkpoint_predictions.csv

figures
  training_dashboard.png
  prediction_report.png
```

## 4. Labeling Process

Reference-discharge capacity is computed by integrating current over the controlled reference-discharge step.

Capacity in ampere-hours:

```text
capacity_ah = integral(|I(t)| dt) / 3600
```

SOH:

```text
SOH (%) = 100 * capacity_ah_at_checkpoint / initial_capacity_ah
```

Each random-walk charge/discharge event is assigned the SOH of the next reference-discharge checkpoint in time.

Many events share the same benchmark SOH label. Therefore, benchmark-block weighting was used:

```text
sample_weight = 1 / number_of_events_sharing_that_checkpoint_label
```

This prevents one benchmark block with many events from dominating the loss.

## 5. Approved Signal-Derived Features

The approved feature families were:

```text
raw voltage/current/temperature/relTime
charge throughput
energy throughput
voltage shape
current statistics
thermal response
voltage-window timing
IC / DV features
distributional statistics
rolling patch differences
```

Features explicitly removed from clean models:

```text
time_start
time_end
absolute_elapsed_time
label_charge_count
sample_count
```

Reason: these can encode aging chronology directly and can inflate model performance.

## 6. Transformer Patch Setup

For the patch transformer models:

```text
one random-walk event = one model sample
one 30-row patch = one transformer token
maximum tokens per event = 11
shorter events are padded
no raw rows are dropped inside an event before patching, except max-token truncation if longer than 11 patches
```

The transformer used:

```text
CLS token
linear input projection
learned positional embedding
transformer encoder
MLP output head
weighted MSE loss
```

Main architecture:

```text
d_model = 96
number of heads = 4
number of encoder layers = 3
feedforward dimension = 192
dropout = 0.10 to 0.15
optimizer = AdamW
gradient clipping = 1.0
```

## 7. Experiments and Results

### A. Earlier High-Accuracy Models With Time/History Proxies

These achieved good results, but some used proxy features such as absolute time, event count, or similar aging indicators.

| Model | Split | Key inputs | Test R2 | MAE | RMSE | Note |
|---|---|---|---:|---:|---:|---|
| Charge-only LightGBM | RW9+RW10 -> RW11 | Engineered features including aging proxies | 0.9576 | 2.55 | 2.97 | Strong but not pure transformer |
| Charge+discharge feature-token transformer | RW9+RW10 -> RW11 | All engineered features, included proxy-like features | 0.9331 | 3.20 | 3.71 | Good but relied on proxy features |
| Charge+discharge LightGBM | RW9+RW10 -> RW11 | All engineered features | 0.9284 | 3.29 | 3.84 | Tabular baseline |
| Patch-only transformer, older tuned run | RW9+RW10 -> RW11 | Patch features, older feature setup | 0.9211 | 3.35 | 4.03 | Older run before strict feature cleanup |

### B. Clean Signal-Only Models Without Time/Count/History Proxies

These used voltage/current/temperature/relTime-derived features only.

| Model | Split | Features | Test R2 | MAE | RMSE | Conclusion |
|---|---|---|---:|---:|---:|---|
| RelTime-only feature-token transformer | RW9+RW10 -> RW11 | 59 clean event-level features | -1.171 | 18.60 | 21.16 | Failed cross-battery generalization |
| Clean patch transformer | RW9+RW10 -> RW11 | 50 approved patch features | -1.221 | 18.85 | 21.40 | Failed cross-battery generalization |
| Rolling-difference patch transformer | RW9+RW10 -> RW11 | 50 patch features + 50 dprev features | -1.240 | 19.60 | 21.49 | Rolling differences did not fix generalization |

### C. History-Assisted Transformer Without Absolute Time Labels

This is the best final result.

It uses no `time_start`, `time_end`, `absolute_elapsed_time`, `label_charge_count`, or `sample_count`.

It adds benchmark history:

```text
prev_reference_soh_percent
prev_reference_soh_change_pp
```

The model predicts residual degradation:

```text
predicted_next_soh = previous_reference_soh + transformer_predicted_residual
```

Results:

| Model row | Train | Test | Test R2 | Test MAE | Test RMSE |
|---|---|---|---:|---:|---:|
| Previous-SOH baseline only | RW9+RW10 | RW11 | 0.9540 | 2.25 | 3.08 |
| History-residual patch transformer | RW9+RW10 | RW11 | 0.9717 | 1.85 | 2.42 |
| Checkpoint-aggregated transformer | RW9+RW10 | RW11 | 0.9649 | 1.97 | 2.59 |

Important clarification:

```text
Previous-SOH baseline only is not a transformer.
Checkpoint-aggregated transformer is the same transformer predictions averaged per benchmark checkpoint.
Only the history-residual patch transformer is the actual final transformer model.
```

## 8. Final High-Accuracy Transformer Inputs

The final high-accuracy transformer uses 102 features per patch token:

```text
100 rolling patch features
2 benchmark-history features
```

The 100 rolling patch features are:

```text
50 original patch features
50 dprev_* rolling-difference features
```

Rolling difference definition:

```text
dprev_feature[k] = feature_at_current_patch[k] - feature_at_previous_patch[k-1]
```

The first patch receives zero for all `dprev_*` features.

The two history features are:

```text
prev_reference_soh_percent
prev_reference_soh_change_pp
```

## 9. Main Scientific Interpretation

The clean waveform-only models did not generalize well from RW9+RW10 to RW11.

The high-accuracy model succeeded because previous reference SOH gives the model a degradation anchor. This is not an absolute time label, but it is still a history feature.

This matches the literature pattern:

```text
many high-accuracy SOH papers do not use timestamps,
but they often use previous capacity/SOH,
reference-discharge information,
discharge duration,
voltage-window timing,
or historical degradation sequences.
```

So the final model should be described as:

```text
history-assisted SOH estimation using charge/discharge random-walk signal features
```

It should not be described as:

```text
pure sensor-only SOH estimation from local random-walk waveform alone
```

## 10. Publications Discussed

Key references discussed:

```text
NASA Randomized Battery Usage Dataset
https://data.nasa.gov/dataset/randomized-battery-usage-1-random-walk

Venugopal & Vigneswaran, 2019, Energies
State-of-Health Estimation of Li-ion Batteries in Electric Vehicle Using IndRNN under Variable Load Condition
https://www.mdpi.com/1996-1073/12/22/4338

Park et al., 2022, Applied Sciences
DWT/LSTM-style reference-discharge voltage/temperature SOH estimation
https://www.mdpi.com/2076-3417/12/8/3996

Zhu et al., 2022, Nature Communications
Voltage relaxation features for battery health
https://www.nature.com/articles/s41467-022-29837-w

Partial IC / transfer-learning SOH example
https://www.mdpi.com/2313-0105/10/9/324
```

The closest same-dataset paper is Venugopal 2019. It uses previous/present capacity information and duration/current-load features, which explains why high accuracy can be achieved without explicit timestamp labels.

## 11. Recommended Next Steps

If continuing this work:

1. Decide whether the final goal allows previous benchmark SOH as an input.
2. If yes, continue with the history-residual transformer and improve interpretability.
3. If no, switch to a reference-discharge profile model or voltage-window/IC-DV model, because local random-walk waveform alone did not generalize.
4. Add SHAP/LIME to the final history-residual transformer if explanation plots are needed.
5. Avoid using absolute time, start time, end time, event count, and sample count unless the presentation explicitly labels them as aging/history proxies.

## 12. Final Model Folder in Original Workspace

Original final model folder:

```text
C:\Users\nimri\Downloads\BatteryData\history_residual_patch_transformer
```

Key files:

```text
metrics.json
training_history.csv
test_predictions.csv
test_checkpoint_predictions.csv
patch_feature_manifest.csv
training_dashboard.png
prediction_report.png
model.pt
```

