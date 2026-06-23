# Extracted RW Data Validation Report

Validation date: 2026-06-23

Source folder:

```text
C:\Users\nimri\Downloads\BatteryData
```

Validated extracted data folder:

```text
C:\Users\nimri\Downloads\BatteryData\raw_mat_extraction_exports
```

## Validation Result

Overall result: PASS

The extracted CSV data was validated against the original MATLAB source files:

```text
RW9.mat
RW10.mat
RW11.mat
RW12.mat
```

## Checks Performed

1. Reloaded each original `.mat` file and recomputed:
   - total step count
   - total sample-row count
   - charge random-walk sample rows
   - discharge random-walk sample rows
   - reference-charge sample rows
   - reference-discharge sample rows
   - rest sample rows
   - reference-discharge step count

2. Recomputed reference-discharge capacity from the raw MATLAB current/time vectors:

```text
capacity_ah = abs(integral(current over time) / 3600)
```

The recomputed 310 reference-discharge capacity values matched `reference_discharge_capacity.csv`.

3. Recounted data rows in every exported CSV file and compared them to `summary.json`.

4. Recomputed QC checks for:
   - missing numeric vectors
   - missing comments
   - invalid step types
   - empty steps
   - inconsistent vector lengths
   - non-monotonic time
   - non-monotonic relative time

No QC issues were found. The exported `qc_issues.csv` correctly contains only the header.

5. Checked current and voltage ranges in the exported charge, discharge, and reference-discharge sample files.

## Matched Counts

| Battery | Steps | Sample rows | Charge RW rows | Discharge RW rows | Ref charge rows | Ref discharge rows | Rest rows | Ref discharge steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RW9 | 113,578 | 8,532,073 | 3,869,140 | 3,737,664 | 148,156 | 37,400 | 541,514 | 80 |
| RW10 | 110,818 | 8,596,025 | 3,901,880 | 3,778,994 | 138,239 | 37,632 | 538,159 | 77 |
| RW11 | 109,389 | 8,664,510 | 3,920,908 | 3,817,817 | 136,335 | 38,101 | 545,595 | 77 |
| RW12 | 110,013 | 8,638,882 | 3,896,344 | 3,742,208 | 124,161 | 44,102 | 595,998 | 76 |
| Total | 443,798 | 34,431,490 | 15,588,272 | 15,076,683 | 546,891 | 157,235 | 2,221,266 | 310 |

## Exported CSV Row Counts

| File | Expected rows | Actual rows | Result |
|---|---:|---:|---|
| `step_metadata.csv` | 443,798 | 443,798 | PASS |
| `step_chronology_map.csv` | 443,798 | 443,798 | PASS |
| `long_samples.csv` | 34,431,490 | 34,431,490 | PASS |
| `charge_random_walk.csv` | 15,588,272 | 15,588,272 | PASS |
| `discharge_random_walk.csv` | 15,076,683 | 15,076,683 | PASS |
| `reference_charge.csv` | 546,891 | 546,891 | PASS |
| `reference_discharge.csv` | 157,235 | 157,235 | PASS |
| `rests.csv` | 2,221,266 | 2,221,266 | PASS |
| `reference_discharge_capacity.csv` | 310 | 310 | PASS |
| `qc_issues.csv` | 0 | 0 | PASS |

## Current and Voltage Sanity Checks

| Export | Rows | Current range (A) | Voltage range (V) | Temperature range (C) |
|---|---:|---:|---:|---:|
| `charge_random_walk.csv` | 15,588,272 | -4.621 to -0.024 | 3.321 to 4.814 | 18.145 to 46.591 |
| `discharge_random_walk.csv` | 15,076,683 | 0.591 to 4.797 | 2.588 to 4.136 | 17.281 to 46.591 |
| `reference_discharge.csv` | 157,235 | 0.831 to 1.120 | 3.200 to 4.117 | 17.532 to 41.559 |

These ranges are consistent with the extraction rules:

- Random-walk charge current is negative.
- Random-walk discharge current is positive.
- Reference discharge current is positive.
- Reference discharge voltage spans roughly full-cell discharge down to 3.2 V.

## Notes

- The validation confirms that the extracted CSV files accurately preserve the source MATLAB step counts, sample counts, step categories, and reference-discharge capacity labels.
- The validation does not claim that the original NASA data is free of experimental noise. It confirms that the extraction from `.mat` to CSV is internally consistent and accurate against the local source files.
