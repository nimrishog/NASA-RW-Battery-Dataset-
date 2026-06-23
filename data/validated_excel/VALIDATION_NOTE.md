# Extracted RW Data Validation

Date: 2026-06-23

Source Excel folder:

```text
C:\Users\nimri\Downloads\BatteryData\excel files version
```

Copied folder:

```text
C:\Users\nimri\Downloads\Extracted RW Data
```

## Result

Overall result: PASS

The copied Excel files were validated against the original MATLAB source files:

| Excel file | Source MAT file | Result |
|---|---|---|
| `RW9.xlsx` | `C:\Users\nimri\Downloads\BatteryData\RW9.mat` | PASS |
| `RW10.xlsx` | `C:\Users\nimri\Downloads\BatteryData\RW10.mat` | PASS |
| `RW11.xlsx` | `C:\Users\nimri\Downloads\BatteryData\RW11.mat` | PASS |
| `RW12.xlsx` | `C:\Users\nimri\Downloads\BatteryData\RW12.mat` | PASS |

## Validation Method

For each `.mat` file, SHA-256 hashes were recomputed from the MATLAB source data for:

- `metadata`
- `step_index`
- `time`
- `relativeTime`
- `voltage`
- `current`
- `temperature`

Those recomputed `.mat` hashes matched the hashes stored in each workbook's `validation` sheet.

## Matched Counts

| Battery | MATLAB steps | Sample values from time vectors |
|---|---:|---:|
| RW9 | 113,578 | 8,532,073 |
| RW10 | 110,818 | 8,596,025 |
| RW11 | 109,389 | 8,664,510 |
| RW12 | 110,013 | 8,638,882 |

No engineered feature files were copied.
