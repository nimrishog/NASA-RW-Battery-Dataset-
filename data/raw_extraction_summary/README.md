# NASA Randomized Battery Usage Raw MAT Extraction

This folder contains a clean extraction of the raw MATLAB files:

- `RW9.mat`
- `RW10.mat`
- `RW11.mat`
- `RW12.mat`

## Raw MATLAB structure

Each MATLAB file contains a top-level struct called `data`.

The important field is:

- `data.step`

`data.step` is an ordered sequence of step structs. Each step may include fields such as:

- `comment`
- `type`
- `date`
- `relativeTime`
- `time`
- `voltage`
- `current`
- `temperature`

The extraction pipeline preserves that original step ordering and also exports a sample-level long table.

## Exported files

- `step_metadata.csv`
  - One row per original MATLAB step.
  - Includes battery ID, step index, comment, type, date, sample count, duration, start time, end time, and derived step category.

- `long_samples.csv`
  - One row per sample point across all steps.
  - Preserves step structure through `battery_id` and `step_index`.

- `charge_random_walk.csv`
- `discharge_random_walk.csv`
- `reference_charge.csv`
- `reference_discharge.csv`
- `rests.csv`
  - Filtered sample-level exports by step category.

- `reference_discharge_capacity.csv`
  - One row per reference discharge step.
  - `capacity_ah` is computed by integrating current over time and converting to ampere-hours.

- `step_chronology_map.csv`
  - Helper mapping file for later modeling.
  - Includes step ordering and cumulative counts of charge/discharge/reference/rest steps.

- `qc_issues.csv`
  - Data quality checks found during extraction:
    - missing fields
    - inconsistent vector lengths
    - non-monotonic time
    - empty steps
    - invalid comments/types

- `summary.json`
  - Aggregate counts and quality-check totals.

## How to use these exports later for modeling

Use `step_metadata.csv` when you need:

- step-level filtering
- chronology
- mapping between raw step IDs and comments/types

Use `long_samples.csv` when you need:

- the full sample sequence for every step
- reconstruction of original charge/discharge/rest/reference sequences

Use the filtered sample tables when you need:

- only random-walk charge samples
- only random-walk discharge samples
- only benchmark reference steps
- only rest samples

Use `reference_discharge_capacity.csv` when you need:

- benchmark capacity labels from reference discharge steps

Use `step_chronology_map.csv` when you need:

- ordering of charge/discharge/reference steps
- grouping around benchmark steps
- helper counts for later segmentation or labeling logic

## Notes

- The extraction is intentionally separate from model training.
- If a step has inconsistent vector lengths, the exported sample rows are truncated to the shortest available vector length for that step, and the issue is recorded in `qc_issues.csv`.
- Missing numeric fields are exported as blanks in CSV rows.
