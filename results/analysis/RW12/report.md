# RW12 analysis

## Core counts
- Total MAT steps: 110,013
- Kept partial-charge segments: 21,693
- Total charge-segment samples: 3,890,796
- Reference discharges: 76

## Charge-segment sequence shape
- Sample count range: 2 to 301
- Median sample count: 301.0
- Duration range (s): 0.00 to 299.99
- Median duration (s): 299.96
- Median sampling interval (s): 1.000

## Reference capacity
- Initial reference capacity (Ah): 2.093718
- Final reference capacity (Ah): 1.086958
- Capacity fade (Ah): 1.006759

## Dominant step classes
- rest (random walk) [R]: 54,517
- discharge (random walk) [D]: 27,273
- charge (random walk) [C]: 27,241
- pulsed load (discharge) [D]: 206
- pulsed load (rest) [R]: 206
- pulsed charge (charge) [C]: 146
- pulsed charge (rest) [R]: 146
- reference charge [C]: 76

## Files
- `all_steps_summary.csv`: one row per original MAT step.
- `charge_segments_summary.csv`: one row per kept partial-charge segment.
- `reference_capacity_summary.csv`: reference-discharge benchmark capacities.
- `01_*.png` to `06_*.png`: decision-oriented figures for model design.