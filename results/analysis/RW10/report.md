# RW10 analysis

## Core counts
- Total MAT steps: 110,818
- Kept partial-charge segments: 23,159
- Total charge-segment samples: 3,897,632
- Reference discharges: 77

## Charge-segment sequence shape
- Sample count range: 2 to 301
- Median sample count: 189.0
- Duration range (s): 0.00 to 299.99
- Median duration (s): 187.94
- Median sampling interval (s): 1.000

## Reference capacity
- Initial reference capacity (Ah): 2.095040
- Final reference capacity (Ah): 0.842042
- Capacity fade (Ah): 1.252999

## Dominant step classes
- rest (random walk) [R]: 54,966
- discharge (random walk) [D]: 27,557
- charge (random walk) [C]: 27,407
- pulsed load (discharge) [D]: 183
- pulsed load (rest) [R]: 183
- pulsed charge (charge) [C]: 118
- pulsed charge (rest) [R]: 118
- reference charge [C]: 77

## Files
- `all_steps_summary.csv`: one row per original MAT step.
- `charge_segments_summary.csv`: one row per kept partial-charge segment.
- `reference_capacity_summary.csv`: reference-discharge benchmark capacities.
- `01_*.png` to `06_*.png`: decision-oriented figures for model design.