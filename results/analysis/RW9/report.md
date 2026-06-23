# RW9 analysis

## Core counts
- Total MAT steps: 113,578
- Kept partial-charge segments: 23,416
- Total charge-segment samples: 3,864,437
- Reference discharges: 80

## Charge-segment sequence shape
- Sample count range: 2 to 301
- Median sample count: 165.0
- Duration range (s): 0.00 to 299.99
- Median duration (s): 163.70
- Median sampling interval (s): 1.000

## Reference capacity
- Initial reference capacity (Ah): 2.097739
- Final reference capacity (Ah): 0.750185
- Capacity fade (Ah): 1.347554

## Dominant step classes
- rest (random walk) [R]: 56,342
- discharge (random walk) [D]: 28,221
- charge (random walk) [C]: 28,119
- pulsed load (discharge) [D]: 184
- pulsed load (rest) [R]: 184
- pulsed charge (charge) [C]: 114
- pulsed charge (rest) [R]: 114
- reference charge [C]: 80

## Files
- `all_steps_summary.csv`: one row per original MAT step.
- `charge_segments_summary.csv`: one row per kept partial-charge segment.
- `reference_capacity_summary.csv`: reference-discharge benchmark capacities.
- `01_*.png` to `06_*.png`: decision-oriented figures for model design.