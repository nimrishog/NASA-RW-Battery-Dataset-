# RW11 analysis

## Core counts
- Total MAT steps: 109,389
- Kept partial-charge segments: 22,759
- Total charge-segment samples: 3,916,601
- Reference discharges: 77

## Charge-segment sequence shape
- Sample count range: 2 to 301
- Median sample count: 214.0
- Duration range (s): 0.00 to 299.99
- Median duration (s): 212.52
- Median sampling interval (s): 1.000

## Reference capacity
- Initial reference capacity (Ah): 2.093988
- Final reference capacity (Ah): 0.949960
- Capacity fade (Ah): 1.144028

## Dominant step classes
- rest (random walk) [R]: 54,244
- discharge (random walk) [D]: 27,176
- charge (random walk) [C]: 27,066
- pulsed load (discharge) [D]: 186
- pulsed load (rest) [R]: 186
- pulsed charge (charge) [C]: 122
- pulsed charge (rest) [R]: 122
- reference charge [C]: 77

## Files
- `all_steps_summary.csv`: one row per original MAT step.
- `charge_segments_summary.csv`: one row per kept partial-charge segment.
- `reference_capacity_summary.csv`: reference-discharge benchmark capacities.
- `01_*.png` to `06_*.png`: decision-oriented figures for model design.