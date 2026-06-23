# Methodology Summary

## Objective

Estimate battery state of health (SOH) from NASA randomized-walk charge and discharge behavior.

## Source Data

The source files are:

```text
RW9.mat
RW10.mat
RW11.mat
RW12.mat
```

Each `.mat` file contains a `data.step` sequence. Each step may include:

- comment
- type
- date
- time
- relativeTime
- voltage
- current
- temperature

Main step categories:

- random-walk charge
- random-walk discharge
- reference charge
- reference discharge
- rest

## Labeling

Reference-discharge capacity is computed from controlled reference-discharge steps:

```text
capacity_ah = abs(integral(current over time) / 3600)
```

SOH is:

```text
SOH (%) = 100 * capacity_ah / initial_capacity_ah
```

Random-walk charge/discharge events are assigned the next available reference-discharge SOH checkpoint.

## Feature Engineering

The final charge+discharge top-20 transformer uses event-level features derived from:

- event direction
- relative time
- current
- voltage

Feature families include:

- event duration
- signed and absolute current
- signed and magnitude current throughput
- signed and magnitude energy throughput
- signed and magnitude power
- starting and ending voltage
- signed voltage change
- voltage distribution statistics
- voltage skewness
- maximum dQ/dV

## Transformer Model

The final top-20 model treats each random-walk event as one token.

```text
Input window: 32 chronological events
Token: one event summarized by 20 features
Embedding: 20 -> 64
Transformer: 2 encoder layers, 4 attention heads
Loss: inverse-checkpoint-block weighted Huber loss
Optimizer: AdamW
```

Main split:

```text
Train/validation source: RW9 + RW10
Validation: 10% stratified windows
Test: RW11
RW12: unused in this final model
```

## XAI

Model behavior was interpreted using:

- global permutation importance
- SHAP-style gradient attribution
- local LIME explanations for charge and discharge examples

