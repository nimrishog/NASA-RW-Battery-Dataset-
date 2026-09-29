# Reviewer-validation experiments

This directory contains the held-out RW11 and leave-one-cell-out results used in the revised manuscript.

- `full_transformer_loco_metrics.csv`: full Transformer leave-one-cell-out metrics for RW9, RW10, and RW11.
- `controlled_rnn_metrics.json`: vanilla RNN baseline trained with the same event features, 32-event windows, split, preprocessing, weighting, validation rule, and evaluation samples as the primary Transformer.
- `transformer_sensitivity/transformer_sensitivity_metrics.csv`: separately retrained Transformer results for label strategies, window lengths, feature groups, local dQ/dV removal, and block-weight removal.
- `transformer_sensitivity/*.png`: figures generated from the Transformer sensitivity table.

The corresponding entry points are:

- `scripts/run_full_transformer_loco_reviewer.py`
- `scripts/run_controlled_simple_rnn_rw11_reviewer.py`
- `scripts/run_transformer_sensitivity_reviewer.py`
