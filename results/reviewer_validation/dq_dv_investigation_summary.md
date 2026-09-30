# Sequential dQ/dV Investigation

## Stage 1: Baseline reproducibility

The seed-42 full-feature RW9/RW10-to-RW11 Transformer reproduced the manuscript result exactly: event-level R2 0.955082, MAE 2.386664, and RMSE 3.038808; diagnostic-checkpoint R2 0.966387, MAE 2.068949, and RMSE 2.535021. This is Tier 1 evidence for deterministic reproduction under the recorded environment and seed.

## Stage 2: Paired-seed RW11 ablation

Across paired seeds 42, 43, and 44, removing maximum local dQ/dV changed event RMSE by -0.092801, -0.033997, and -0.064324 SOH points, respectively. Diagnostic-checkpoint RMSE changed by -0.141642, -0.021584, and -0.089482. Negative values favor removal. The mean changes were -0.063707 +/- 0.029407 at event level and -0.084236 +/- 0.060200 at checkpoint level. The direction repeated across all three seeds, although the effect was small.

## Stage 3: Cross-cell ablation

At seed 42, event RMSE changed by -0.060159 for held-out RW9, -0.050162 for RW10, and -0.092801 for RW11. Checkpoint RMSE changed by -0.029556, -0.097851, and -0.141642, respectively. Mean event RMSE changed from 2.902567 to 2.834860, and mean checkpoint RMSE changed from 2.200457 to 2.110774. The feature was therefore not necessary for predictive accuracy in any tested fold.

## Stage 4: Global Gradient SHAP redistribution

Using matched seed-42 RW11 models, 128 identical training-background windows, and 512 identical test windows, the largest increases in normalized attribution share after removal occurred for throughput magnitude (+0.054362), signed throughput (+0.051719), voltage spread (+0.050947), and energy magnitude (+0.041050). This directly demonstrates attribution redistribution for these fitted models, but it does not establish causal reconstruction of dQ/dV.

## Stage 5: Matched local LIME

The same temporal-median charge-ending and discharge-ending RW11 windows were explained for both models. Local coefficient magnitudes increased most for voltage spread, throughput, energy magnitude, voltage maximum, and current magnitude. This local result is directionally consistent with global redistribution but is limited to two predefined windows and remains surrogate-dependent.

## Stage 6: Training-only dependence and redundancy

Using RW9 and RW10 only, dQ/dV had Spearman correlations of 0.798221 with energy magnitude and 0.793870 with throughput magnitude. Cross-cell nonlinear prediction of dQ/dV from all remaining descriptors achieved mean R2 0.966905; loading-only descriptors achieved 0.931192 and voltage-only descriptors achieved 0.660477. These results provide direct evidence of substantial representational redundancy, not causal equivalence.

## Stages 7 and 8: Architecture interaction

The Transformer and vanilla RNN were each trained with and without dQ/dV under paired seeds 42-44. Transformer event RMSE improved slightly in all seeds. RNN event RMSE improved in seeds 42 and 44 but worsened slightly in seed 43. RNN checkpoint RMSE improved in every seed. The event-RMSE architecture interaction averaged -0.179518 with a wide 95% interval [-0.701096, 0.342060], so a general event-level interaction was not established. The checkpoint-RMSE interaction averaged -0.160566 with a three-seed interval [-0.205081, -0.116051], indicating a larger checkpoint-scale removal benefit for the RNN under these runs.

## Stopping decisions

Cross-architecture XAI was not performed because architecture-dependent event behavior was not sufficiently stable. Attention analysis was not performed because no stable Transformer-specific mechanism had been established. Representation analysis was also skipped. No claim is made that attention heads reconstruct or recover dQ/dV information.

## Stage 12: Prediction regimes

Across three Transformer seeds on RW11, removal improved RMSE for charge and discharge events and nearly all examined regimes. The largest mean changes occurred for short-duration events (-0.091462), middle-life SOH (-0.086969), high-voltage events (-0.082324), and high-current events (-0.075367). Long-duration events were effectively unchanged (-0.002109), and early-life SOH showed a small variable worsening (+0.008986 +/- 0.110430).

## Evidence-based conclusion

The complete Transformer uses dQ/dV according to Gradient SHAP and LIME (Tier 1 attribution evidence). dQ/dV is not necessary for predictive accuracy under the tested protocol (Tier 1 ablation evidence). The small removal benefit is reproducible across three RW11 seeds and across the three seed-42 held-out cells, but its mechanism is not established. Strong training-only associations and cross-cell predictability support partial redundancy with throughput, energy, duration, and voltage-spread descriptors (Tier 2 interpretation). A noise or regularization mechanism remains Tier 3. A causal reconstruction mechanism and attention-head recovery remain unsupported and must not be claimed.
