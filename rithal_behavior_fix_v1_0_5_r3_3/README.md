# Rithal Behavior Fix V1.0.5 R3.3

## Proven defect

R3.2 restored the engine's canonical `_score_model_output` and canonical rank prewarm, while the V1.0.5 family-repair `infer` wrapper remained active. The two contracts were incompatible:

- the canonical scorer publishes `true_regime_probs` and `rank_regime_probs`;
- the V1.0.5 inference wrapper requires `regime_valid`, `p_trend`, `p_chop`, `p_breakout` and `regime_state`.

The valid posterior therefore reached the inference wrapper without its compatibility fields. All six symbols were blocked with `regime_unknown_or_invalid_four_class_posterior`, and the rich logger attempted to format `None` values.

A second issue existed in the R3.2 side overlay: the canonical score dictionary did not expose MAE and direction-confidence heads, so the overlay silently used default MAE values while recomputing long/short edges. R3.3 reuses the canonical edges, which were already calculated with the real model heads, and extracts the real raw-head values only for audit telemetry and direction bias.

## Preserved without modification

- model checkpoints and weights;
- 168-feature order and scaler artifacts;
- canonical `raw_composite`;
- canonical true and rank posterior arrays;
- canonical rank-prewarm distribution;
- static per-symbol score thresholds;
- entry and exit fees;
- TP/SL geometry;
- paper balance, margin, leverage and heat settings;
- position ledger and accounting;
- R3.1 coherent family repair;
- V1.0.5 fail-closed inference and thesis handling;
- existing paper manager authority path.

## Changed

- a compatibility projection derives scalar regime telemetry from the existing four-value posterior arrays without replacing those arrays;
- risk-adjusted side selection reuses canonical MAE-based long/short edges;
- the rich logger reads `p_breakout` rather than `p_panic` and prints `nan` rather than raising when invalid telemetry is intentionally fail-closed;
- an explicit startup activation block ensures R3.3 supersedes R3.2 before later duplicate activation attempts.

The installer edits only `mythos/neural/live.py` and adds `mythos/neural/rithal_behavior_fix_v105_r3_3.py`, verifier, rollback and report files. Existing behavior modules are hashed before and after and must remain unchanged.
