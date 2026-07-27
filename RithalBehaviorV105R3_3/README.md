# Rithal V1.0.5 R3.3 — exact five-class regime contract

## Proven root cause

The deployed `MythosNeuralV2` architecture has five trained regime classes:

1. `TREND_UP`
2. `TREND_DOWN`
3. `CHOP`
4. `BREAKOUT`
5. `PANIC`

The R3/R3.1 behavior scorer required exactly four logits. R3.2 preserved an
active scorer, but its synthetic certification also supplied only four regime
probabilities and assumed the active scorer already published `regime_valid`.
The actual canonical scorer publishes five true/rank probabilities but did not
publish that explicit flag. Consequently, startup's synthetic four-logit check
passed while real five-logit checkpoint inference failed closed with:

```text
regime_unknown_or_invalid_four_class_posterior
```

## What R3.3 changes

- accepts **exactly five** finite regime logits;
- computes `true_regime_probs = softmax(regime_logits)`;
- computes `rank_regime_probs = softmax(true_regime_probs)`;
- maps trend, chop, breakout and panic to their actual trained classes;
- restores the canonical raw-composite/rank-distribution calculation;
- keeps the V1.0.5 risk-adjusted side overlay;
- fixes the rich logger so `p_breakout` prints the breakout field;
- changes the synthetic startup probe from four logits to five;
- uses a numeric fail-closed fallback (`p_chop=1`, `p_panic=1`) for malformed
  outputs so no entry can pass and logging cannot crash.

## Explicitly unchanged

- all six checkpoints and model weights;
- the 168-feature order and scaler state;
- static symbol score thresholds;
- TP/SL, fees, leverage, margin and heat;
- paper wallet, ledger and accounting;
- the data engine and enrichment;
- inference routing, feature preparation and rank prewarming;
- Trade Manager logic, settings and authority.

The installer hashes the protected files and all six checkpoints before and
after installation and rolls back if any protected hash changes.

## Install

Stop the live engine first. From the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\RithalBehaviorV105R3_3\Install-RithalBehaviorV105R3_3.ps1 -ProjectRoot .
```

Then restart with the existing project command:

```powershell
.\Start-Rithal18KContractV2.ps1
```

## Verify

Source-only verification:

```powershell
python .\RithalBehaviorV105R3_3\Verify-RithalBehaviorV105R3_3.py --project-root . --source-only
```

Runtime import verification while the engine is stopped:

```powershell
python .\RithalBehaviorV105R3_3\Verify-RithalBehaviorV105R3_3.py --project-root .
```

Expected startup marker:

```text
[RITHAL_BEHAVIOR_FIX_V1_0_5_R3_3] installed: exact 5-class regime contract
```

At the first fully streamed 15-minute candle, `REGIME-P` must contain finite
trend, chop and breakout values. The feature contract must not contain
`regime_unknown_or_invalid_four_class_posterior`.
