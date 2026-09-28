# Rithal Layer Truth V1.0.4.7 R1

Incremental non-retraining repair for the locked Rithal instance:

- instance: `rithal-1-0-contract-locked`
- models: `checkpoints\rithal_clean`
- paper contract metadata: `$18,000` starting equity, `10%` fixed margin, `10x` leverage, `60%` allocation cap, `3%` heat cap, two maximum open positions
- manager authority: `SHADOW_ONLY`
- exchange authority: unchanged and disabled

## What it installs

- `mythos\neural\rithal_layer_truth_v1047.py`
- exact trade/ticket manager binding with `LINKED`, `UNBOUND`, `STALE`, and `UNAVAILABLE`
- source priority `V3.3 > V3.2 > Exit Brain > legacy`
- separate `recommended_action`, `effective_authority`, and `actual_execution`
- true four-class regime projection with real breakout probability and `REGIME_UNKNOWN`
- risk-adjusted side comparison in shadow
- proposed/effective/applied sizing truth
- complete exact-bound entry-thesis projection when the interaction journal carries the trade or ticket identity
- revised manager counterfactuals for earlier loss recognition, mature-profit stagnation, and persistent P80 erosion; all remain shadow-only
- dashboard endpoint `/api/rithal-layer-truth`
- transactional backup, rollback, compile check, and runtime verifier

The installer does not reset the realised paper wallet. It publishes the locked `$18,000` starting contract as metadata and aligns Dashboard 1's next-session policy to `10% / 10x / 60%`.

## Install

Run from:

```powershell
C:\Users\muham\Downloads\mythos_v24_full
```

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-layer-truth-v1.0.4.7/rithal_layer_truth_v1_0_4_7/Install-RithalLayerTruthV1_0_4_7_R1.ps1'; Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalLayerTruthV1_0_4_7_R1.ps1; powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalLayerTruthV1_0_4_7_R1.ps1 -ProjectRoot .; .\Start-RithalLayerTruthV1_0_4_7.ps1
```

## Verify

```powershell
.\Verify-RithalLayerTruthV1_0_4_7.ps1
```

Expected terminal line:

```text
[RITHAL_LAYER_TRUTH_V1_0_4_7] VERIFICATION PASS
```

Dashboard API:

```text
http://127.0.0.1:8888/api/rithal-layer-truth
```

## Roll back

The installer prints the timestamped backup directory. Use the generated rollback command:

```powershell
.\Rollback-RithalLayerTruthV1_0_4_7.ps1
```
