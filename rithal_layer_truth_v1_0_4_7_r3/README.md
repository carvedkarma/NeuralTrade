# Rithal Layer Truth V1.0.4.7 R3.2

Canonical non-retraining layer repair for:

- instance `rithal-1-0-contract-locked`
- checkpoints `checkpoints\rithal_clean`
- starting paper profile `$18,000 / 10% fixed margin / 10x leverage / 60% allocation cap / 3% heat cap`
- maximum concurrent slots `6`, with risk/heat guards remaining independent
- manager authority `SHADOW_ONLY`
- live-exchange authority unchanged and disabled

R3.2 is the only supported installer in this branch. It installs an instance-aware, read-only truth sidecar; resolves manager decisions by exact trade/ticket identity; prioritizes actionable decisions before source priority; separates recommendation, authority, and execution; publishes true regime probabilities and breakout probability; detects sizing contradictions and runtime profile drift; and runs revised manager policies only as shadow counterfactuals.

## Install

Run from `C:\Users\muham\Downloads\mythos_v24_full`:

```powershell
$u='https://raw.githubusercontent.com/carvedkarma/NeuralTrade/rithal-layer-truth-v1.0.4.7-r3/rithal_layer_truth_v1_0_4_7_r3/Install-RithalLayerTruthV1_0_4_7_R3_2.ps1'; Invoke-WebRequest -UseBasicParsing $u -OutFile .\Install-RithalLayerTruthV1_0_4_7_R3_2.ps1; powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-RithalLayerTruthV1_0_4_7_R3_2.ps1 -ProjectRoot .; if ($LASTEXITCODE -eq 0) { .\Start-RithalLayerTruthV1_0_4_7.ps1 }
```

## Verify

```powershell
.\Verify-RithalLayerTruthV1_0_4_7.ps1
```

`PASS WITH WARNINGS` means the runtime is operating but the observed live profile or a source file differs from the locked contract. It is not silently converted to a pass.

Dashboard endpoint:

```text
http://127.0.0.1:8888/api/rithal-layer-truth
```

## Roll back

```powershell
.\Rollback-RithalLayerTruthV1_0_4_7.ps1
```

The installer backs up every file it changes, preserves the existing control document's schema/module and authority metadata, and does not reset current realised paper equity.
