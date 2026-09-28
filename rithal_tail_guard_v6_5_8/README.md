# Rithal Tail Guard V6.5.8

V6.5.8 fixes the final false-negative seen after V6.5.7 successfully committed the full-context live overlay.

## Proven production shape

The transaction committed, all six symbols showed 100% source coverage, and every published file ended at the immutable run anchor. The run then crossed one 15-minute wall-clock boundary before final validation. The guard calculated `latest_bar_age_min` from completion time, not the run anchor, so the anchored candle appeared 41 minutes old and failed the 35-minute limit even though it was only 26 minutes old relative to the immutable anchor.

## Correction

- Keep wall-clock age as diagnostic telemetry.
- Add `anchored_latest_bar_age_min` measured from `anchor_now_ms`.
- Use anchored age for the Python feature-publication contract.
- Leave the existing one-bar PowerShell rollover handshake responsible for wall-clock catch-up.
- Preserve the V6.5.7 full-context 16,384-row live-overlay rebase.
- Preserve all strict 132-column parity checks and tolerances.
- Preserve the immutable five-year canonical archive.
- Do not touch the model, checkpoints, thresholds, ledger, positions, fees, TP/SL, manager or dashboards.

The installer is cumulative: it reapplies the validated V6.5.6 base, the corrected V6.5.7 full-context rebase, then the V6.5.8 anchored freshness delta. This is required because a failed earlier installer restored the three source files even though the data transaction itself had already committed.
