/**
 * V7 Path A Paper-Trading Engine
 *
 * Current runtime policy (V7 "Precision" adaptation):
 *   - Tradeable book: ADA, XRP
 *   - AVAX/SOL/ETH/BTC/BNB excluded from live V7 universe
 *   - Selectivity: top 2.0% of |pred| via rolling 30-day quantile per symbol
 *     (warmup: 50 prior samples min before first trade)
 *   - Geometry: fixed 60-minute market exit; NO stop, NO take-profit, NO trail
 *   - Volatility floor: only enter when realized vol_16 >= 100 bps
 *   - One live position per symbol (no stacking)
 *   - Per-symbol performance gate:
 *       recent net mean/cumulative <= 0 at assumed costs -> block new entries
 *   - Deep loss kill-switch remains as a hard backstop (manual resume required)
 *   - Default DISABLED on boot. Operator must POST /api/v7/enable to start.
 *
 * Signal source: hooks into updateGPUPrediction() in ml-predictor.ts.
 *   Uses |pred.returnH2| as the |pred| magnitude (60-min horizon, matching
 *   the back-test's sign_60m target). Direction = sign(pred.returnH2).
 *
 * Entry/exit prices: pulled from the `candles` table (15m timeframe) so paper
 *   P&L matches the close-to-close convention used in the back-test.
 *
 * Persistence: in-memory state mirrored to .local/v7_path_a_state.json on
 *   change so a restart doesn't force the 30-day warm-up.
 *
 * Reports gross P&L; net P&L at 4/6/8 bps assumed costs is computed at read
 * time in the API/dashboard so we don't bake a cost assumption into storage.
 */

import { promises as fs } from "fs";
import path from "path";
import { db } from "../db";
import { candles } from "@shared/schema";
import { and, desc, eq, gte, lte } from "drizzle-orm";
import * as storage from "./storage";
import type { GPUPrediction } from "../ml-predictor";
import type { PaperPosition } from "@shared/schema";

// ---- Config (locked) ----
export const V7_TRADEABLE = ["ADAUSDT", "XRPUSDT"] as const;
export const V7_PROBATIONARY = [] as const;
export const V7_DISABLED = ["AVAXUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT"] as const;
export const V7_UNIVERSE = [...V7_TRADEABLE, ...V7_PROBATIONARY] as const;
export const V7_HOLD_BARS = 4;                       // 60 minutes / 15 min
export const V7_HOLD_MS = V7_HOLD_BARS * 15 * 60 * 1000;
export const V7_TOP_FRACTION = 0.02;                 // top 2.0%
export const V7_MIN_VOL16_BPS = 100;                 // precision gate
export const V7_ROLLING_WINDOW_DAYS = 30;
export const V7_ROLLING_WINDOW_MS = V7_ROLLING_WINDOW_DAYS * 86400 * 1000;
export const V7_MIN_BUFFER_SAMPLES = 50;             // warm-up gate
export const V7_KILL_THRESHOLD_BPS = -1500;          // per-symbol cum net @6bps
export const V7_KILL_COST_BPS = 6;                   // assumed cost for kill-switch evaluation
export const V7_PERF_GATE_COST_BPS = 8;              // mirrors strict offline gate
export const V7_PERF_GATE_MIN_TRADES = 20;           // wait for enough samples
export const V7_PERF_GATE_LOOKBACK_TRADES = 120;
export const V7_SOURCE_TRADEABLE = "v7_path_a";
export const V7_SOURCE_PROBATIONARY = "v7_path_a_prob";
const STATE_PATH = path.resolve(".local/v7_path_a_state.json");

// ---- State (persisted) ----
interface V7PerfGateState {
  recentNetBps: number[];
  rollingMeanNetBps: number;
  rollingCumNetBps: number;
  blocked: boolean;
  updatedTs: number;
  reason?: string;
}

interface V7State {
  enabled: boolean;
  notionalUsd: number;
  predBuffer: Record<string, Array<{ ts: number; predAbs: number }>>;
  // cumGrossBps is the running sum since the last (re)set baseline for the symbol.
  cumGrossBps: Record<string, number>;
  // Number of historical closed v7 positions at the moment the symbol's
  // cumGrossBps baseline was last reset (boot or manual resume). Used so the
  // kill-switch net calculation only counts trades since the active baseline.
  closedAtBaseline: Record<string, number>;
  killed: Record<string, { ts: number; cumGrossBps: number; cumNetBps: number; reason: string } | null>;
  perfGate: Record<string, V7PerfGateState>;
  thresholdSnapshots: Array<{ symbol: string; ts: number; threshold: number; n: number }>;
}

let state: V7State = {
  enabled: process.env.V7_PAPER_ENABLED === "true",
  notionalUsd: parseFloat(process.env.V7_NOTIONAL_USD || "100"),
  predBuffer: {},
  cumGrossBps: {},
  closedAtBaseline: {},
  killed: {},
  perfGate: {},
  thresholdSnapshots: [],
};

let savePending = false;
async function persist(): Promise<void> {
  if (savePending) return;
  savePending = true;
  setTimeout(async () => {
    savePending = false;
    try {
      await fs.mkdir(path.dirname(STATE_PATH), { recursive: true });
      await fs.writeFile(STATE_PATH, JSON.stringify(state, null, 2));
    } catch (e) {
      console.error("[V7] persist failed:", e);
    }
  }, 250);
}

async function hydrate(): Promise<void> {
  try {
    const raw = await fs.readFile(STATE_PATH, "utf8");
    const loaded = JSON.parse(raw) as Partial<V7State>;
    state = {
      enabled: state.enabled,                                      // env wins on boot
      notionalUsd: loaded.notionalUsd ?? state.notionalUsd,
      predBuffer: loaded.predBuffer ?? {},
      cumGrossBps: loaded.cumGrossBps ?? {},
      closedAtBaseline: loaded.closedAtBaseline ?? {},
      killed: loaded.killed ?? {},
      perfGate: loaded.perfGate ?? {},
      thresholdSnapshots: loaded.thresholdSnapshots ?? [],
    };
    for (const sym of V7_UNIVERSE) {
      if (!state.perfGate[sym]) {
        state.perfGate[sym] = {
          recentNetBps: [],
          rollingMeanNetBps: 0,
          rollingCumNetBps: 0,
          blocked: false,
          updatedTs: 0,
        };
      }
    }
    // Trim buffers to current 30d window
    const cutoff = Date.now() - V7_ROLLING_WINDOW_MS;
    for (const sym of Object.keys(state.predBuffer)) {
      state.predBuffer[sym] = (state.predBuffer[sym] || []).filter(x => x.ts >= cutoff);
    }
    // Trim threshold snapshots to last 90d
    const snapCutoff = Date.now() - 90 * 86400 * 1000;
    state.thresholdSnapshots = state.thresholdSnapshots.filter(s => s.ts >= snapCutoff);
    console.log(`[V7] hydrated state from ${STATE_PATH}: enabled=${state.enabled} notional=$${state.notionalUsd}`);
  } catch (e: any) {
    if (e.code !== "ENOENT") console.error("[V7] hydrate failed:", e);
    console.log("[V7] starting with fresh state");
  }
}

function updatePerfGate(symbol: string): void {
  const perf = state.perfGate[symbol] ?? {
    recentNetBps: [],
    rollingMeanNetBps: 0,
    rollingCumNetBps: 0,
    blocked: false,
    updatedTs: 0,
  };
  if (perf.recentNetBps.length > V7_PERF_GATE_LOOKBACK_TRADES) {
    perf.recentNetBps = perf.recentNetBps.slice(-V7_PERF_GATE_LOOKBACK_TRADES);
  }
  const n = perf.recentNetBps.length;
  const sum = perf.recentNetBps.reduce((a, b) => a + b, 0);
  const mean = n > 0 ? sum / n : 0;
  const shouldBlock = n >= V7_PERF_GATE_MIN_TRADES && (mean <= 0 || sum <= 0);
  const changed = perf.blocked !== shouldBlock;
  perf.rollingMeanNetBps = mean;
  perf.rollingCumNetBps = sum;
  perf.blocked = shouldBlock;
  perf.updatedTs = Date.now();
  perf.reason = shouldBlock
    ? `Perf gate: n=${n}, mean=${mean.toFixed(2)}bps, cum=${sum.toFixed(1)}bps <= 0`
    : undefined;
  state.perfGate[symbol] = perf;
  if (changed) {
    if (shouldBlock) {
      console.warn(`[V7] ⚠️ PERF GATE BLOCKED ${symbol}: n=${n} mean=${mean.toFixed(2)}bps cum=${sum.toFixed(1)}bps`);
    } else {
      console.log(`[V7] ✅ PERF GATE UNBLOCKED ${symbol}: n=${n} mean=${mean.toFixed(2)}bps cum=${sum.toFixed(1)}bps`);
    }
  }
}

// ---- Buffer / threshold helpers ----
function pushPrediction(symbol: string, ts: number, predAbs: number): void {
  if (!state.predBuffer[symbol]) state.predBuffer[symbol] = [];
  const buf = state.predBuffer[symbol];
  buf.push({ ts, predAbs });
  const cutoff = ts - V7_ROLLING_WINDOW_MS;
  while (buf.length > 0 && buf[0].ts < cutoff) buf.shift();
}

function rollingThreshold(symbol: string, p: number = V7_TOP_FRACTION): number | null {
  const buf = state.predBuffer[symbol] || [];
  if (buf.length < V7_MIN_BUFFER_SAMPLES) return null;
  const vals = buf.map(x => x.predAbs).sort((a, b) => a - b);
  const q = Math.max(0, Math.min(1, 1 - p));
  const idx = Math.min(vals.length - 1, Math.floor(q * (vals.length - 1)));
  return vals[idx];
}

function snapshotThreshold(symbol: string, ts: number, threshold: number, n: number): void {
  // De-dupe: only one snapshot per symbol per UTC day
  const day = Math.floor(ts / 86400000);
  const lastForSym = [...state.thresholdSnapshots].reverse()
    .find(s => s.symbol === symbol);
  if (lastForSym && Math.floor(lastForSym.ts / 86400000) === day) return;
  state.thresholdSnapshots.push({ symbol, ts, threshold, n });
  if (state.thresholdSnapshots.length > 5000) {
    state.thresholdSnapshots = state.thresholdSnapshots.slice(-3000);
  }
}

// ---- Candle helpers ----
async function latestCandle(symbol: string, atOrBeforeTs?: number): Promise<{ ts: number; close: number } | null> {
  const conds = [eq(candles.symbol, symbol), eq(candles.timeframe, "15m")];
  if (atOrBeforeTs !== undefined) conds.push(lte(candles.timestamp, atOrBeforeTs));
  const rows = await db.select({ ts: candles.timestamp, close: candles.close })
    .from(candles)
    .where(and(...conds))
    .orderBy(desc(candles.timestamp))
    .limit(1);
  if (rows.length === 0) return null;
  return { ts: rows[0].ts, close: Number(rows[0].close) };
}

async function candleAt(symbol: string, ts: number): Promise<{ ts: number; close: number } | null> {
  const rows = await db.select({ ts: candles.timestamp, close: candles.close })
    .from(candles)
    .where(and(
      eq(candles.symbol, symbol),
      eq(candles.timeframe, "15m"),
      gte(candles.timestamp, ts),
    ))
    .orderBy(candles.timestamp)
    .limit(1);
  if (rows.length === 0) return null;
  return { ts: rows[0].ts, close: Number(rows[0].close) };
}

async function recentVol16Bps(symbol: string, atOrBeforeTs: number): Promise<number | null> {
  const rows = await db.select({ ts: candles.timestamp, close: candles.close })
    .from(candles)
    .where(and(
      eq(candles.symbol, symbol),
      eq(candles.timeframe, "15m"),
      lte(candles.timestamp, atOrBeforeTs),
    ))
    .orderBy(desc(candles.timestamp))
    .limit(17);
  if (rows.length < 17) return null;
  const closes = [...rows]
    .reverse()
    .map(r => Number(r.close))
    .filter(v => isFinite(v) && v > 0);
  if (closes.length < 17) return null;
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    rets.push(Math.log(closes[i] / closes[i - 1]));
  }
  if (rets.length < 16) return null;
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const varPop = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length;
  if (!isFinite(varPop) || varPop <= 0) return 0;
  return Math.sqrt(varPop) * 1e4;
}

// ---- Open / close ----
async function openV7Position(
  symbol: string,
  side: "LONG" | "SHORT",
  entryPrice: number,
  entryTs: number,
  predAbs: number,
): Promise<PaperPosition | null> {
  const isProbationary = (V7_PROBATIONARY as readonly string[]).includes(symbol);
  const source = isProbationary ? V7_SOURCE_PROBATIONARY : V7_SOURCE_TRADEABLE;
  const notional = state.notionalUsd;
  const qty = notional / entryPrice;
  if (!isFinite(qty) || qty <= 0) return null;

  const pos = await storage.createPosition({
    symbol,
    side,
    status: "OPEN",
    entryTs,
    entryPrice,
    qty,
    notionalUsdt: notional,
    leverage: 1,
    stopLoss: null,
    tp1: null,
    tp2: null,
    trailMode: "none",
    trailPrice: null,
    trailActive: 0,
    trailBestPrice: null,
    timeStopBars: V7_HOLD_BARS,
    barsOpen: 0,
    primaryHorizon: 60,
    initialRiskUsdt: notional,
    feesPaidUsdt: 0,
    fundingPaidUsdt: 0,
    exitTs: null,
    exitPrice: null,
    realizedPnlUsdt: null,
    exitReason: null,
    signalConfidence: predAbs,
    signalEdge: null,
    v5Score: null,
    peakProfit: 0,
    initialStopDistance: null,
    regime: null,
    source,
  });

  await storage.createTrade({
    positionId: pos.id,
    ts: entryTs,
    action: "OPEN",
    price: entryPrice,
    qty,
    feeUsdt: 0,
    slippageUsdt: 0,
    fundingUsdt: 0,
    pnlUsdt: 0,
    reason: `V7 ${source} ${side} @ ${entryPrice} (predAbs=${predAbs.toExponential(3)})`,
  });

  console.log(`[V7] OPEN ${source} ${symbol} ${side} qty=${qty.toFixed(6)} @ ${entryPrice} predAbs=${predAbs.toExponential(3)}`);
  return pos;
}

async function closeV7Position(pos: PaperPosition, exitPrice: number, exitTs: number, reason: string): Promise<void> {
  const dir = pos.side === "LONG" ? 1 : -1;
  const grossPnl = (exitPrice - pos.entryPrice) * pos.qty * dir;
  const grossBps = Math.log(exitPrice / pos.entryPrice) * dir * 1e4;

  await storage.updatePosition(pos.id, {
    status: "CLOSED",
    exitTs,
    exitPrice,
    realizedPnlUsdt: grossPnl,
    exitReason: reason,
  });
  await storage.createTrade({
    positionId: pos.id,
    ts: exitTs,
    action: "CLOSE",
    price: exitPrice,
    qty: pos.qty,
    feeUsdt: 0,
    slippageUsdt: 0,
    fundingUsdt: 0,
    pnlUsdt: grossPnl,
    reason: `V7 ${reason} grossBps=${grossBps.toFixed(2)}`,
  });

  state.cumGrossBps[pos.symbol] = (state.cumGrossBps[pos.symbol] || 0) + grossBps;
  const netForGate = grossBps - V7_PERF_GATE_COST_BPS;
  const perf = state.perfGate[pos.symbol] ?? {
    recentNetBps: [],
    rollingMeanNetBps: 0,
    rollingCumNetBps: 0,
    blocked: false,
    updatedTs: 0,
  };
  perf.recentNetBps.push(netForGate);
  state.perfGate[pos.symbol] = perf;
  updatePerfGate(pos.symbol);
  // Kill-switch evaluated on cumulative NET at the assumed cost level. We
  // count only trades closed since the active baseline (boot or last manual
  // resume) so a resumed symbol gets a fresh probationary window rather than
  // re-tripping on the next trade.
  const closedCount = await countClosedV7Trades(pos.symbol);
  const baseline = state.closedAtBaseline[pos.symbol] ?? 0;
  const tradesSinceBaseline = Math.max(0, closedCount - baseline);
  const cumNetBps = (state.cumGrossBps[pos.symbol] || 0) - tradesSinceBaseline * V7_KILL_COST_BPS;
  if (cumNetBps < V7_KILL_THRESHOLD_BPS && !state.killed[pos.symbol]) {
    state.killed[pos.symbol] = {
      ts: Date.now(),
      cumGrossBps: state.cumGrossBps[pos.symbol] || 0,
      cumNetBps,
      reason: `Cumulative net ${cumNetBps.toFixed(0)} bps < ${V7_KILL_THRESHOLD_BPS} bps threshold`,
    };
    console.log(`[V7] 🚨 KILL SWITCH ENGAGED for ${pos.symbol}: cumNet=${cumNetBps.toFixed(0)} bps (manual resume only)`);
  }
  await persist();
  console.log(`[V7] CLOSE ${pos.symbol} ${pos.side} @ ${exitPrice} grossBps=${grossBps.toFixed(2)} reason=${reason}`);
}

async function countClosedV7Trades(symbol: string): Promise<number> {
  const rows = await storage.getPositionsBySymbol(symbol, "CLOSED", 10000);
  return rows.filter(r => r.source === V7_SOURCE_TRADEABLE || r.source === V7_SOURCE_PROBATIONARY).length;
}

// ---- Public hooks ----

/** Called from updateGPUPrediction(). Drives all V7 entries. */
export async function onV7Prediction(pred: GPUPrediction): Promise<void> {
  const sym = pred.symbol;
  if (!(V7_UNIVERSE as readonly string[]).includes(sym)) return;
  if (!isFinite(pred.returnH2) || pred.returnH2 === 0) return;

  const predAbs = Math.abs(pred.returnH2);
  pushPrediction(sym, pred.timestamp, predAbs);
  await persist();

  if (!state.enabled) return;
  if (state.killed[sym]) return;
  if (state.perfGate[sym]?.blocked) return;

  const threshold = rollingThreshold(sym, V7_TOP_FRACTION);
  if (threshold === null) return;          // warm-up
  snapshotThreshold(sym, pred.timestamp, threshold, state.predBuffer[sym].length);

  if (predAbs < threshold) return;

  // Entry: use the close of the candle whose timestamp matches the prediction's
  // bar (or the latest candle if we can't find an exact match).
  const c = await latestCandle(sym, pred.timestamp + 60_000);
  if (!c) {
    console.warn(`[V7] no candle for ${sym} at/before ${pred.timestamp}, skipping signal`);
    return;
  }
  const vol16Bps = await recentVol16Bps(sym, c.ts);
  if (vol16Bps == null) return;
  if (vol16Bps < V7_MIN_VOL16_BPS) return;
  // No stacking: at most one OPEN V7 position per symbol.
  const existing = await storage.getPositionsBySymbol(sym, "OPEN", 10);
  const hasOpen = existing.some(p =>
    (p.source === V7_SOURCE_TRADEABLE || p.source === V7_SOURCE_PROBATIONARY));
  if (hasOpen) return;

  const side: "LONG" | "SHORT" = pred.returnH2 > 0 ? "LONG" : "SHORT";
  await openV7Position(sym, side, c.close, c.ts, predAbs);
}

/** Background tick: check for 60-min exits. Called every ~30s. */
export async function tickV7(): Promise<void> {
  const open = await storage.getPositions("OPEN", 1000);
  const v7Open = open.filter(p => p.source === V7_SOURCE_TRADEABLE || p.source === V7_SOURCE_PROBATIONARY);
  const now = Date.now();
  for (const pos of v7Open) {
    const exitDueTs = pos.entryTs + V7_HOLD_MS;
    if (now < exitDueTs) continue;
    // Find the candle closest to (entryTs + V7_HOLD_MS)
    const c = (await candleAt(pos.symbol, exitDueTs)) ?? (await latestCandle(pos.symbol));
    if (!c) continue;
    await closeV7Position(pos, c.close, c.ts, "TIME_EXIT_60M");
  }
}

let tickTimer: NodeJS.Timeout | null = null;
export function startV7Engine(intervalMs: number = 30_000): void {
  hydrate().then(() => {
    if (tickTimer) clearInterval(tickTimer);
    tickTimer = setInterval(() => {
      tickV7().catch(e => console.error("[V7] tick error:", e));
    }, intervalMs);
    console.log(`[V7] engine started (enabled=${state.enabled} notional=$${state.notionalUsd} interval=${intervalMs}ms)`);
  });
}

export function stopV7Engine(): void {
  if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
}

// ---- Operator API ----
export function getV7State() {
  return {
    enabled: state.enabled,
    notionalUsd: state.notionalUsd,
    universe: {
      tradeable: V7_TRADEABLE,
      probationary: V7_PROBATIONARY,
      disabled: V7_DISABLED,
    },
    config: {
      topPct: V7_TOP_FRACTION * 100,
      holdMinutes: V7_HOLD_BARS * 15,
      rollingDays: V7_ROLLING_WINDOW_DAYS,
      minBufferSamples: V7_MIN_BUFFER_SAMPLES,
      minVol16Bps: V7_MIN_VOL16_BPS,
      killThresholdBps: V7_KILL_THRESHOLD_BPS,
      killCostBps: V7_KILL_COST_BPS,
      perfGateCostBps: V7_PERF_GATE_COST_BPS,
      perfGateMinTrades: V7_PERF_GATE_MIN_TRADES,
      perfGateLookbackTrades: V7_PERF_GATE_LOOKBACK_TRADES,
    },
    bufferSizes: Object.fromEntries(
      [...V7_UNIVERSE].map(s => [s, (state.predBuffer[s] || []).length])
    ),
    thresholds: Object.fromEntries(
      [...V7_UNIVERSE].map(s => [s, rollingThreshold(s)])
    ),
    cumGrossBps: { ...state.cumGrossBps },
    killed: { ...state.killed },
    perfGate: Object.fromEntries(
      [...V7_UNIVERSE].map(s => [s, state.perfGate[s] ?? null])
    ),
    recentSnapshots: state.thresholdSnapshots.slice(-20),
  };
}

export function setV7Enabled(v: boolean): void {
  state.enabled = v;
  persist();
  console.log(`[V7] enabled = ${v}`);
}

export function setV7Notional(usd: number): void {
  if (!isFinite(usd) || usd <= 0) throw new Error("notional must be > 0");
  state.notionalUsd = usd;
  persist();
}

export async function manualResume(symbol: string): Promise<{ resumed: boolean; reason?: string }> {
  if (!(V7_UNIVERSE as readonly string[]).includes(symbol)) {
    return { resumed: false, reason: "symbol not in V7 universe" };
  }
  const perfBlocked = state.perfGate[symbol]?.blocked === true;
  if (!state.killed[symbol] && !perfBlocked) {
    return { resumed: false, reason: "symbol is not blocked" };
  }
  // Reset the kill-switch counters so the symbol gets a fresh probationary
  // window: zero out cumGrossBps and rebase the closed-trade count to "now"
  // so the next trade isn't immediately re-killed by the historical deficit.
  const closedNow = await countClosedV7Trades(symbol);
  state.cumGrossBps[symbol] = 0;
  state.closedAtBaseline[symbol] = closedNow;
  state.killed[symbol] = null;
  state.perfGate[symbol] = {
    recentNetBps: [],
    rollingMeanNetBps: 0,
    rollingCumNetBps: 0,
    blocked: false,
    updatedTs: Date.now(),
  };
  await persist();
  console.log(`[V7] ✅ MANUAL RESUME for ${symbol} (baseline reset @ ${closedNow} closed trades, perf gate reset)`);
  return { resumed: true };
}

/** Compute live divergence stats vs back-test (last N V7 closed trades). */
export async function getV7Performance(lookback: number = 200) {
  const closed = await storage.getPositions("CLOSED", 1000);
  const v7 = closed
    .filter(p => p.source === V7_SOURCE_TRADEABLE || p.source === V7_SOURCE_PROBATIONARY)
    .filter(p => p.realizedPnlUsdt != null && p.exitPrice != null && p.entryPrice > 0)
    .slice(-lookback);

  function bookFor(syms: readonly string[]) {
    const trades = v7.filter(p => syms.includes(p.symbol));
    if (trades.length === 0) return { n: 0 };
    const grossBps = trades.map(p => {
      const dir = p.side === "LONG" ? 1 : -1;
      return Math.log((p.exitPrice as number) / p.entryPrice) * dir * 1e4;
    });
    const meanGross = grossBps.reduce((a, b) => a + b, 0) / grossBps.length;
    const wr = grossBps.filter(x => x > 0).length / grossBps.length * 100;
    const at = (cost: number) => meanGross - cost;
    return {
      n: trades.length,
      gross_mean_bps: meanGross,
      wr_pct: wr,
      net_mean_bps_4: at(4),
      net_mean_bps_6: at(6),
      net_mean_bps_8: at(8),
    };
  }

  // Back-test reference (V7 precision P1 offline run):
  // tradeable book mean gross +66.46 bps, net@6 +60.46 bps, net@8 +58.46 bps
  const tradeableLive = bookFor(V7_TRADEABLE);
  const probationaryLive = bookFor(V7_PROBATIONARY);
  const refTradeable = { gross_mean_bps: 66.46, net_mean_bps_6: 60.46 };
  const refProbationary = { gross_mean_bps: 0.0, net_mean_bps_6: 0.0 };

  function divergence(live: any, ref: any) {
    if (!live.n) return null;
    const drift = (live.net_mean_bps_6 ?? 0) - ref.net_mean_bps_6;
    let band = "WITHIN_BAND";
    if (drift < -5) band = "BELOW_BAND";
    else if (drift > 5) band = "ABOVE_BAND";
    return { drift_bps: drift, band, ref_net_bps_6: ref.net_mean_bps_6 };
  }

  return {
    tradeable: { live: tradeableLive, ref: refTradeable, divergence: divergence(tradeableLive, refTradeable) },
    probationary: { live: probationaryLive, ref: refProbationary, divergence: divergence(probationaryLive, refProbationary) },
    note: "Net@6bps assumes 6 bps round-trip cost; live cost should be verified separately.",
  };
}
