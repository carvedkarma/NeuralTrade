import * as bitget from "./client";
import { computeSignalLeverage } from "../paper/engine";
import { db } from "../db";
import { settings, candles as candlesTable } from "@shared/schema";
import { eq, and, desc } from "drizzle-orm";
import { TRADING_SYMBOLS, QTY_PRECISION, PRICE_PRECISION } from "@shared/symbols";

const SUPPORTED_SYMBOLS: readonly string[] = TRADING_SYMBOLS;
const MAX_OPEN_POSITIONS = 6;

interface BitgetLiveConfig {
  enabled: boolean;
  riskPerTradePct: number;
  maxDailyLossUsdt: number;
  dailyLossUsdt: number;
  lastResetDate: string;
  trailActivation: number;  // Activate trail when profit >= N × ATR (default 1.0)
  trailDistance: number;    // Trail sits N × ATR behind best price (default 0.8)
}

let liveConfig: BitgetLiveConfig = {
  enabled: false,
  riskPerTradePct: 0.3,   // Base margin pct — multiplied by exchange leverage (e.g. 25x × 0.3% = 7.5% effective risk)
  maxDailyLossUsdt: 500,
  dailyLossUsdt: 0,
  lastResetDate: new Date().toISOString().split("T")[0],
  trailActivation: 1.0,
  trailDistance: 0.8,
};

export function isBitgetLiveTradingEnabled(): boolean {
  return liveConfig.enabled && bitget.isConfigured();
}

export function getBitgetLiveConfig(): BitgetLiveConfig {
  return { ...liveConfig };
}

export async function setBitgetLiveTradingEnabled(enabled: boolean): Promise<void> {
  if (enabled && !bitget.isConfigured()) {
    throw new Error("Cannot enable live trading: Bitget API credentials not configured");
  }
  if (enabled) {
    const test = await bitget.testConnection();
    if (!test.success) {
      throw new Error(`Cannot enable live trading: ${test.error}`);
    }
  }
  liveConfig.enabled = enabled;
  await saveBitgetLiveConfig();
  console.log(`[Bitget Live] ${enabled ? "ENABLED" : "DISABLED"}`);
}

export async function updateBitgetLiveConfig(updates: Partial<Pick<BitgetLiveConfig, "riskPerTradePct" | "maxDailyLossUsdt" | "trailActivation" | "trailDistance">>): Promise<void> {
  if (updates.riskPerTradePct !== undefined) liveConfig.riskPerTradePct = updates.riskPerTradePct;
  if (updates.maxDailyLossUsdt !== undefined) liveConfig.maxDailyLossUsdt = updates.maxDailyLossUsdt;
  if (updates.trailActivation !== undefined) liveConfig.trailActivation = updates.trailActivation;
  if (updates.trailDistance !== undefined) liveConfig.trailDistance = updates.trailDistance;
  await saveBitgetLiveConfig();
}

async function saveBitgetLiveConfig(): Promise<void> {
  try {
    const existing = await db.select().from(settings).where(eq(settings.key, "bitget_live_config")).limit(1);
    const configToSave = { ...liveConfig };
    if (existing.length > 0) {
      await db.update(settings).set({ valueJson: configToSave, updatedAt: BigInt(Date.now()) }).where(eq(settings.key, "bitget_live_config"));
    } else {
      await db.insert(settings).values({ key: "bitget_live_config", valueJson: configToSave, updatedAt: BigInt(Date.now()) });
    }
  } catch (err: any) {
    console.error("[Bitget Live] Failed to save config:", err.message);
  }
}

export async function loadBitgetLiveConfig(): Promise<void> {
  try {
    await bitget.loadCredentials();
    const rows = await db.select().from(settings).where(eq(settings.key, "bitget_live_config")).limit(1);
    if (rows.length > 0 && rows[0].valueJson) {
      const saved = rows[0].valueJson as any;
      liveConfig = {
        enabled: saved.enabled ?? false,
        riskPerTradePct: saved.riskPerTradePct ?? 0.3,
        maxDailyLossUsdt: saved.maxDailyLossUsdt ?? 500,
        dailyLossUsdt: saved.dailyLossUsdt ?? 0,
        lastResetDate: saved.lastResetDate ?? new Date().toISOString().split("T")[0],
        trailActivation: saved.trailActivation ?? 1.0,
        trailDistance: saved.trailDistance ?? 0.8,
      };
      if (!bitget.isConfigured()) {
        liveConfig.enabled = false;
      }
    }
  } catch (err: any) {
    console.error("[Bitget Live] Failed to load config:", err.message);
  }
}

function resetDailyLossIfNeeded(): void {
  const today = new Date().toISOString().split("T")[0];
  if (liveConfig.lastResetDate !== today) {
    liveConfig.dailyLossUsdt = 0;
    liveConfig.lastResetDate = today;
  }
}

function formatQty(symbol: string, qty: number): string {
  const precision = QTY_PRECISION[symbol] ?? 3;
  return qty.toFixed(precision);
}

function formatPrice(symbol: string, price: number): string {
  const precision = PRICE_PRECISION[symbol] ?? 2;
  return price.toFixed(precision);
}

export async function openBitgetLivePosition(params: {
  symbol: string;
  side: "LONG" | "SHORT";
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  v5Score: number;
  signalConfidence?: number;
  chopLeverageMult?: number;
}): Promise<{ success: boolean; orderId?: string; qty?: string; leverage?: number; error?: string }> {
  const { symbol, side, entryPrice, stopLoss, takeProfit, v5Score, chopLeverageMult } = params;

  if (!isBitgetLiveTradingEnabled()) {
    return { success: false, error: "Bitget live trading is not enabled" };
  }

  if (!SUPPORTED_SYMBOLS.includes(symbol)) {
    return { success: false, error: `Unsupported symbol: ${symbol}` };
  }

  resetDailyLossIfNeeded();
  if (liveConfig.dailyLossUsdt >= liveConfig.maxDailyLossUsdt) {
    return { success: false, error: `Daily loss limit reached: $${liveConfig.dailyLossUsdt.toFixed(2)} / $${liveConfig.maxDailyLossUsdt}` };
  }

  try {
    const positionsResp = await bitget.getAllPositions();
    if (positionsResp.code !== "00000") {
      return { success: false, error: `Failed to check positions: ${positionsResp.msg}` };
    }

    const existingPos = (positionsResp.data || []).filter(
      (p: any) => p.symbol === symbol && parseFloat(p.total || "0") > 0
    );
    if (existingPos.length > 0) {
      return { success: false, error: `Already have an open position for ${symbol}` };
    }

    const allOpen = (positionsResp.data || []).filter(
      (p: any) => parseFloat(p.total || "0") > 0
    );
    if (allOpen.length >= MAX_OPEN_POSITIONS) {
      return { success: false, error: `Max ${MAX_OPEN_POSITIONS} open positions reached (current: ${allOpen.length})` };
    }

    const balanceResp = await bitget.getAccountBalance();
    if (balanceResp.code !== "00000") {
      return { success: false, error: `Failed to get balance: ${balanceResp.msg}` };
    }
    const usdtAccount = (balanceResp.data || []).find((a: any) => a.marginCoin === "USDT");
    const equity = parseFloat(usdtAccount?.accountEquity || usdtAccount?.available || "0");
    if (equity <= 0) {
      return { success: false, error: "No USDT equity available" };
    }

    let leverage = computeSignalLeverage(v5Score);
    if (chopLeverageMult != null && chopLeverageMult < 1) {
      leverage = Math.max(1, Math.round(leverage * chopLeverageMult));
      console.log(`[Bitget Live] Chop throttle applied: leverage reduced to ${leverage}x (mult=${chopLeverageMult})`);
    }
    const stopDistance = Math.abs(entryPrice - stopLoss);
    if (stopDistance <= 0 || !Number.isFinite(stopDistance)) {
      return { success: false, error: `Invalid stop distance: ${stopDistance} (entry=${entryPrice}, SL=${stopLoss})` };
    }
    const riskUsd = equity * (liveConfig.riskPerTradePct / 100);
    const baseQty = riskUsd / stopDistance;
    const finalQty = baseQty * leverage;
    if (!Number.isFinite(finalQty) || finalQty <= 0) {
      return { success: false, error: `Invalid qty calculation: ${finalQty}` };
    }

    const holdSide = side === "LONG" ? "long" : "short";
    try {
      await bitget.setLeverage(symbol, String(leverage), holdSide);
    } catch (leverageErr: any) {
      console.log(`[Bitget Live] Leverage set note for ${symbol}: ${leverageErr.message}`);
    }

    const bitgetSide = side === "LONG" ? "buy" : "sell";
    const qtyStr = formatQty(symbol, finalQty);

    console.log(`[Bitget Live] Opening ${side} ${symbol}: qty=${qtyStr}, leverage=${leverage}x, risk=$${riskUsd.toFixed(2)}, SL=${formatPrice(symbol, stopLoss)}, TP=${formatPrice(symbol, takeProfit)}, v5Score=${v5Score}`);

    const orderResp = await bitget.placeOrder({
      symbol,
      side: bitgetSide as "buy" | "sell",
      tradeSide: "open",
      orderType: "market",
      size: qtyStr,
      presetStopLossPrice: formatPrice(symbol, stopLoss),
      presetStopSurplusPrice: formatPrice(symbol, takeProfit),
    });

    if (orderResp.code !== "00000") {
      console.error(`[Bitget Live] Order failed for ${symbol}: ${orderResp.msg}`);
      return { success: false, error: `Order failed: ${orderResp.msg}` };
    }

    console.log(`[Bitget Live] Order placed for ${symbol}: orderId=${orderResp.data?.orderId}`);

    const { broadcast } = await import("../ws");
    broadcast("LIVE_TRADE_OPEN", {
      symbol,
      side,
      qty: qtyStr,
      leverage,
      orderId: orderResp.data?.orderId,
      v5Score,
      exchange: "bitget",
    });

    return {
      success: true,
      orderId: orderResp.data?.orderId,
      qty: qtyStr,
      leverage,
    };
  } catch (err: any) {
    console.error(`[Bitget Live] Error opening ${symbol}:`, err.message);
    return { success: false, error: err.message };
  }
}

export async function closeBitgetLivePosition(symbol: string): Promise<{ success: boolean; orderId?: string; error?: string }> {
  if (!bitget.isConfigured()) {
    return { success: false, error: "Bitget not configured" };
  }

  try {
    const posResp = await bitget.getAllPositions();
    if (posResp.code !== "00000") {
      return { success: false, error: `Failed to get positions: ${posResp.msg}` };
    }

    const position = (posResp.data || []).find(
      (p: any) => p.symbol === symbol && parseFloat(p.total || "0") > 0
    );
    if (!position) {
      return { success: false, error: `No open position for ${symbol}` };
    }

    const holdSide = position.holdSide === "long" ? "long" : "short";

    console.log(`[Bitget Live] Closing ${symbol}: holdSide=${holdSide}`);

    const closeResp = await bitget.closePosition(symbol, holdSide as "long" | "short");

    if (closeResp.code !== "00000") {
      return { success: false, error: `Close failed: ${closeResp.msg}` };
    }

    const upl = parseFloat(position.unrealizedPL || "0");
    if (upl < 0) {
      liveConfig.dailyLossUsdt += Math.abs(upl);
      await saveBitgetLiveConfig();
    }

    const { broadcast } = await import("../ws");
    broadcast("LIVE_TRADE_CLOSE", {
      symbol,
      pnl: upl,
      exchange: "bitget",
    });

    return { success: true };
  } catch (err: any) {
    console.error(`[Bitget Live] Error closing ${symbol}:`, err.message);
    return { success: false, error: err.message };
  }
}

export async function getBitgetLivePositions(): Promise<{
  positions: Array<{
    symbol: string;
    side: string;
    size: string;
    avgPrice: string;
    markPrice: string;
    unrealisedPnl: string;
    leverage: string;
    stopLoss: string;
    takeProfit: string;
    liqPrice: string;
    positionValue: string;
    marginCoin: string;
  }>;
  error?: string;
}> {
  if (!bitget.isConfigured()) {
    return { positions: [], error: "Bitget not configured" };
  }

  try {
    const resp = await bitget.getAllPositions();
    if (resp.code !== "00000") {
      return { positions: [], error: resp.msg };
    }

    const openPositions = (resp.data || [])
      .filter((p: any) => parseFloat(p.total || "0") > 0)
      .map((p: any) => ({
        symbol: p.symbol || "",
        side: p.holdSide === "long" ? "LONG" : "SHORT",
        size: p.total || "0",
        avgPrice: p.openPriceAvg || "0",
        markPrice: p.markPrice || "0",
        unrealisedPnl: p.unrealizedPL || "0",
        leverage: p.leverage || "1",
        stopLoss: p.stopLossPrice || "0",
        takeProfit: p.stopSurplusPrice || "0",
        liqPrice: p.liquidationPrice || "0",
        positionValue: p.marginSize || "0",
        marginCoin: p.marginCoin || "USDT",
      }));

    return { positions: openPositions };
  } catch (err: any) {
    return { positions: [], error: err.message };
  }
}

export async function getBitgetLiveBalance(): Promise<{
  equity: string;
  walletBalance: string;
  availableBalance: string;
  unrealisedPnl: string;
  error?: string;
}> {
  if (!bitget.isConfigured()) {
    return { equity: "0", walletBalance: "0", availableBalance: "0", unrealisedPnl: "0", error: "Bitget not configured" };
  }

  try {
    const resp = await bitget.getAccountBalance();
    if (resp.code !== "00000") {
      return { equity: "0", walletBalance: "0", availableBalance: "0", unrealisedPnl: "0", error: resp.msg };
    }

    const usdtAccount = (resp.data || []).find((a: any) => a.marginCoin === "USDT");

    return {
      equity: usdtAccount?.accountEquity || "0",
      walletBalance: usdtAccount?.crossedMaxAvailable || usdtAccount?.available || "0",
      availableBalance: usdtAccount?.available || "0",
      unrealisedPnl: usdtAccount?.unrealizedPL || "0",
    };
  } catch (err: any) {
    return { equity: "0", walletBalance: "0", availableBalance: "0", unrealisedPnl: "0", error: err.message };
  }
}

// ─── ATR-Distance Trailing Stop Monitor ──────────────────────────────────────

interface BitgetTrailState {
  entryPrice: number;
  stopDistance: number;  // Initial SL distance in price units (for floor calc)
  side: "LONG" | "SHORT";
  bestPrice: number;
  trailActive: boolean;
  trailPrice: number | null;
  isRunner: boolean;
}

// In-memory trail state per symbol (survives position lifecycle within one server run)
const trailStates = new Map<string, BitgetTrailState>();

// Trail configuration — static constants for non-configurable params
const TRAIL_MIN_PROFIT_R = 0.15; // Lock in at least 15% of stopDistance
const TRAIL_ALLOW_RUNNER = true; // Tighten to 0.4× after TP is passed
// trailActivation and trailDistance come from liveConfig (DB-persisted)

async function computeAtr(symbol: string, periods: number = 14): Promise<number | null> {
  try {
    const rows = await db
      .select({ high: candlesTable.high, low: candlesTable.low, close: candlesTable.close })
      .from(candlesTable)
      .where(and(eq(candlesTable.symbol, symbol), eq(candlesTable.timeframe, "15m")))
      .orderBy(desc(candlesTable.timestamp))
      .limit(periods + 1);

    if (rows.length < 2) return null;
    const reversed = rows.reverse();
    let atrSum = 0;
    let count = 0;
    for (let i = 1; i < reversed.length; i++) {
      const prev = reversed[i - 1];
      const curr = reversed[i];
      const tr = Math.max(
        curr.high - curr.low,
        Math.abs(curr.high - prev.close),
        Math.abs(curr.low - prev.close)
      );
      atrSum += tr;
      count++;
    }
    return count > 0 ? atrSum / count : null;
  } catch {
    return null;
  }
}

export async function updateBitgetTrailingStops(): Promise<void> {
  if (!bitget.isConfigured()) return;

  try {
    const posResp = await bitget.getAllPositions();
    if (posResp.code !== "00000") return;

    const openPositions = (posResp.data || []).filter(
      (p: any) => parseFloat(p.total || "0") > 0
    );

    // Clear stale trail states for symbols no longer open
    const openSymbols = new Set(openPositions.map((p: any) => p.symbol as string));
    for (const sym of trailStates.keys()) {
      if (!openSymbols.has(sym)) trailStates.delete(sym);
    }

    for (const pos of openPositions) {
      const symbol: string = pos.symbol;
      const side: "LONG" | "SHORT" = pos.holdSide === "long" ? "LONG" : "SHORT";
      const entryPrice = parseFloat(pos.openPriceAvg || "0");
      const markPrice  = parseFloat(pos.markPrice || "0");
      const slPrice    = parseFloat(pos.stopLossPrice || "0");
      const tpPrice    = parseFloat(pos.stopSurplusPrice || "0");

      if (!entryPrice || !markPrice) continue;

      // Compute stop distance from original SL (or fall back to ATR-based estimate)
      const originalStopDist = slPrice > 0 ? Math.abs(entryPrice - slPrice) : 0;

      // Initialize trail state on first encounter
      if (!trailStates.has(symbol)) {
        trailStates.set(symbol, {
          entryPrice,
          stopDistance: originalStopDist,
          side,
          bestPrice: markPrice,
          trailActive: false,
          trailPrice: null,
          isRunner: false,
        });
      }

      const state = trailStates.get(symbol)!;
      // Sync side/entry in case position was replaced
      state.side = side;
      state.entryPrice = entryPrice;

      // Update best price
      if (side === "LONG") {
        state.bestPrice = Math.max(state.bestPrice, markPrice);
      } else {
        state.bestPrice = Math.min(state.bestPrice, markPrice);
      }

      // Get ATR for this symbol
      const atr = await computeAtr(symbol);
      if (!atr || atr <= 0) continue;

      // Use stopDistance for floor; fall back to ATR×1.2 if not set
      const stopDist = state.stopDistance > 0 ? state.stopDistance : atr * 1.2;
      const minProfitDist = TRAIL_MIN_PROFIT_R * stopDist;
      const activationDist = liveConfig.trailActivation * atr;

      // Check activation
      const peakPriceMove = Math.abs(state.bestPrice - entryPrice);
      if (peakPriceMove < activationDist) continue;

      // Runner mode: best price has passed TP
      const isRunner = TRAIL_ALLOW_RUNNER && tpPrice > 0 && (
        side === "LONG" ? state.bestPrice >= tpPrice : state.bestPrice <= tpPrice
      );
      state.isRunner = isRunner;

      // Compute trail distance (uses DB-backed liveConfig)
      const trailDist = isRunner ? liveConfig.trailDistance * 0.4 * atr : liveConfig.trailDistance * atr;

      // Compute candidate trail stop
      let candidateTrail: number;
      if (side === "LONG") {
        candidateTrail = Math.max(entryPrice + minProfitDist, state.bestPrice - trailDist);
      } else {
        candidateTrail = Math.min(entryPrice - minProfitDist, state.bestPrice + trailDist);
      }

      // Ratchet
      let finalTrail = candidateTrail;
      if (state.trailPrice !== null) {
        if (side === "LONG") {
          finalTrail = Math.max(candidateTrail, state.trailPrice);
        } else {
          finalTrail = Math.min(candidateTrail, state.trailPrice);
        }
      }

      const wasActive = state.trailActive;
      const prevTrailPrice = state.trailPrice;
      state.trailActive = true;
      state.trailPrice = finalTrail;

      const trailMoved = !wasActive || prevTrailPrice === null || Math.abs(finalTrail - prevTrailPrice) > 0.000001;

      if (!wasActive) {
        console.log(`[Bitget Trail] ACTIVATED ${side} ${symbol}: trailStop=${formatPrice(symbol, finalTrail)}, bestPrice=${formatPrice(symbol, state.bestPrice)}, ATR=${atr.toFixed(4)}`);
      }

      // ── Update exchange stop-loss order when trail advances ──────────────────
      if (trailMoved) {
        try {
          const holdSide = side === "LONG" ? "long" : "short";
          const slResp = await bitget.setPositionTpSl({
            symbol,
            holdSide,
            stopLossPrice: formatPrice(symbol, finalTrail),
          });
          if (slResp.code !== "00000") {
            console.warn(`[Bitget Trail] SL update failed for ${symbol}: ${slResp.msg} — will market-close if trail hit`);
          } else if (!wasActive) {
            console.log(`[Bitget Trail] Exchange SL set to ${formatPrice(symbol, finalTrail)} for ${symbol}`);
          }
        } catch (slErr: any) {
          console.warn(`[Bitget Trail] SL API error for ${symbol}: ${slErr.message}`);
        }
      }

      // Check if current mark price has crossed the trail stop (app-level safety net)
      const trailHit = side === "LONG"
        ? markPrice <= finalTrail
        : markPrice >= finalTrail;

      if (trailHit) {
        const profitPct = side === "LONG"
          ? ((markPrice - entryPrice) / entryPrice * 100).toFixed(2)
          : ((entryPrice - markPrice) / entryPrice * 100).toFixed(2);
        console.log(`[Bitget Trail] TRAIL HIT ${side} ${symbol}: markPrice=${formatPrice(symbol, markPrice)}, trailStop=${formatPrice(symbol, finalTrail)}, P&L pct=${profitPct}%`);
        trailStates.delete(symbol);
        await closeBitgetLivePosition(symbol);
      }
    }
  } catch (err: any) {
    console.error("[Bitget Trail] Monitor error:", err.message);
  }
}

export function getBitgetTrailStates(): Map<string, BitgetTrailState> {
  return trailStates;
}

let trailMonitorIntervalId: ReturnType<typeof setInterval> | null = null;

export function startBitgetTrailMonitor(): void {
  if (trailMonitorIntervalId) return;
  console.log("[Bitget Trail] Trail monitor started (5s interval)");
  trailMonitorIntervalId = setInterval(() => {
    if (isBitgetLiveTradingEnabled()) {
      updateBitgetTrailingStops().catch((err) =>
        console.error("[Bitget Trail] Monitor tick error:", err.message)
      );
    }
  }, 5_000);
}

export function stopBitgetTrailMonitor(): void {
  if (trailMonitorIntervalId) {
    clearInterval(trailMonitorIntervalId);
    trailMonitorIntervalId = null;
    console.log("[Bitget Trail] Trail monitor stopped");
  }
}

// ─────────────────────────────────────────────────────────────────────────────

export {
  isBitgetLiveTradingEnabled as isLiveTradingEnabled,
  openBitgetLivePosition as openLivePosition,
  closeBitgetLivePosition as closeLivePosition,
  getBitgetLivePositions as getLivePositions,
  getBitgetLiveBalance as getLiveBalance,
  getBitgetLiveConfig as getLiveConfig,
  setBitgetLiveTradingEnabled as setLiveTradingEnabled,
  updateBitgetLiveConfig as updateLiveConfig,
  loadBitgetLiveConfig as loadLiveConfig,
};
