import * as bitget from "./client";
import { computeSignalLeverage } from "../paper/engine";
import { db } from "../db";
import { settings } from "@shared/schema";
import { eq } from "drizzle-orm";
import { TRADING_SYMBOLS, QTY_PRECISION, PRICE_PRECISION } from "@shared/symbols";

const SUPPORTED_SYMBOLS: readonly string[] = TRADING_SYMBOLS;
const MAX_OPEN_POSITIONS = 6;

interface BitgetLiveConfig {
  enabled: boolean;
  riskPerTradePct: number;
  maxDailyLossUsdt: number;
  dailyLossUsdt: number;
  lastResetDate: string;
}

let liveConfig: BitgetLiveConfig = {
  enabled: false,
  riskPerTradePct: 0.5,
  maxDailyLossUsdt: 500,
  dailyLossUsdt: 0,
  lastResetDate: new Date().toISOString().split("T")[0],
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

export async function updateBitgetLiveConfig(updates: Partial<Pick<BitgetLiveConfig, "riskPerTradePct" | "maxDailyLossUsdt">>): Promise<void> {
  if (updates.riskPerTradePct !== undefined) liveConfig.riskPerTradePct = updates.riskPerTradePct;
  if (updates.maxDailyLossUsdt !== undefined) liveConfig.maxDailyLossUsdt = updates.maxDailyLossUsdt;
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
        riskPerTradePct: saved.riskPerTradePct ?? 0.5,
        maxDailyLossUsdt: saved.maxDailyLossUsdt ?? 500,
        dailyLossUsdt: saved.dailyLossUsdt ?? 0,
        lastResetDate: saved.lastResetDate ?? new Date().toISOString().split("T")[0],
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
}): Promise<{ success: boolean; orderId?: string; qty?: string; leverage?: number; error?: string }> {
  const { symbol, side, entryPrice, stopLoss, takeProfit, v5Score } = params;

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

    const leverage = computeSignalLeverage(v5Score);
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
