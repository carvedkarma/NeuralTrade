import * as bybit from "./client";
import { computeSignalLeverage } from "../paper/engine";
import { db } from "../db";
import { settings } from "@shared/schema";
import { eq } from "drizzle-orm";
import { TRADING_SYMBOLS, QTY_PRECISION, PRICE_PRECISION } from "@shared/symbols";

const SUPPORTED_SYMBOLS: readonly string[] = TRADING_SYMBOLS;
const MAX_OPEN_POSITIONS = 6;

interface LiveTradingConfig {
  enabled: boolean;
  riskPerTradePct: number;
  maxDailyLossUsdt: number;
  dailyLossUsdt: number;
  lastResetDate: string;
}

let liveConfig: LiveTradingConfig = {
  enabled: false,
  riskPerTradePct: 0.5,
  maxDailyLossUsdt: 500,
  dailyLossUsdt: 0,
  lastResetDate: new Date().toISOString().split("T")[0],
};

export function isLiveTradingEnabled(): boolean {
  return liveConfig.enabled && bybit.isConfigured();
}

export function getLiveConfig(): LiveTradingConfig {
  return { ...liveConfig };
}

export async function setLiveTradingEnabled(enabled: boolean): Promise<void> {
  if (enabled && !bybit.isConfigured()) {
    throw new Error("Cannot enable live trading: Bybit API credentials not configured");
  }
  if (enabled) {
    const test = await bybit.testConnection();
    if (!test.success) {
      throw new Error(`Cannot enable live trading: ${test.error}`);
    }
  }
  liveConfig.enabled = enabled;
  await saveLiveConfig();
  console.log(`[Live Trading] ${enabled ? "ENABLED" : "DISABLED"}`);
}

export async function updateLiveConfig(updates: Partial<Pick<LiveTradingConfig, "riskPerTradePct" | "maxDailyLossUsdt">>): Promise<void> {
  if (updates.riskPerTradePct !== undefined) liveConfig.riskPerTradePct = updates.riskPerTradePct;
  if (updates.maxDailyLossUsdt !== undefined) liveConfig.maxDailyLossUsdt = updates.maxDailyLossUsdt;
  await saveLiveConfig();
}

async function saveLiveConfig(): Promise<void> {
  try {
    const existing = await db.select().from(settings).where(eq(settings.key, "live_trading_config")).limit(1);
    const configToSave = { ...liveConfig };
    if (existing.length > 0) {
      await db.update(settings).set({ valueJson: configToSave, updatedAt: BigInt(Date.now()) }).where(eq(settings.key, "live_trading_config"));
    } else {
      await db.insert(settings).values({ key: "live_trading_config", valueJson: configToSave, updatedAt: BigInt(Date.now()) });
    }
  } catch (err: any) {
    console.error("[Live Trading] Failed to save config:", err.message);
  }
}

export async function loadLiveConfig(): Promise<void> {
  try {
    const rows = await db.select().from(settings).where(eq(settings.key, "live_trading_config")).limit(1);
    if (rows.length > 0 && rows[0].valueJson) {
      const saved = rows[0].valueJson as any;
      liveConfig = {
        enabled: saved.enabled ?? false,
        riskPerTradePct: saved.riskPerTradePct ?? 0.5,
        maxDailyLossUsdt: saved.maxDailyLossUsdt ?? 500,
        dailyLossUsdt: saved.dailyLossUsdt ?? 0,
        lastResetDate: saved.lastResetDate ?? new Date().toISOString().split("T")[0],
      };
      if (!bybit.isConfigured()) {
        liveConfig.enabled = false;
      }
    }
  } catch (err: any) {
    console.error("[Live Trading] Failed to load config:", err.message);
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

export async function openLivePosition(params: {
  symbol: string;
  side: "LONG" | "SHORT";
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  v5Score: number;
  signalConfidence?: number;
}): Promise<{ success: boolean; orderId?: string; qty?: string; leverage?: number; error?: string }> {
  const { symbol, side, entryPrice, stopLoss, takeProfit, v5Score } = params;

  if (!isLiveTradingEnabled()) {
    return { success: false, error: "Live trading is not enabled" };
  }

  if (!SUPPORTED_SYMBOLS.includes(symbol)) {
    return { success: false, error: `Unsupported symbol: ${symbol}` };
  }

  resetDailyLossIfNeeded();
  if (liveConfig.dailyLossUsdt >= liveConfig.maxDailyLossUsdt) {
    return { success: false, error: `Daily loss limit reached: $${liveConfig.dailyLossUsdt.toFixed(2)} / $${liveConfig.maxDailyLossUsdt}` };
  }

  try {
    const positionsResp = await bybit.getPositions(symbol);
    if (positionsResp.retCode !== 0) {
      return { success: false, error: `Failed to check positions: ${positionsResp.retMsg}` };
    }
    const existingPos = positionsResp.result.list.filter(
      (p) => p.size !== "0" && p.size !== ""
    );
    if (existingPos.length > 0) {
      return { success: false, error: `Already have an open position for ${symbol}` };
    }

    const allPositionsResp = await bybit.getPositions();
    if (allPositionsResp.retCode === 0) {
      const openCount = allPositionsResp.result.list.filter(
        (p) => p.size !== "0" && p.size !== ""
      ).length;
      if (openCount >= MAX_OPEN_POSITIONS) {
        return { success: false, error: `Max ${MAX_OPEN_POSITIONS} open positions reached (current: ${openCount})` };
      }
    }

    const balanceResp = await bybit.getWalletBalance("USDT");
    if (balanceResp.retCode !== 0) {
      return { success: false, error: `Failed to get balance: ${balanceResp.retMsg}` };
    }
    const usdtCoin = balanceResp.result.list?.[0]?.coin?.find((c) => c.coin === "USDT");
    const equity = parseFloat(usdtCoin?.equity || "0");
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

    try {
      await bybit.setLeverage(symbol, String(leverage), String(leverage));
    } catch (leverageErr: any) {
      console.log(`[Live Trading] Leverage set note for ${symbol}: ${leverageErr.message}`);
    }

    const bybitSide = side === "LONG" ? "Buy" : "Sell";
    const qtyStr = formatQty(symbol, finalQty);

    console.log(`[Live Trading] Opening ${side} ${symbol}: qty=${qtyStr}, leverage=${leverage}x, risk=$${riskUsd.toFixed(2)}, SL=${formatPrice(symbol, stopLoss)}, TP=${formatPrice(symbol, takeProfit)}, v5Score=${v5Score}`);

    const orderResp = await bybit.createOrder({
      symbol,
      side: bybitSide,
      orderType: "Market",
      qty: qtyStr,
      stopLoss: formatPrice(symbol, stopLoss),
      takeProfit: formatPrice(symbol, takeProfit),
    });

    if (orderResp.retCode !== 0) {
      console.error(`[Live Trading] Order failed for ${symbol}: ${orderResp.retMsg}`);
      return { success: false, error: `Order failed: ${orderResp.retMsg}` };
    }

    console.log(`[Live Trading] Order placed for ${symbol}: orderId=${orderResp.result.orderId}`);

    const { broadcast } = await import("../ws");
    broadcast("LIVE_TRADE_OPEN", {
      symbol,
      side,
      qty: qtyStr,
      leverage,
      orderId: orderResp.result.orderId,
      v5Score,
    });

    return {
      success: true,
      orderId: orderResp.result.orderId,
      qty: qtyStr,
      leverage,
    };
  } catch (err: any) {
    console.error(`[Live Trading] Error opening ${symbol}:`, err.message);
    return { success: false, error: err.message };
  }
}

export async function closeLivePosition(symbol: string, qty?: string): Promise<{ success: boolean; orderId?: string; error?: string }> {
  if (!bybit.isConfigured()) {
    return { success: false, error: "Bybit not configured" };
  }

  try {
    const posResp = await bybit.getPositions(symbol);
    if (posResp.retCode !== 0) {
      return { success: false, error: `Failed to get position: ${posResp.retMsg}` };
    }

    const position = posResp.result.list.find(
      (p) => p.size !== "0" && p.size !== ""
    );
    if (!position) {
      return { success: false, error: `No open position for ${symbol}` };
    }

    const closeSide = position.side === "Buy" ? "Sell" : "Buy";
    const closeQty = qty || position.size;

    console.log(`[Live Trading] Closing ${symbol}: side=${closeSide}, qty=${closeQty}`);

    const orderResp = await bybit.createOrder({
      symbol,
      side: closeSide as "Buy" | "Sell",
      orderType: "Market",
      qty: closeQty,
      reduceOnly: true,
    });

    if (orderResp.retCode !== 0) {
      return { success: false, error: `Close order failed: ${orderResp.retMsg}` };
    }

    const upl = parseFloat(position.unrealisedPnl || "0");
    if (upl < 0) {
      liveConfig.dailyLossUsdt += Math.abs(upl);
      await saveLiveConfig();
    }

    const { broadcast } = await import("../ws");
    broadcast("LIVE_TRADE_CLOSE", {
      symbol,
      orderId: orderResp.result.orderId,
      pnl: upl,
    });

    return { success: true, orderId: orderResp.result.orderId };
  } catch (err: any) {
    console.error(`[Live Trading] Error closing ${symbol}:`, err.message);
    return { success: false, error: err.message };
  }
}

export async function amendLiveSLTP(symbol: string, params: {
  stopLoss?: number;
  takeProfit?: number;
}): Promise<{ success: boolean; error?: string }> {
  if (!bybit.isConfigured()) {
    return { success: false, error: "Bybit not configured" };
  }

  try {
    const updateParams: any = { symbol, positionIdx: 0 };
    if (params.stopLoss !== undefined) updateParams.stopLoss = formatPrice(symbol, params.stopLoss);
    if (params.takeProfit !== undefined) updateParams.takeProfit = formatPrice(symbol, params.takeProfit);

    console.log(`[Live Trading] Amending ${symbol}: SL=${params.stopLoss}, TP=${params.takeProfit}`);

    const resp = await bybit.setTradingStop(updateParams);
    if (resp.retCode !== 0) {
      return { success: false, error: `Amend failed: ${resp.retMsg}` };
    }

    return { success: true };
  } catch (err: any) {
    console.error(`[Live Trading] Error amending ${symbol}:`, err.message);
    return { success: false, error: err.message };
  }
}

export async function getLivePositions(): Promise<{
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
  }>;
  error?: string;
}> {
  if (!bybit.isConfigured()) {
    return { positions: [], error: "Bybit not configured" };
  }

  try {
    const resp = await bybit.getPositions();
    if (resp.retCode !== 0) {
      return { positions: [], error: resp.retMsg };
    }

    const openPositions = resp.result.list
      .filter((p) => p.size !== "0" && p.size !== "")
      .map((p) => ({
        symbol: p.symbol,
        side: p.side === "Buy" ? "LONG" : "SHORT",
        size: p.size,
        avgPrice: p.avgPrice,
        markPrice: p.markPrice,
        unrealisedPnl: p.unrealisedPnl,
        leverage: p.leverage,
        stopLoss: p.stopLoss,
        takeProfit: p.takeProfit,
        liqPrice: p.liqPrice,
        positionValue: p.positionValue,
      }));

    return { positions: openPositions };
  } catch (err: any) {
    return { positions: [], error: err.message };
  }
}

export async function getLiveBalance(): Promise<{
  equity: string;
  walletBalance: string;
  availableBalance: string;
  unrealisedPnl: string;
  error?: string;
}> {
  if (!bybit.isConfigured()) {
    return { equity: "0", walletBalance: "0", availableBalance: "0", unrealisedPnl: "0", error: "Bybit not configured" };
  }

  try {
    const resp = await bybit.getWalletBalance("USDT");
    if (resp.retCode !== 0) {
      return { equity: "0", walletBalance: "0", availableBalance: "0", unrealisedPnl: "0", error: resp.retMsg };
    }

    const account = resp.result.list?.[0];
    const usdtCoin = account?.coin?.find((c) => c.coin === "USDT");

    return {
      equity: usdtCoin?.equity || account?.totalEquity || "0",
      walletBalance: usdtCoin?.walletBalance || account?.totalWalletBalance || "0",
      availableBalance: usdtCoin?.availableToWithdraw || account?.totalAvailableBalance || "0",
      unrealisedPnl: usdtCoin?.unrealisedPnl || account?.totalPerpUPL || "0",
    };
  } catch (err: any) {
    return { equity: "0", walletBalance: "0", availableBalance: "0", unrealisedPnl: "0", error: err.message };
  }
}
