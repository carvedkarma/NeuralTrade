import { db } from "./db";
import { candles } from "./db/schema";
import { eq, and, desc } from "drizzle-orm";
import { getCandlesBinanceVision, getBTCPriceBinanceVision } from "./binance-vision";
import { TRADING_SYMBOLS } from "@shared/symbols";

const FIFTEEN_MIN_MS = 15 * 60 * 1000;
const SYNC_INTERVAL_MS = 60000;

const SYNC_SYMBOLS = [...TRADING_SYMBOLS];

interface SyncStatus {
  isRunning: boolean;
  lastSyncTs: number | null;
  lastSyncResult: string | null;
  candlesSynced: number;
  lastCandleTs: number | null;
  currentPrice: number | null;
  dataFreshness: "fresh" | "stale" | "critical" | "unknown";
  staleDurationMinutes: number;
}

let syncInterval: NodeJS.Timeout | null = null;
let syncStatus: SyncStatus = {
  isRunning: false,
  lastSyncTs: null,
  lastSyncResult: null,
  candlesSynced: 0,
  lastCandleTs: null,
  currentPrice: null,
  dataFreshness: "unknown",
  staleDurationMinutes: 0,
};

export function getSyncStatus(): SyncStatus {
  return { ...syncStatus };
}

export async function syncSymbolCandles(symbol: string): Promise<number> {
  const lastCandle = await db.select()
    .from(candles)
    .where(and(
      eq(candles.symbol, symbol),
      eq(candles.timeframe, "15m")
    ))
    .orderBy(desc(candles.timestamp))
    .limit(1);

  const lastCandleTs = lastCandle[0]?.timestamp || 0;
  const now = Date.now();
  const expectedLastCandleTs = Math.floor(now / FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS - FIFTEEN_MIN_MS;
  const gapMs = expectedLastCandleTs - lastCandleTs;
  const gapBars = Math.floor(gapMs / FIFTEEN_MIN_MS);

  if (gapBars <= 0) {
    return 0;
  }

  const fetchLimit = Math.min(gapBars + 10, 500);
  const newCandles = await getCandlesBinanceVision(symbol, "15m", fetchLimit);

  if (newCandles.length === 0) {
    return 0;
  }

  let inserted = 0;
  for (const candle of newCandles) {
    if (candle.timestamp > lastCandleTs) {
      try {
        await db.insert(candles).values({
          symbol,
          timeframe: "15m",
          timestamp: candle.timestamp,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          volume: candle.volume,
        }).onConflictDoNothing();
        inserted++;
      } catch (e) {
      }
    }
  }

  return inserted;
}

export async function syncLatest15mCandles(): Promise<{
  success: boolean;
  candlesInserted: number;
  latestCandleTs: number | null;
  message: string;
}> {
  console.log("[Live Sync] Starting 15m candle sync via Binance Vision...");
  
  try {
    const currentPrice = await getBTCPriceBinanceVision();
    syncStatus.currentPrice = currentPrice;
    
    let totalInserted = 0;
    const perSymbol: string[] = [];
    
    for (const symbol of SYNC_SYMBOLS) {
      try {
        const inserted = await syncSymbolCandles(symbol);
        totalInserted += inserted;
        if (inserted > 0) {
          perSymbol.push(`${symbol}:${inserted}`);
        }
      } catch (err) {
        console.error(`[Live Sync] Error syncing ${symbol}:`, err);
      }
    }
    
    const lastCandle = await db.select()
      .from(candles)
      .where(and(
        eq(candles.symbol, "BTCUSDT"),
        eq(candles.timeframe, "15m")
      ))
      .orderBy(desc(candles.timestamp))
      .limit(1);
    
    const latestTs = lastCandle[0]?.timestamp || null;
    
    const now = Date.now();
    const expectedLastCandleTs = Math.floor(now / FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS - FIFTEEN_MIN_MS;
    const gapMs = latestTs ? expectedLastCandleTs - latestTs : Infinity;
    const staleDurationMinutes = Math.floor(gapMs / (60 * 1000));
    syncStatus.staleDurationMinutes = staleDurationMinutes;
    if (staleDurationMinutes <= 15) {
      syncStatus.dataFreshness = "fresh";
    } else if (staleDurationMinutes <= 60) {
      syncStatus.dataFreshness = "stale";
    } else {
      syncStatus.dataFreshness = "critical";
    }
    
    const msg = totalInserted > 0 
      ? `Synced ${totalInserted} new 15m candles (${perSymbol.join(", ")})` 
      : "Data is fresh, no sync needed";
    
    console.log(`[Live Sync] ${msg}`);
    
    syncStatus.lastSyncTs = Date.now();
    syncStatus.lastSyncResult = msg;
    syncStatus.candlesSynced += totalInserted;
    syncStatus.lastCandleTs = latestTs;
    
    return { success: true, candlesInserted: totalInserted, latestCandleTs: latestTs, message: msg };
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error(`[Live Sync] Error: ${msg}`);
    syncStatus.lastSyncResult = `Error: ${msg}`;
    return { success: false, candlesInserted: 0, latestCandleTs: null, message: msg };
  }
}

export function startLiveCandleSync(): void {
  if (syncInterval) {
    console.log("[Live Sync] Already running");
    return;
  }
  
  console.log("[Live Sync] Starting continuous 15m candle sync service (every 60s)");
  syncStatus.isRunning = true;
  
  syncLatest15mCandles();
  
  syncInterval = setInterval(syncLatest15mCandles, SYNC_INTERVAL_MS);
}

export function stopLiveCandleSync(): void {
  if (syncInterval) {
    clearInterval(syncInterval);
    syncInterval = null;
    syncStatus.isRunning = false;
    console.log("[Live Sync] Stopped continuous sync");
  }
}

export async function checkDataFreshness(): Promise<{
  isFresh: boolean;
  lastCandleTs: number | null;
  lastCandleAge: number;
  currentPrice: number | null;
  message: string;
}> {
  const lastCandle = await db.select()
    .from(candles)
    .where(and(
      eq(candles.symbol, "BTCUSDT"),
      eq(candles.timeframe, "15m")
    ))
    .orderBy(desc(candles.timestamp))
    .limit(1);
  
  const lastCandleTs = lastCandle[0]?.timestamp || null;
  const now = Date.now();
  const expectedLastCandleTs = Math.floor(now / FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS - FIFTEEN_MIN_MS;
  const ageMinutes = lastCandleTs 
    ? Math.floor((expectedLastCandleTs - lastCandleTs) / (60 * 1000))
    : Infinity;
  
  const currentPrice = await getBTCPriceBinanceVision();
  
  const isFresh = ageMinutes <= 15;
  const message = isFresh 
    ? "Data is fresh" 
    : `Data is ${ageMinutes} minutes behind`;
  
  return {
    isFresh,
    lastCandleTs,
    lastCandleAge: ageMinutes,
    currentPrice,
    message,
  };
}
