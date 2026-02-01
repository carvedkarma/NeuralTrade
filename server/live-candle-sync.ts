import { db } from "./db";
import { candles } from "./db/schema";
import { eq, and, desc } from "drizzle-orm";
import { getBTCCandlesBinanceVision, getBTCPriceBinanceVision } from "./binance-vision";

const FIFTEEN_MIN_MS = 15 * 60 * 1000;
const SYNC_INTERVAL_MS = 60000; // Check every minute

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

export async function syncLatest15mCandles(): Promise<{
  success: boolean;
  candlesInserted: number;
  latestCandleTs: number | null;
  message: string;
}> {
  console.log("[Live Sync] Starting 15m candle sync via Binance Vision...");
  
  try {
    // Get current price
    const currentPrice = await getBTCPriceBinanceVision();
    syncStatus.currentPrice = currentPrice;
    
    // Get last candle from database
    const lastCandle = await db.select()
      .from(candles)
      .where(and(
        eq(candles.symbol, "BTCUSDT"),
        eq(candles.timeframe, "15m")
      ))
      .orderBy(desc(candles.timestamp))
      .limit(1);
    
    const lastCandleTs = lastCandle[0]?.timestamp || 0;
    
    // Calculate how many candles we need
    const now = Date.now();
    const expectedLastCandleTs = Math.floor(now / FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS - FIFTEEN_MIN_MS;
    const gapMs = expectedLastCandleTs - lastCandleTs;
    const gapBars = Math.floor(gapMs / FIFTEEN_MIN_MS);
    
    // Update freshness status
    const staleDurationMinutes = gapBars * 15;
    syncStatus.staleDurationMinutes = staleDurationMinutes;
    if (staleDurationMinutes <= 15) {
      syncStatus.dataFreshness = "fresh";
    } else if (staleDurationMinutes <= 60) {
      syncStatus.dataFreshness = "stale";
    } else {
      syncStatus.dataFreshness = "critical";
    }
    
    if (gapBars <= 0) {
      const msg = "Data is fresh, no sync needed";
      console.log(`[Live Sync] ${msg}`);
      syncStatus.lastSyncTs = Date.now();
      syncStatus.lastSyncResult = msg;
      syncStatus.lastCandleTs = lastCandleTs;
      return { success: true, candlesInserted: 0, latestCandleTs: lastCandleTs, message: msg };
    }
    
    console.log(`[Live Sync] Gap detected: ${gapBars} bars (${staleDurationMinutes} minutes), fetching...`);
    
    // Fetch candles from Binance Vision
    const fetchLimit = Math.min(gapBars + 10, 500);
    const newCandles = await getBTCCandlesBinanceVision("15m", fetchLimit);
    
    if (newCandles.length === 0) {
      const msg = "Binance Vision returned no candles";
      console.error(`[Live Sync] ${msg}`);
      syncStatus.lastSyncResult = msg;
      return { success: false, candlesInserted: 0, latestCandleTs: lastCandleTs, message: msg };
    }
    
    console.log(`[Live Sync] Fetched ${newCandles.length} candles from Binance Vision`);
    
    // Insert new candles
    let inserted = 0;
    for (const candle of newCandles) {
      if (candle.timestamp > lastCandleTs) {
        try {
          await db.insert(candles).values({
            symbol: "BTCUSDT",
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
          // Ignore duplicate key errors
        }
      }
    }
    
    // Get updated last candle
    const updatedLastCandle = await db.select()
      .from(candles)
      .where(and(
        eq(candles.symbol, "BTCUSDT"),
        eq(candles.timeframe, "15m")
      ))
      .orderBy(desc(candles.timestamp))
      .limit(1);
    
    const latestTs = updatedLastCandle[0]?.timestamp || null;
    const msg = inserted > 0 
      ? `Synced ${inserted} new 15m candles` 
      : "All candles already up to date";
    
    console.log(`[Live Sync] ${msg}`);
    
    syncStatus.lastSyncTs = Date.now();
    syncStatus.lastSyncResult = msg;
    syncStatus.candlesSynced += inserted;
    syncStatus.lastCandleTs = latestTs;
    syncStatus.dataFreshness = "fresh";
    syncStatus.staleDurationMinutes = 0;
    
    return { success: true, candlesInserted: inserted, latestCandleTs: latestTs, message: msg };
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
  
  // Run immediately
  syncLatest15mCandles();
  
  // Then run every minute
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
