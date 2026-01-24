import { db } from "./db";
import { candles, learningState } from "./db/schema";
import { eq, and, gte, lte, sql, desc, asc } from "drizzle-orm";

const BINANCE_VISION_BASE = "https://data-api.binance.vision";
const BINANCE_FAPI_BASE = "https://fapi.binance.com";
const CANDLES_PER_REQUEST = 1000;
const MS_PER_15M = 15 * 60 * 1000;
const BACKFILL_DAYS = 370;

interface BinanceKline {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
}

async function fetchKlinesBatch(
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number
): Promise<BinanceKline[]> {
  const url = `${BINANCE_FAPI_BASE}/fapi/v1/klines?symbol=${symbol}&interval=${interval}&startTime=${startTime}&endTime=${endTime}&limit=${CANDLES_PER_REQUEST}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      if (response.status === 451) {
        console.log("[Historical] Binance futures API blocked, trying spot...");
        return await fetchKlinesBatchSpot(symbol, interval, startTime, endTime);
      }
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }
    
    const data = await response.json();
    return data.map((k: any[]) => ({
      openTime: k[0],
      open: k[1],
      high: k[2],
      low: k[3],
      close: k[4],
      volume: k[5],
      closeTime: k[6],
    }));
  } catch (error) {
    console.error("[Historical] Error fetching from Binance futures:", error);
    return await fetchKlinesBatchSpot(symbol, interval, startTime, endTime);
  }
}

async function fetchKlinesBatchSpot(
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number
): Promise<BinanceKline[]> {
  const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&startTime=${startTime}&endTime=${endTime}&limit=${CANDLES_PER_REQUEST}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      console.log("[Historical] Spot API also failed, trying Vision...");
      return await fetchKlinesBatchVision(symbol, interval, startTime, endTime);
    }
    
    const data = await response.json();
    return data.map((k: any[]) => ({
      openTime: k[0],
      open: k[1],
      high: k[2],
      low: k[3],
      close: k[4],
      volume: k[5],
      closeTime: k[6],
    }));
  } catch (error) {
    console.error("[Historical] Error fetching from Binance spot:", error);
    return await fetchKlinesBatchVision(symbol, interval, startTime, endTime);
  }
}

async function fetchKlinesBatchVision(
  symbol: string,
  interval: string,
  startTime: number,
  endTime: number
): Promise<BinanceKline[]> {
  const url = `${BINANCE_VISION_BASE}/api/v3/klines?symbol=${symbol}&interval=${interval}&startTime=${startTime}&endTime=${endTime}&limit=${CANDLES_PER_REQUEST}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Vision API HTTP ${response.status}`);
    }
    
    const data = await response.json();
    return data.map((k: any[]) => ({
      openTime: k[0],
      open: k[1],
      high: k[2],
      low: k[3],
      close: k[4],
      volume: k[5],
      closeTime: k[6],
    }));
  } catch (error) {
    console.error("[Historical] Vision API also failed:", error);
    return [];
  }
}

export async function getDataRangeInfo(): Promise<{
  startTs: number | null;
  endTs: number | null;
  totalCandles: number;
  backfillComplete: boolean;
}> {
  try {
    const state = await db.select().from(learningState).where(eq(learningState.key, "btcusdt_15m")).limit(1);
    
    if (state.length === 0) {
      const candleCount = await db.select({ count: sql<number>`count(*)` })
        .from(candles)
        .where(and(
          eq(candles.symbol, "BTCUSDT"),
          eq(candles.timeframe, "15m")
        ));
      
      const oldest = await db.select({ ts: candles.timestamp })
        .from(candles)
        .where(and(eq(candles.symbol, "BTCUSDT"), eq(candles.timeframe, "15m")))
        .orderBy(asc(candles.timestamp))
        .limit(1);
      
      const newest = await db.select({ ts: candles.timestamp })
        .from(candles)
        .where(and(eq(candles.symbol, "BTCUSDT"), eq(candles.timeframe, "15m")))
        .orderBy(desc(candles.timestamp))
        .limit(1);
      
      return {
        startTs: oldest[0]?.ts ?? null,
        endTs: newest[0]?.ts ?? null,
        totalCandles: Number(candleCount[0]?.count ?? 0),
        backfillComplete: false,
      };
    }
    
    return {
      startTs: state[0].dataRangeStartTs ?? null,
      endTs: state[0].dataRangeEndTs ?? null,
      totalCandles: state[0].totalCandles ?? 0,
      backfillComplete: state[0].backfillComplete ?? false,
    };
  } catch (error) {
    console.error("[Historical] Error getting data range:", error);
    return { startTs: null, endTs: null, totalCandles: 0, backfillComplete: false };
  }
}

async function upsertCandles(klines: BinanceKline[], symbol: string, timeframe: string): Promise<number> {
  if (klines.length === 0) return 0;
  
  let inserted = 0;
  for (const kline of klines) {
    try {
      const existing = await db.select({ id: candles.id })
        .from(candles)
        .where(and(
          eq(candles.symbol, symbol),
          eq(candles.timestamp, kline.openTime),
          eq(candles.timeframe, timeframe)
        ))
        .limit(1);
      
      if (existing.length === 0) {
        await db.insert(candles).values({
          symbol,
          timestamp: kline.openTime,
          timeframe,
          open: parseFloat(kline.open),
          high: parseFloat(kline.high),
          low: parseFloat(kline.low),
          close: parseFloat(kline.close),
          volume: parseFloat(kline.volume),
        });
        inserted++;
      }
    } catch (error) {
    }
  }
  
  return inserted;
}

async function updateLearningState(symbol: string, timeframe: string): Promise<void> {
  const key = `${symbol.toLowerCase()}_${timeframe}`;
  
  const candleStats = await db.select({
    count: sql<number>`count(*)`,
    minTs: sql<number>`min(timestamp)`,
    maxTs: sql<number>`max(timestamp)`,
  })
    .from(candles)
    .where(and(
      eq(candles.symbol, symbol),
      eq(candles.timeframe, timeframe)
    ));
  
  const totalCandles = Number(candleStats[0]?.count ?? 0);
  const startTs = Number(candleStats[0]?.minTs ?? 0);
  const endTs = Number(candleStats[0]?.maxTs ?? 0);
  
  const expectedCandles = Math.floor((BACKFILL_DAYS * 24 * 60) / 15);
  const backfillComplete = totalCandles >= expectedCandles * 0.95;
  
  const existing = await db.select().from(learningState).where(eq(learningState.key, key)).limit(1);
  
  if (existing.length === 0) {
    await db.insert(learningState).values({
      key,
      dataRangeStartTs: startTs,
      dataRangeEndTs: endTs,
      totalCandles,
      lastIngestedTs: endTs,
      backfillComplete,
      updatedTs: Date.now(),
    });
  } else {
    await db.update(learningState)
      .set({
        dataRangeStartTs: startTs,
        dataRangeEndTs: endTs,
        totalCandles,
        lastIngestedTs: endTs,
        backfillComplete,
        updatedTs: Date.now(),
      })
      .where(eq(learningState.key, key));
  }
}

export async function backfillHistoricalData(
  symbol: string = "BTCUSDT",
  timeframe: string = "15m",
  daysBack: number = BACKFILL_DAYS,
  onProgress?: (progress: number, message: string) => void
): Promise<{ success: boolean; totalCandles: number; newCandles: number; gaps: number[] }> {
  const now = Date.now();
  const startTime = now - (daysBack * 24 * 60 * 60 * 1000);
  const msPerCandle = timeframe === "15m" ? MS_PER_15M : 60 * 1000;
  
  console.log(`[Historical] Starting backfill for ${symbol} ${timeframe}, ${daysBack} days back...`);
  onProgress?.(0, `Starting backfill for ${daysBack} days of data...`);
  
  let currentStart = startTime;
  let totalNewCandles = 0;
  let batchCount = 0;
  const expectedBatches = Math.ceil((now - startTime) / (CANDLES_PER_REQUEST * msPerCandle));
  
  while (currentStart < now) {
    const batchEnd = Math.min(currentStart + (CANDLES_PER_REQUEST * msPerCandle), now);
    
    try {
      const klines = await fetchKlinesBatch(symbol, timeframe, currentStart, batchEnd);
      
      if (klines.length > 0) {
        const inserted = await upsertCandles(klines, symbol, timeframe);
        totalNewCandles += inserted;
        
        const progress = Math.min(100, Math.round((batchCount / expectedBatches) * 100));
        onProgress?.(progress, `Fetched ${klines.length} candles, ${inserted} new. Total new: ${totalNewCandles}`);
        
        if (batchCount % 10 === 0) {
          console.log(`[Historical] Progress: ${progress}%, ${totalNewCandles} new candles inserted`);
        }
      }
      
      currentStart = batchEnd;
      batchCount++;
      
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error) {
      console.error(`[Historical] Error in batch at ${new Date(currentStart).toISOString()}:`, error);
      currentStart = batchEnd;
      batchCount++;
    }
  }
  
  await updateLearningState(symbol, timeframe);
  
  const gaps = await detectGaps(symbol, timeframe, startTime, now);
  
  console.log(`[Historical] Backfill complete: ${totalNewCandles} new candles, ${gaps.length} gaps detected`);
  onProgress?.(100, `Backfill complete! ${totalNewCandles} new candles added.`);
  
  const rangeInfo = await getDataRangeInfo();
  
  return {
    success: true,
    totalCandles: rangeInfo.totalCandles,
    newCandles: totalNewCandles,
    gaps,
  };
}

async function detectGaps(
  symbol: string,
  timeframe: string,
  startTime: number,
  endTime: number
): Promise<number[]> {
  const msPerCandle = timeframe === "15m" ? MS_PER_15M : 60 * 1000;
  const gaps: number[] = [];
  
  const allCandles = await db.select({ timestamp: candles.timestamp })
    .from(candles)
    .where(and(
      eq(candles.symbol, symbol),
      eq(candles.timeframe, timeframe),
      gte(candles.timestamp, startTime),
      lte(candles.timestamp, endTime)
    ))
    .orderBy(asc(candles.timestamp));
  
  for (let i = 1; i < allCandles.length; i++) {
    const expected = allCandles[i - 1].timestamp + msPerCandle;
    const actual = allCandles[i].timestamp;
    
    if (actual - expected > msPerCandle * 1.5) {
      gaps.push(expected);
    }
  }
  
  return gaps;
}

export async function fillGaps(
  symbol: string = "BTCUSDT",
  timeframe: string = "15m"
): Promise<number> {
  const rangeInfo = await getDataRangeInfo();
  if (!rangeInfo.startTs || !rangeInfo.endTs) return 0;
  
  const gaps = await detectGaps(symbol, timeframe, rangeInfo.startTs, rangeInfo.endTs);
  
  if (gaps.length === 0) {
    console.log("[Historical] No gaps detected");
    return 0;
  }
  
  console.log(`[Historical] Filling ${gaps.length} gaps...`);
  
  let filled = 0;
  const msPerCandle = timeframe === "15m" ? MS_PER_15M : 60 * 1000;
  
  for (const gapStart of gaps) {
    const klines = await fetchKlinesBatch(symbol, timeframe, gapStart, gapStart + msPerCandle * 10);
    const inserted = await upsertCandles(klines, symbol, timeframe);
    filled += inserted;
    
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  if (filled > 0) {
    await updateLearningState(symbol, timeframe);
  }
  
  return filled;
}

export async function incrementalUpdate(
  symbol: string = "BTCUSDT",
  timeframe: string = "15m"
): Promise<{ newCandles: number; totalCandles: number }> {
  const rangeInfo = await getDataRangeInfo();
  const msPerCandle = timeframe === "15m" ? MS_PER_15M : 60 * 1000;
  
  let startFrom: number;
  if (rangeInfo.endTs) {
    startFrom = rangeInfo.endTs;
  } else {
    startFrom = Date.now() - (BACKFILL_DAYS * 24 * 60 * 60 * 1000);
  }
  
  const now = Date.now();
  
  if (now - startFrom < msPerCandle) {
    return { newCandles: 0, totalCandles: rangeInfo.totalCandles };
  }
  
  console.log(`[Historical] Incremental update from ${new Date(startFrom).toISOString()}`);
  
  let totalNew = 0;
  let currentStart = startFrom;
  
  while (currentStart < now) {
    const batchEnd = Math.min(currentStart + (CANDLES_PER_REQUEST * msPerCandle), now);
    
    try {
      const klines = await fetchKlinesBatch(symbol, timeframe, currentStart, batchEnd);
      
      if (klines.length > 0) {
        const inserted = await upsertCandles(klines, symbol, timeframe);
        totalNew += inserted;
      }
      
      currentStart = batchEnd;
      await new Promise(resolve => setTimeout(resolve, 50));
      
    } catch (error) {
      console.error(`[Historical] Incremental update error:`, error);
      currentStart = batchEnd;
    }
  }
  
  if (totalNew > 0) {
    await updateLearningState(symbol, timeframe);
  }
  
  const newRangeInfo = await getDataRangeInfo();
  
  return {
    newCandles: totalNew,
    totalCandles: newRangeInfo.totalCandles,
  };
}

export async function loadCandlesFromDb(
  symbol: string = "BTCUSDT",
  timeframe: string = "15m",
  limit?: number
): Promise<Array<{
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}>> {
  if (limit) {
    const result = await db.select({
      timestamp: candles.timestamp,
      open: candles.open,
      high: candles.high,
      low: candles.low,
      close: candles.close,
      volume: candles.volume,
    })
      .from(candles)
      .where(and(
        eq(candles.symbol, symbol),
        eq(candles.timeframe, timeframe)
      ))
      .orderBy(desc(candles.timestamp))
      .limit(limit);
    
    return result.reverse();
  }
  
  return await db.select({
    timestamp: candles.timestamp,
    open: candles.open,
    high: candles.high,
    low: candles.low,
    close: candles.close,
    volume: candles.volume,
  })
    .from(candles)
    .where(and(
      eq(candles.symbol, symbol),
      eq(candles.timeframe, timeframe)
    ))
    .orderBy(asc(candles.timestamp));
}

export async function getStoredCandleCount(): Promise<number> {
  const result = await db.select({ count: sql<number>`count(*)` })
    .from(candles)
    .where(and(
      eq(candles.symbol, "BTCUSDT"),
      eq(candles.timeframe, "15m")
    ));
  return Number(result[0]?.count ?? 0);
}

export async function getStoredCandles(limit: number = 1000): Promise<typeof candles.$inferSelect[]> {
  return await db.select().from(candles)
    .where(and(
      eq(candles.symbol, "BTCUSDT"),
      eq(candles.timeframe, "15m")
    ))
    .orderBy(desc(candles.timestamp))
    .limit(limit);
}

export async function getCandlesInRange(startTs: number, endTs: number): Promise<typeof candles.$inferSelect[]> {
  return await db.select()
    .from(candles)
    .where(and(
      eq(candles.symbol, "BTCUSDT"),
      eq(candles.timeframe, "15m"),
      gte(candles.timestamp, startTs),
      lte(candles.timestamp, endTs)
    ))
    .orderBy(asc(candles.timestamp));
}
