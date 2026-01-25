import { db } from "./db";
import { candles, learningState, backfillJobs } from "./db/schema";
import { eq, and, gte, lte, sql, desc, asc, or, ne } from "drizzle-orm";

const BINANCE_VISION_BASE = "https://data-api.binance.vision";
const BINANCE_FAPI_BASE = "https://fapi.binance.com";
const CANDLES_PER_REQUEST = 1000;
const MS_PER_15M = 15 * 60 * 1000;
const BACKFILL_DAYS = 1826; // 5 years of historical data

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

const CANDLES_PER_DAY = 96;
const TARGET_DAYS = 1826; // 5 years of data for comprehensive training
const CANDLES_FOR_TARGET = TARGET_DAYS * CANDLES_PER_DAY; // ~175,296 candles

export interface DataRangeInfo {
  startTs: number | null;
  endTs: number | null;
  totalCandles: number;
  backfillComplete: boolean;
  daysOfData: number;
  expectedForTarget: number;
  completionPct: number;
}

export interface IntegrityReport {
  totalCandles: number;
  daysOfData: number;
  completionPct: number;
  missingRanges: Array<{ start: string; end: string; gapCandles: number }>;
  duplicateCount: number;
  lastCandleTs: number | null;
  alignmentHealthy: boolean;
  overallHealth: "complete" | "missing_ranges" | "out_of_sync" | "no_data";
}

export async function getDataRangeInfo(): Promise<DataRangeInfo> {
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
      
      const totalCandles = Number(candleCount[0]?.count ?? 0);
      const startTs = oldest[0]?.ts ?? null;
      const endTs = newest[0]?.ts ?? null;
      const daysOfData = startTs && endTs 
        ? Math.round((endTs - startTs) / (24 * 60 * 60 * 1000))
        : 0;
      const completionPct = Math.min((totalCandles / CANDLES_FOR_TARGET) * 100, 100);
      
      return {
        startTs,
        endTs,
        totalCandles,
        backfillComplete: false,
        daysOfData,
        expectedForTarget: CANDLES_FOR_TARGET,
        completionPct,
      };
    }
    
    const totalCandles = state[0].totalCandles ?? 0;
    const startTs = state[0].dataRangeStartTs ?? null;
    const endTs = state[0].dataRangeEndTs ?? null;
    const daysOfData = startTs && endTs 
      ? Math.round((endTs - startTs) / (24 * 60 * 60 * 1000))
      : 0;
    const completionPct = Math.min((totalCandles / CANDLES_FOR_TARGET) * 100, 100);
    
    return {
      startTs,
      endTs,
      totalCandles,
      backfillComplete: state[0].backfillComplete ?? false,
      daysOfData,
      expectedForTarget: CANDLES_FOR_TARGET,
      completionPct,
    };
  } catch (error) {
    console.error("[Historical] Error getting data range:", error);
    return { 
      startTs: null, 
      endTs: null, 
      totalCandles: 0, 
      backfillComplete: false,
      daysOfData: 0,
      expectedForTarget: CANDLES_FOR_TARGET,
      completionPct: 0,
    };
  }
}

export async function getIntegrityReport(
  symbol: string = "BTCUSDT",
  timeframe: string = "15m"
): Promise<IntegrityReport> {
  try {
    const msPerCandle = timeframe === "15m" ? MS_PER_15M : 60 * 1000;
    
    const candleCount = await db.select({ count: sql<number>`count(*)` })
      .from(candles)
      .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, timeframe)));
    
    const totalCandles = Number(candleCount[0]?.count ?? 0);
    
    if (totalCandles === 0) {
      return {
        totalCandles: 0,
        daysOfData: 0,
        completionPct: 0,
        missingRanges: [],
        duplicateCount: 0,
        lastCandleTs: null,
        alignmentHealthy: true,
        overallHealth: "no_data",
      };
    }
    
    const oldest = await db.select({ ts: candles.timestamp })
      .from(candles)
      .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, timeframe)))
      .orderBy(asc(candles.timestamp))
      .limit(1);
    
    const newest = await db.select({ ts: candles.timestamp })
      .from(candles)
      .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, timeframe)))
      .orderBy(desc(candles.timestamp))
      .limit(1);
    
    const startTs = oldest[0]?.ts ?? 0;
    const endTs = newest[0]?.ts ?? 0;
    const daysOfData = Math.round((endTs - startTs) / (24 * 60 * 60 * 1000));
    const completionPct = Math.min((totalCandles / CANDLES_FOR_TARGET) * 100, 100);
    
    const duplicateCheck = await db.execute(sql`
      SELECT COUNT(*) - COUNT(DISTINCT timestamp) as duplicate_count
      FROM candles 
      WHERE symbol = ${symbol} AND timeframe = ${timeframe}
    `);
    const duplicateCount = Number((duplicateCheck as any)[0]?.duplicate_count ?? 0);
    
    const allTimestamps = await db.select({ ts: candles.timestamp })
      .from(candles)
      .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, timeframe)))
      .orderBy(asc(candles.timestamp));
    
    const missingRanges: Array<{ start: string; end: string; gapCandles: number }> = [];
    
    for (let i = 1; i < allTimestamps.length; i++) {
      const expected = allTimestamps[i - 1].ts + msPerCandle;
      const actual = allTimestamps[i].ts;
      
      if (actual - expected > msPerCandle * 1.5) {
        const gapCandles = Math.floor((actual - expected) / msPerCandle);
        missingRanges.push({
          start: new Date(expected).toISOString(),
          end: new Date(actual).toISOString(),
          gapCandles,
        });
      }
    }
    
    const alignmentHealthy = allTimestamps.every(t => t.ts % msPerCandle === 0);
    
    let overallHealth: "complete" | "missing_ranges" | "out_of_sync" | "no_data" = "complete";
    if (missingRanges.length > 0) overallHealth = "missing_ranges";
    if (duplicateCount > 0 || !alignmentHealthy) overallHealth = "out_of_sync";
    
    return {
      totalCandles,
      daysOfData,
      completionPct,
      missingRanges: missingRanges.slice(0, 20),
      duplicateCount,
      lastCandleTs: endTs,
      alignmentHealthy,
      overallHealth,
    };
  } catch (error) {
    console.error("[Historical] Error getting integrity report:", error);
    return {
      totalCandles: 0,
      daysOfData: 0,
      completionPct: 0,
      missingRanges: [],
      duplicateCount: 0,
      lastCandleTs: null,
      alignmentHealthy: false,
      overallHealth: "out_of_sync",
    };
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

async function findOrCreateBackfillJob(
  symbol: string,
  timeframe: string,
  daysBack: number
): Promise<{ id: number; startTs: number; currentCursor: number; candlesFetched: number; isResume: boolean }> {
  const now = Date.now();
  const startTs = now - (daysBack * 24 * 60 * 60 * 1000);
  
  const existingJob = await db.select()
    .from(backfillJobs)
    .where(and(
      eq(backfillJobs.symbol, symbol),
      eq(backfillJobs.timeframe, timeframe),
      or(
        eq(backfillJobs.status, "running"),
        eq(backfillJobs.status, "pending"),
        eq(backfillJobs.status, "error")
      )
    ))
    .orderBy(desc(backfillJobs.id))
    .limit(1);
  
  if (existingJob.length > 0 && existingJob[0].currentCursor) {
    const job = existingJob[0];
    
    const progress = job.progressPct ?? 0;
    if (progress >= 100) {
      console.log(`[Historical] Job ${job.id} already complete (${progress}%), creating new job`);
    } else {
      console.log(`[Historical] Resuming backfill job ${job.id} (status: ${job.status}) from cursor ${new Date(job.currentCursor!).toISOString()}`);
      
      await db.update(backfillJobs)
        .set({ status: "running", updatedTs: now, errorMessage: null })
        .where(eq(backfillJobs.id, job.id));
      
      return {
        id: job.id,
        startTs: job.startTs ?? startTs,
        currentCursor: job.currentCursor!,
        candlesFetched: job.candlesFetched ?? 0,
        isResume: true,
      };
    }
  }
  
  const msPerCandle = timeframe === "15m" ? MS_PER_15M : 60 * 1000;
  const expectedCandles = Math.floor((now - startTs) / msPerCandle);
  
  const [newJob] = await db.insert(backfillJobs)
    .values({
      symbol,
      timeframe,
      status: "running",
      startTs,
      endTs: now,
      currentCursor: startTs,
      candlesFetched: 0,
      candlesExpected: expectedCandles,
      progressPct: 0,
      createdTs: now,
      updatedTs: now,
    })
    .returning();
  
  console.log(`[Historical] Created new backfill job ${newJob.id}`);
  
  return {
    id: newJob.id,
    startTs,
    currentCursor: startTs,
    candlesFetched: 0,
    isResume: false,
  };
}

async function updateBackfillJobProgress(
  jobId: number,
  currentCursor: number,
  candlesFetched: number,
  progressPct: number,
  status: string = "running"
): Promise<void> {
  await db.update(backfillJobs)
    .set({
      currentCursor,
      candlesFetched,
      progressPct,
      status,
      updatedTs: Date.now(),
    })
    .where(eq(backfillJobs.id, jobId));
}

export async function getActiveBackfillJob(symbol: string = "BTCUSDT", timeframe: string = "15m") {
  const jobs = await db.select()
    .from(backfillJobs)
    .where(and(
      eq(backfillJobs.symbol, symbol),
      eq(backfillJobs.timeframe, timeframe),
      or(
        eq(backfillJobs.status, "running"),
        eq(backfillJobs.status, "pending"),
        eq(backfillJobs.status, "error")
      )
    ))
    .orderBy(desc(backfillJobs.id))
    .limit(1);
  
  const job = jobs[0];
  if (!job) return null;
  
  if (job.status === "error" && job.progressPct && job.progressPct >= 100) {
    return null;
  }
  
  return job;
}

export async function backfillHistoricalData(
  symbol: string = "BTCUSDT",
  timeframe: string = "15m",
  daysBack: number = BACKFILL_DAYS,
  onProgress?: (progress: number, message: string) => void
): Promise<{ success: boolean; totalCandles: number; newCandles: number; gaps: number[]; jobId: number }> {
  const now = Date.now();
  const msPerCandle = timeframe === "15m" ? MS_PER_15M : 60 * 1000;
  
  const job = await findOrCreateBackfillJob(symbol, timeframe, daysBack);
  
  console.log(`[Historical] ${job.isResume ? 'Resuming' : 'Starting'} backfill for ${symbol} ${timeframe}, ${daysBack} days back...`);
  onProgress?.(0, job.isResume 
    ? `Resuming backfill from ${new Date(job.currentCursor).toISOString().split('T')[0]}...`
    : `Starting backfill for ${daysBack} days of data...`);
  
  let currentStart = job.currentCursor;
  let totalNewCandles = job.candlesFetched;
  let batchCount = 0;
  const expectedBatches = Math.ceil((now - job.startTs) / (CANDLES_PER_REQUEST * msPerCandle));
  const completedBatches = Math.ceil((job.currentCursor - job.startTs) / (CANDLES_PER_REQUEST * msPerCandle));
  batchCount = completedBatches;
  
  while (currentStart < now) {
    const batchEnd = Math.min(currentStart + (CANDLES_PER_REQUEST * msPerCandle), now);
    
    try {
      const klines = await fetchKlinesBatch(symbol, timeframe, currentStart, batchEnd);
      
      if (klines.length > 0) {
        const inserted = await upsertCandles(klines, symbol, timeframe);
        totalNewCandles += inserted;
      }
      
      const progress = Math.min(99, Math.round((batchCount / expectedBatches) * 100));
      onProgress?.(progress, `Fetched ${klines.length} candles. Total: ${totalNewCandles}`);
      
      currentStart = batchEnd;
      batchCount++;
      
      if (batchCount % 5 === 0) {
        await updateBackfillJobProgress(job.id, currentStart, totalNewCandles, progress);
        console.log(`[Historical] Progress: ${progress}%, cursor saved at ${new Date(currentStart).toISOString()}`);
      }
      
      await new Promise(resolve => setTimeout(resolve, 100));
      
    } catch (error: any) {
      console.error(`[Historical] Error in batch at ${new Date(currentStart).toISOString()}:`, error);
      
      await updateBackfillJobProgress(job.id, currentStart, totalNewCandles, 
        Math.round((batchCount / expectedBatches) * 100), "error");
      await db.update(backfillJobs)
        .set({ errorMessage: error.message || "Unknown error" })
        .where(eq(backfillJobs.id, job.id));
      
      currentStart = batchEnd;
      batchCount++;
    }
  }
  
  await updateBackfillJobProgress(job.id, now, totalNewCandles, 100, "completed");
  await updateLearningState(symbol, timeframe);
  
  const gaps = await detectGaps(symbol, timeframe, job.startTs, now);
  
  console.log(`[Historical] Backfill complete: ${totalNewCandles} candles, ${gaps.length} gaps detected`);
  onProgress?.(100, `Backfill complete! ${totalNewCandles} candles stored.`);
  
  const rangeInfo = await getDataRangeInfo();
  
  return {
    success: true,
    totalCandles: rangeInfo.totalCandles,
    newCandles: totalNewCandles,
    gaps,
    jobId: job.id,
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

export async function checkIncompleteBackfillJobs(): Promise<{
  hasIncomplete: boolean;
  jobId?: number;
  symbol?: string;
  timeframe?: string;
  progressPct?: number;
  candlesFetched?: number;
  status?: string;
}> {
  const incompleteJobs = await db.select()
    .from(backfillJobs)
    .where(or(
      eq(backfillJobs.status, "running"),
      eq(backfillJobs.status, "pending"),
      eq(backfillJobs.status, "error")
    ))
    .orderBy(desc(backfillJobs.id))
    .limit(1);
  
  if (incompleteJobs.length > 0) {
    const job = incompleteJobs[0];
    
    if (job.status === "error" && job.progressPct && job.progressPct >= 100) {
      return { hasIncomplete: false };
    }
    
    const isResumable = (job.progressPct ?? 0) < 100 && job.currentCursor;
    
    if (isResumable) {
      console.log(`[Historical] Found incomplete backfill job ${job.id}: ${job.status}, ${job.progressPct}% complete`);
      
      return {
        hasIncomplete: true,
        jobId: job.id,
        symbol: job.symbol ?? "BTCUSDT",
        timeframe: job.timeframe ?? "15m",
        progressPct: job.progressPct ?? 0,
        candlesFetched: job.candlesFetched ?? 0,
        status: job.status ?? "unknown",
      };
    }
  }
  
  return { hasIncomplete: false };
}

// Multi-asset data management
const SUPPORTED_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];

export interface AssetDataSummary {
  symbol: string;
  totalCandles: number;
  startTs: number | null;
  endTs: number | null;
  daysOfData: number;
  yearsOfData: number;
  startDate: string | null;
  endDate: string | null;
}

export interface MultiAssetDataSummary {
  assets: AssetDataSummary[];
  totalCandles: number;
  alignedTimeRange: {
    startTs: number | null;
    endTs: number | null;
    startDate: string | null;
    endDate: string | null;
  };
  allAssetsAvailable: boolean;
}

export async function getMultiAssetDataSummary(): Promise<MultiAssetDataSummary> {
  const assetSummaries: AssetDataSummary[] = [];
  
  for (const symbol of SUPPORTED_ASSETS) {
    const candleCount = await db.select({ count: sql<number>`count(*)` })
      .from(candles)
      .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, "15m")));
    
    const totalCandles = Number(candleCount[0]?.count ?? 0);
    
    if (totalCandles === 0) {
      assetSummaries.push({
        symbol,
        totalCandles: 0,
        startTs: null,
        endTs: null,
        daysOfData: 0,
        yearsOfData: 0,
        startDate: null,
        endDate: null,
      });
      continue;
    }
    
    const oldest = await db.select({ ts: candles.timestamp })
      .from(candles)
      .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, "15m")))
      .orderBy(asc(candles.timestamp))
      .limit(1);
    
    const newest = await db.select({ ts: candles.timestamp })
      .from(candles)
      .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, "15m")))
      .orderBy(desc(candles.timestamp))
      .limit(1);
    
    const startTs = oldest[0]?.ts ?? null;
    const endTs = newest[0]?.ts ?? null;
    const daysOfData = startTs && endTs ? Math.round((endTs - startTs) / (24 * 60 * 60 * 1000)) : 0;
    const yearsOfData = Math.round(daysOfData / 365 * 10) / 10;
    
    assetSummaries.push({
      symbol,
      totalCandles,
      startTs,
      endTs,
      daysOfData,
      yearsOfData,
      startDate: startTs ? new Date(startTs).toISOString().split("T")[0] : null,
      endDate: endTs ? new Date(endTs).toISOString().split("T")[0] : null,
    });
  }
  
  const assetsWithData = assetSummaries.filter(a => a.totalCandles > 0);
  const allAssetsAvailable = assetsWithData.length === SUPPORTED_ASSETS.length;
  
  let alignedStart: number | null = null;
  let alignedEnd: number | null = null;
  
  if (assetsWithData.length > 0) {
    alignedStart = Math.max(...assetsWithData.map(a => a.startTs!));
    alignedEnd = Math.min(...assetsWithData.map(a => a.endTs!));
  }
  
  return {
    assets: assetSummaries,
    totalCandles: assetSummaries.reduce((sum, a) => sum + a.totalCandles, 0),
    alignedTimeRange: {
      startTs: alignedStart,
      endTs: alignedEnd,
      startDate: alignedStart ? new Date(alignedStart).toISOString().split("T")[0] : null,
      endDate: alignedEnd ? new Date(alignedEnd).toISOString().split("T")[0] : null,
    },
    allAssetsAvailable,
  };
}

export interface BulkDownloadProgress {
  symbol: string;
  status: "pending" | "downloading" | "complete" | "error";
  progress: number;
  candlesFetched: number;
  totalExpected: number;
  error?: string;
}

export interface BulkDownloadResult {
  success: boolean;
  assetsDownloaded: string[];
  totalCandlesFetched: number;
  errors: { symbol: string; error: string }[];
}

let bulkDownloadInProgress = false;
let bulkDownloadProgress: Map<string, BulkDownloadProgress> = new Map();

export function getBulkDownloadStatus(): { inProgress: boolean; progress: BulkDownloadProgress[] } {
  return {
    inProgress: bulkDownloadInProgress,
    progress: Array.from(bulkDownloadProgress.values()),
  };
}

export async function downloadMultiAssetData(
  years: number,
  assets: string[] = SUPPORTED_ASSETS,
  onProgress?: (symbol: string, progress: number, candlesFetched: number) => void
): Promise<BulkDownloadResult> {
  if (bulkDownloadInProgress) {
    throw new Error("Bulk download already in progress");
  }
  
  bulkDownloadInProgress = true;
  bulkDownloadProgress.clear();
  
  const daysToFetch = years * 365;
  const candlesExpected = Math.floor(daysToFetch * 24 * 4); // 15-min candles
  const now = Date.now();
  const startTime = now - (daysToFetch * 24 * 60 * 60 * 1000);
  
  const results: BulkDownloadResult = {
    success: true,
    assetsDownloaded: [],
    totalCandlesFetched: 0,
    errors: [],
  };
  
  for (const symbol of assets) {
    bulkDownloadProgress.set(symbol, {
      symbol,
      status: "pending",
      progress: 0,
      candlesFetched: 0,
      totalExpected: candlesExpected,
    });
  }
  
  try {
    for (const symbol of assets) {
      bulkDownloadProgress.set(symbol, {
        symbol,
        status: "downloading",
        progress: 0,
        candlesFetched: 0,
        totalExpected: candlesExpected,
      });
      
      console.log(`[Bulk Download] Starting ${symbol}: fetching ${years} years (${candlesExpected} candles)`);
      
      try {
        let cursor = startTime;
        let totalFetched = 0;
        
        while (cursor < now) {
          const batchEnd = Math.min(cursor + (CANDLES_PER_REQUEST * MS_PER_15M), now);
          
          const klines = await fetchKlinesBatch(symbol, "15m", cursor, batchEnd);
          
          if (klines.length > 0) {
            const candleInserts = klines.map(k => ({
              symbol,
              timestamp: k.openTime,
              timeframe: "15m" as const,
              open: parseFloat(k.open),
              high: parseFloat(k.high),
              low: parseFloat(k.low),
              close: parseFloat(k.close),
              volume: parseFloat(k.volume),
            }));
            
            for (const candle of candleInserts) {
              await db.insert(candles)
                .values(candle)
                .onConflictDoNothing();
            }
            
            totalFetched += klines.length;
            cursor = klines[klines.length - 1].openTime + MS_PER_15M;
          } else {
            cursor = batchEnd + MS_PER_15M;
          }
          
          const progress = Math.min(((cursor - startTime) / (now - startTime)) * 100, 100);
          bulkDownloadProgress.set(symbol, {
            symbol,
            status: "downloading",
            progress,
            candlesFetched: totalFetched,
            totalExpected: candlesExpected,
          });
          
          if (onProgress) {
            onProgress(symbol, progress, totalFetched);
          }
          
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        bulkDownloadProgress.set(symbol, {
          symbol,
          status: "complete",
          progress: 100,
          candlesFetched: totalFetched,
          totalExpected: candlesExpected,
        });
        
        results.assetsDownloaded.push(symbol);
        results.totalCandlesFetched += totalFetched;
        
        console.log(`[Bulk Download] Completed ${symbol}: ${totalFetched} candles`);
        
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : String(error);
        bulkDownloadProgress.set(symbol, {
          symbol,
          status: "error",
          progress: 0,
          candlesFetched: 0,
          totalExpected: candlesExpected,
          error: errorMsg,
        });
        results.errors.push({ symbol, error: errorMsg });
        results.success = false;
        console.error(`[Bulk Download] Error fetching ${symbol}:`, errorMsg);
      }
    }
  } finally {
    bulkDownloadInProgress = false;
  }
  
  return results;
}

export async function clearAllAssetData(): Promise<{ success: boolean; candlesDeleted: number }> {
  try {
    const countResult = await db.select({ count: sql<number>`count(*)` }).from(candles);
    const totalCandles = Number(countResult[0]?.count ?? 0);
    
    await db.delete(candles);
    await db.delete(learningState);
    await db.delete(backfillJobs);
    
    console.log(`[Data Clear] Deleted ${totalCandles} candles and reset learning state`);
    
    return { success: true, candlesDeleted: totalCandles };
  } catch (error) {
    console.error("[Data Clear] Error:", error);
    return { success: false, candlesDeleted: 0 };
  }
}

export async function loadAssetCandlesFromDb(symbol: string, timeframe: string = "15m"): Promise<any[]> {
  const result = await db.select()
    .from(candles)
    .where(and(eq(candles.symbol, symbol), eq(candles.timeframe, timeframe)))
    .orderBy(asc(candles.timestamp));
  
  return result.map(c => ({
    timestamp: c.timestamp,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
    volume: c.volume,
  }));
}

export function getSupportedAssets(): string[] {
  return SUPPORTED_ASSETS;
}
