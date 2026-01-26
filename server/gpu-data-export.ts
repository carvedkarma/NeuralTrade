import { db } from "./db";
import { candles } from "@shared/schema";
import { eq, and, gte, lte, asc, desc, sql } from "drizzle-orm";

interface CandleRow {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface MultiTFCandle {
  timestamp: number;
  symbol: string;
  tf_1m: CandleRow | null;
  tf_5m: CandleRow | null;
  tf_15m: CandleRow | null;
  tf_1h: CandleRow | null;
  tf_4h: CandleRow | null;
}

interface FeatureSpec {
  name: string;
  category: string;
  formula: string;
  window?: number;
  normalize: boolean;
}

interface TrainingSample {
  timestamp: number;
  symbol: string;
  features: number[];
  labels: {
    return_15m: number;
    return_60m: number;
    return_240m: number;
    direction_15m: number;
    direction_60m: number;
    direction_240m: number;
  };
}

interface ScalerParams {
  feature: string;
  median: number;
  iqr: number;
  min: number;
  max: number;
}

interface WalkForwardFold {
  foldId: number;
  trainStart: number;
  trainEnd: number;
  valStart: number;
  valEnd: number;
  testStart: number;
  testEnd: number;
}

export const FEATURE_SPECS: FeatureSpec[] = [
  { name: "log_return_1", category: "returns", formula: "log(close/close[-1])", normalize: true },
  { name: "log_return_5", category: "returns", formula: "log(close/close[-5])", normalize: true },
  { name: "log_return_20", category: "returns", formula: "log(close/close[-20])", normalize: true },
  { name: "volatility_20", category: "volatility", formula: "std(log_return_1, 20)", window: 20, normalize: true },
  { name: "volatility_60", category: "volatility", formula: "std(log_return_1, 60)", window: 60, normalize: true },
  { name: "volatility_240", category: "volatility", formula: "std(log_return_1, 240)", window: 240, normalize: true },
  { name: "ema_ratio_20", category: "trend", formula: "close/ema(close, 20) - 1", window: 20, normalize: true },
  { name: "ema_ratio_50", category: "trend", formula: "close/ema(close, 50) - 1", window: 50, normalize: true },
  { name: "ema_ratio_200", category: "trend", formula: "close/ema(close, 200) - 1", window: 200, normalize: true },
  { name: "rsi_14", category: "momentum", formula: "rsi(close, 14)", window: 14, normalize: false },
  { name: "macd_line", category: "momentum", formula: "ema(close, 12) - ema(close, 26)", normalize: true },
  { name: "macd_signal", category: "momentum", formula: "ema(macd_line, 9)", normalize: true },
  { name: "macd_hist", category: "momentum", formula: "macd_line - macd_signal", normalize: true },
  { name: "atr_14_norm", category: "volatility", formula: "atr(14) / close", window: 14, normalize: true },
  { name: "body_ratio", category: "candle", formula: "(close - open) / open", normalize: true },
  { name: "wick_up_ratio", category: "candle", formula: "(high - max(open, close)) / open", normalize: true },
  { name: "wick_dn_ratio", category: "candle", formula: "(min(open, close) - low) / open", normalize: true },
  { name: "volume_log", category: "volume", formula: "log1p(volume)", normalize: true },
  { name: "volume_zscore", category: "volume", formula: "(volume - mean(volume, 20)) / std(volume, 20)", window: 20, normalize: false },
  { name: "trend_slope_20", category: "regime", formula: "linreg_slope(ema(close, 20), 20)", window: 20, normalize: true },
  { name: "volatility_regime", category: "regime", formula: "quantile_bucket(volatility_20, [0.33, 0.67])", normalize: false },
  { name: "eth_return_1", category: "cross_asset", formula: "log(eth_close/eth_close[-1])", normalize: true },
  { name: "sol_return_1", category: "cross_asset", formula: "log(sol_close/sol_close[-1])", normalize: true },
  { name: "bnb_return_1", category: "cross_asset", formula: "log(bnb_close/bnb_close[-1])", normalize: true },
  { name: "btc_eth_corr_20", category: "cross_asset", formula: "rolling_corr(btc_return_1, eth_return_1, 20)", window: 20, normalize: false },
  { name: "btc_sol_corr_20", category: "cross_asset", formula: "rolling_corr(btc_return_1, sol_return_1, 20)", window: 20, normalize: false },
  { name: "eth_relative_strength", category: "cross_asset", formula: "eth_return_20 - btc_return_20", normalize: true },
  { name: "sol_relative_strength", category: "cross_asset", formula: "sol_return_20 - btc_return_20", normalize: true },
];

const TF_MINUTES: Record<string, number> = {
  "1m": 1,
  "5m": 5,
  "15m": 15,
  "1h": 60,
  "4h": 240,
};

export async function getAvailableTimeframes(): Promise<{ timeframe: string; symbols: string[]; count: number }[]> {
  const result = await db.execute(sql`
    SELECT timeframe, symbol, COUNT(*) as count 
    FROM candles 
    GROUP BY timeframe, symbol 
    ORDER BY timeframe, symbol
  `);
  
  const tfMap = new Map<string, { symbols: string[]; count: number }>();
  for (const row of result as any[]) {
    const tf = row.timeframe;
    if (!tfMap.has(tf)) {
      tfMap.set(tf, { symbols: [], count: 0 });
    }
    tfMap.get(tf)!.symbols.push(row.symbol);
    tfMap.get(tf)!.count += parseInt(row.count);
  }
  
  return Array.from(tfMap.entries()).map(([timeframe, data]) => ({
    timeframe,
    symbols: data.symbols,
    count: data.count,
  }));
}

export async function getDataRange(symbol: string = "BTCUSDT", timeframe: string = "1m"): Promise<{
  startTs: number;
  endTs: number;
  count: number;
  gaps: number[];
}> {
  const result = await db.execute(sql`
    SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as count
    FROM candles
    WHERE symbol = ${symbol} AND timeframe = ${timeframe}
  `);
  
  const row = (result as any[])[0];
  return {
    startTs: parseInt(row?.min_ts) || 0,
    endTs: parseInt(row?.max_ts) || 0,
    count: parseInt(row?.count) || 0,
    gaps: [],
  };
}

export async function exportMultiTFCandles(
  symbol: string,
  baseTF: string,
  startTs: number,
  endTs: number,
  limit: number = 100000
): Promise<{
  candles: any[];
  count: number;
  baseTF: string;
  higherTFs: string[];
}> {
  const baseMinutes = TF_MINUTES[baseTF] || 1;
  const higherTFs = Object.keys(TF_MINUTES).filter(tf => TF_MINUTES[tf] > baseMinutes);
  
  const baseCandles = await db
    .select()
    .from(candles)
    .where(
      and(
        eq(candles.symbol, symbol),
        eq(candles.timeframe, baseTF),
        gte(candles.timestamp, startTs),
        lte(candles.timestamp, endTs)
      )
    )
    .orderBy(asc(candles.timestamp))
    .limit(limit);
  
  const higherTFData: Record<string, Map<number, CandleRow>> = {};
  for (const tf of higherTFs) {
    const tfCandles = await db
      .select()
      .from(candles)
      .where(
        and(
          eq(candles.symbol, symbol),
          eq(candles.timeframe, tf),
          gte(candles.timestamp, startTs - TF_MINUTES[tf] * 60 * 1000),
          lte(candles.timestamp, endTs)
        )
      )
      .orderBy(asc(candles.timestamp));
    
    higherTFData[tf] = new Map();
    for (const c of tfCandles) {
      higherTFData[tf].set(c.timestamp, {
        timestamp: c.timestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
      });
    }
  }
  
  const asOfJoin = (tfMap: Map<number, CandleRow>, targetTs: number, tfMinutes: number): CandleRow | null => {
    const tfMs = tfMinutes * 60 * 1000;
    const alignedTs = Math.floor(targetTs / tfMs) * tfMs;
    
    for (let offset = 0; offset <= tfMs * 2; offset += tfMs) {
      const checkTs = alignedTs - offset;
      if (tfMap.has(checkTs)) {
        return tfMap.get(checkTs)!;
      }
    }
    return null;
  };
  
  const result = baseCandles.map(base => {
    const row: any = {
      timestamp: base.timestamp,
      symbol: base.symbol,
      [`${baseTF}_open`]: base.open,
      [`${baseTF}_high`]: base.high,
      [`${baseTF}_low`]: base.low,
      [`${baseTF}_close`]: base.close,
      [`${baseTF}_volume`]: base.volume,
    };
    
    for (const tf of higherTFs) {
      const htfCandle = asOfJoin(higherTFData[tf], base.timestamp, TF_MINUTES[tf]);
      if (htfCandle) {
        row[`${tf}_open`] = htfCandle.open;
        row[`${tf}_high`] = htfCandle.high;
        row[`${tf}_low`] = htfCandle.low;
        row[`${tf}_close`] = htfCandle.close;
        row[`${tf}_volume`] = htfCandle.volume;
      } else {
        row[`${tf}_open`] = null;
        row[`${tf}_high`] = null;
        row[`${tf}_low`] = null;
        row[`${tf}_close`] = null;
        row[`${tf}_volume`] = null;
      }
    }
    
    return row;
  });
  
  return {
    candles: result,
    count: result.length,
    baseTF,
    higherTFs,
  };
}

export async function exportCrossAssetAligned(
  baseTF: string,
  startTs: number,
  endTs: number,
  limit: number = 100000
): Promise<{
  candles: any[];
  count: number;
  symbols: string[];
}> {
  const symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"];
  
  const allCandles: Record<string, Map<number, CandleRow>> = {};
  for (const sym of symbols) {
    const symCandles = await db
      .select()
      .from(candles)
      .where(
        and(
          eq(candles.symbol, sym),
          eq(candles.timeframe, baseTF),
          gte(candles.timestamp, startTs),
          lte(candles.timestamp, endTs)
        )
      )
      .orderBy(asc(candles.timestamp))
      .limit(limit);
    
    allCandles[sym] = new Map();
    for (const c of symCandles) {
      allCandles[sym].set(c.timestamp, {
        timestamp: c.timestamp,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
      });
    }
  }
  
  const btcTimestamps = Array.from(allCandles["BTCUSDT"].keys()).sort((a, b) => a - b);
  
  const result = btcTimestamps.map(ts => {
    const row: any = { timestamp: ts };
    
    for (const sym of symbols) {
      const prefix = sym.replace("USDT", "").toLowerCase();
      const candle = allCandles[sym].get(ts);
      if (candle) {
        row[`${prefix}_open`] = candle.open;
        row[`${prefix}_high`] = candle.high;
        row[`${prefix}_low`] = candle.low;
        row[`${prefix}_close`] = candle.close;
        row[`${prefix}_volume`] = candle.volume;
      } else {
        row[`${prefix}_open`] = null;
        row[`${prefix}_high`] = null;
        row[`${prefix}_low`] = null;
        row[`${prefix}_close`] = null;
        row[`${prefix}_volume`] = null;
      }
    }
    
    return row;
  });
  
  return {
    candles: result,
    count: result.length,
    symbols,
  };
}

export function computeRobustScalers(data: number[][]): ScalerParams[] {
  const numFeatures = data[0]?.length || 0;
  const scalers: ScalerParams[] = [];
  
  for (let f = 0; f < numFeatures; f++) {
    const values = data.map(row => row[f]).filter(v => !isNaN(v) && isFinite(v));
    values.sort((a, b) => a - b);
    
    const n = values.length;
    const median = n > 0 ? values[Math.floor(n / 2)] : 0;
    const q1 = n > 0 ? values[Math.floor(n * 0.25)] : 0;
    const q3 = n > 0 ? values[Math.floor(n * 0.75)] : 0;
    const iqr = q3 - q1 || 1;
    const min = values[0] ?? 0;
    const max = values[n - 1] ?? 1;
    
    scalers.push({
      feature: FEATURE_SPECS[f]?.name || `feature_${f}`,
      median,
      iqr,
      min,
      max,
    });
  }
  
  return scalers;
}

export function generateWalkForwardFolds(
  startTs: number,
  endTs: number,
  trainMonths: number = 12,
  valMonths: number = 2,
  testMonths: number = 2,
  stepMonths: number = 2
): WalkForwardFold[] {
  const msPerMonth = 30 * 24 * 60 * 60 * 1000;
  const folds: WalkForwardFold[] = [];
  
  let foldId = 0;
  let currentStart = startTs;
  
  while (true) {
    const trainEnd = currentStart + trainMonths * msPerMonth;
    const valEnd = trainEnd + valMonths * msPerMonth;
    const testEnd = valEnd + testMonths * msPerMonth;
    
    if (testEnd > endTs) break;
    
    folds.push({
      foldId: foldId++,
      trainStart: currentStart,
      trainEnd: trainEnd,
      valStart: trainEnd,
      valEnd: valEnd,
      testStart: valEnd,
      testEnd: testEnd,
    });
    
    currentStart += stepMonths * msPerMonth;
  }
  
  return folds;
}

export const TRADING_COSTS = {
  makerFee: 0.0002,
  takerFee: 0.0004,
  slippage: 0.0001,
  spreadEstimate: 0.0002,
  totalRoundTrip: 0.0009,
};

// Data integrity validation for GPU export
export interface DataIntegrityReport {
  valid: boolean;
  symbol: string;
  timeframe: string;
  totalRecords: number;
  distinctSymbols: string[];
  distinctTimeframes: string[];
  symbolMismatch: boolean;
  timeframeMismatch: boolean;
  duplicateTimestamps: number;
  gapCount: number;
  timestampRange: { min: number; max: number };
  warnings: string[];
}

export async function validateDataIntegrity(
  symbol: string,
  timeframe: string,
  startTs?: number,
  endTs?: number
): Promise<DataIntegrityReport> {
  const warnings: string[] = [];
  
  // Query to check for any cross-contamination
  let query = sql`
    SELECT 
      symbol, 
      timeframe, 
      COUNT(*) as count,
      MIN(timestamp) as min_ts,
      MAX(timestamp) as max_ts
    FROM candles 
    WHERE symbol = ${symbol} AND timeframe = ${timeframe}
  `;
  
  if (startTs && endTs) {
    query = sql`
      SELECT 
        symbol, 
        timeframe, 
        COUNT(*) as count,
        MIN(timestamp) as min_ts,
        MAX(timestamp) as max_ts
      FROM candles 
      WHERE symbol = ${symbol} AND timeframe = ${timeframe}
        AND timestamp >= ${startTs} AND timestamp <= ${endTs}
      GROUP BY symbol, timeframe
    `;
  } else {
    query = sql`
      SELECT 
        symbol, 
        timeframe, 
        COUNT(*) as count,
        MIN(timestamp) as min_ts,
        MAX(timestamp) as max_ts
      FROM candles 
      WHERE symbol = ${symbol} AND timeframe = ${timeframe}
      GROUP BY symbol, timeframe
    `;
  }
  
  const result = await db.execute(query);
  const rows = result as any[];
  
  // Check distinct symbols and timeframes in the filtered data
  const distinctSymbols = Array.from(new Set(rows.map(r => r.symbol)));
  const distinctTimeframes = Array.from(new Set(rows.map(r => r.timeframe)));
  
  const symbolMismatch = distinctSymbols.length > 1 || (distinctSymbols.length === 1 && distinctSymbols[0] !== symbol);
  const timeframeMismatch = distinctTimeframes.length > 1 || (distinctTimeframes.length === 1 && distinctTimeframes[0] !== timeframe);
  
  if (symbolMismatch) {
    warnings.push(`CRITICAL: Found symbols ${distinctSymbols.join(", ")} when expecting only ${symbol}`);
  }
  if (timeframeMismatch) {
    warnings.push(`CRITICAL: Found timeframes ${distinctTimeframes.join(", ")} when expecting only ${timeframe}`);
  }
  
  // Check for duplicate timestamps
  const dupResult = await db.execute(sql`
    SELECT COUNT(*) as dup_count FROM (
      SELECT timestamp, COUNT(*) as cnt 
      FROM candles 
      WHERE symbol = ${symbol} AND timeframe = ${timeframe}
      ${startTs && endTs ? sql`AND timestamp >= ${startTs} AND timestamp <= ${endTs}` : sql``}
      GROUP BY timestamp 
      HAVING COUNT(*) > 1
    ) as dups
  `);
  const duplicateTimestamps = parseInt((dupResult as any[])[0]?.dup_count || "0");
  
  if (duplicateTimestamps > 0) {
    warnings.push(`WARNING: Found ${duplicateTimestamps} duplicate timestamps in ${symbol} ${timeframe}`);
  }
  
  // Calculate gap count (missing candles)
  const totalRecords = rows.reduce((sum, r) => sum + parseInt(r.count || "0"), 0);
  const minTs = Math.min(...rows.map(r => parseInt(r.min_ts || "0")).filter(t => t > 0));
  const maxTs = Math.max(...rows.map(r => parseInt(r.max_ts || "0")).filter(t => t > 0));
  
  const tfMinutes = TF_MINUTES[timeframe] || 1;
  const expectedCandles = minTs && maxTs ? Math.floor((maxTs - minTs) / (tfMinutes * 60 * 1000)) + 1 : 0;
  const gapCount = Math.max(0, expectedCandles - totalRecords);
  
  if (gapCount > expectedCandles * 0.05) {
    warnings.push(`WARNING: ${gapCount} missing candles (${(gapCount / expectedCandles * 100).toFixed(1)}% gaps)`);
  }
  
  return {
    valid: !symbolMismatch && !timeframeMismatch && duplicateTimestamps === 0,
    symbol,
    timeframe,
    totalRecords,
    distinctSymbols,
    distinctTimeframes,
    symbolMismatch,
    timeframeMismatch,
    duplicateTimestamps,
    gapCount,
    timestampRange: { min: minTs || 0, max: maxTs || 0 },
    warnings,
  };
}

// Add metadata to export responses for Python-side validation
export interface ExportMetadata {
  exportTimestamp: number;
  requestedSymbol: string;
  requestedTimeframe: string;
  actualRecordCount: number;
  integrityCheck: DataIntegrityReport;
  checksum: string;
}

function computeChecksum(data: any[]): string {
  // Simple checksum: hash of first/last timestamps + count
  if (data.length === 0) return "empty";
  const first = data[0]?.timestamp || 0;
  const last = data[data.length - 1]?.timestamp || 0;
  return `${first}-${last}-${data.length}`;
}

// Institution-grade horizon-specific configuration
export const HORIZON_CONFIG = {
  h15: {
    bars: 15,
    purpose: "active_trades",
    minEdge: 0.0015,           // 15 bps minimum edge after costs
    minConfidence: 1.25,        // μ/σ ratio threshold
    weight: 0.5,                // Training/prediction weight
    maxHoldBars: 15,            // Time stop: forced exit at horizon
  },
  h60: {
    bars: 60,
    purpose: "swing_intraday",
    minEdge: 0.0025,           // 25 bps minimum edge after costs
    minConfidence: 1.10,        // μ/σ ratio threshold
    weight: 0.35,               // Training/prediction weight
    maxHoldBars: 60,            // Time stop: forced exit at horizon
  },
  h240: {
    bars: 240,
    purpose: "trend_filter",
    minEdge: 0.0040,           // 40 bps minimum edge after costs
    minConfidence: 0.90,        // μ/σ ratio threshold (trend is slower)
    weight: 0.15,               // Confirmation only, not primary trading
    maxHoldBars: 240,           // Time stop: forced exit at horizon
  },
};

// NO-TRADE conditions (critical for precision)
export const NO_TRADE_CONDITIONS = {
  // Edge dead zone: if |edge| < threshold, signal is noise
  deadZoneThreshold: 0.0015,    // ±15 bps dead zone
  
  // Uncertainty spike: if σ > 95th percentile, skip
  uncertaintyPercentile: 0.95,
  maxUncertaintyMultiplier: 2.5, // σ > 2.5x median = panic
  
  // Horizon disagreement: if μ_15 and μ_60 disagree in sign, skip
  horizonDisagreementVeto: true,
  
  // Loss streak: if recent losses >= threshold, reduce or skip
  maxLossStreak: 3,
  lossStreakSizeReduction: 0.5, // Cut size by 50% after loss streak
  
  // Funding rate flip: aggressive funding change = caution
  fundingFlipWindow: 4,         // bars
  fundingFlipThreshold: 0.001,  // 10 bps flip is aggressive
};

export function getGPUTrainerConfig() {
  return {
    architecture: {
      type: "transformer_encoder",
      numLayers: 6,
      dModel: 256,
      numHeads: 8,
      dFf: 1024,
      dropout: 0.1,
      seqLen: 256,
      numFeatures: FEATURE_SPECS.length,
    },
    training: {
      batchSize: 128,
      learningRate: 2e-4,
      weightDecay: 0.01,
      epochs: 30,
      warmupSteps: 1000,
      gradientClip: 1.0,
      mixedPrecision: true,
      earlyStopping: {
        patience: 5,
        metric: "val_pnl",
        mode: "max",
      },
    },
    loss: {
      returnLoss: "huber",
      returnWeight: 0.5,
      directionalLoss: "bce",
      directionalWeight: 0.2,
      quantiles: [0.1, 0.5, 0.9],
      quantileWeight: 0.15,
      tradeWorthyLoss: "bce",
      tradeWorthyWeight: 0.15,
    },
    horizons: HORIZON_CONFIG,
    noTradeConditions: NO_TRADE_CONDITIONS,
    thresholds: {
      // Deprecated: Use horizons.hX.minEdge instead
      minReturnForTrade: 0.0015,
      minConfidenceForTrade: 1.25,
      maxUncertaintyForTrade: 0.02,
    },
    tradingCosts: TRADING_COSTS,
    sampleWeighting: {
      enabled: true,
      moveWeightPower: 0.5,
      volatilityBoost: 1.5,
      chopPenalty: 0.3,
      minWeight: 0.1,
      maxWeight: 5.0,
    },
    tradeWorthyLabeling: {
      enabled: true,
      minEdge: 0.001,
      cleanMoveThreshold: 0.5,
    },
    rtx4070: {
      optimalBatchSize: 128,
      maxSeqLen: 512,
      enableTF32: true,
      cudnnBenchmark: true,
    },
  };
}

export interface EnhancedLabels {
  rawReturns: number[];
  costAdjustedEdges: number[];
  directions: number[];
  tradeWorthy: number[];
  sampleWeight: number;
}

// Get horizon-specific minimum edge threshold
function getMinEdgeForHorizon(horizon: number): number {
  if (horizon <= 15) return HORIZON_CONFIG.h15.minEdge;
  if (horizon <= 60) return HORIZON_CONFIG.h60.minEdge;
  return HORIZON_CONFIG.h240.minEdge;
}

// Get horizon-specific minimum confidence threshold
function getMinConfidenceForHorizon(horizon: number): number {
  if (horizon <= 15) return HORIZON_CONFIG.h15.minConfidence;
  if (horizon <= 60) return HORIZON_CONFIG.h60.minConfidence;
  return HORIZON_CONFIG.h240.minConfidence;
}

export function getEnhancedLabels(
  closes: number[],
  highs: number[],
  lows: number[],
  idx: number,
  horizons: number[],
  volatility20: number = 0.01,
  costs: typeof TRADING_COSTS = TRADING_COSTS
): EnhancedLabels {
  const rawReturns: number[] = [];
  const costAdjustedEdges: number[] = [];
  const directions: number[] = [];
  const tradeWorthy: number[] = [];
  
  let maxAbsReturn = 0;
  
  for (const h of horizons) {
    const futureIdx = idx + h;
    if (futureIdx < closes.length) {
      const currentClose = closes[idx];
      const futureClose = closes[futureIdx];
      const logReturn = Math.log(futureClose / currentClose);
      
      // Use horizon-specific minimum edge threshold
      const horizonMinEdge = getMinEdgeForHorizon(h);
      const edge = Math.abs(logReturn) - costs.totalRoundTrip;
      
      rawReturns.push(logReturn);
      costAdjustedEdges.push(edge);
      
      // Direction uses horizon-specific dead zone threshold
      directions.push(logReturn > horizonMinEdge ? 1 : logReturn < -horizonMinEdge ? -1 : 0);
      
      // Calculate clean move ratio (directional excursion analysis)
      let maxFavorable = 0;
      let maxAdverse = 0;
      const exitDir = logReturn >= 0 ? 1 : -1;
      
      for (let i = idx + 1; i <= futureIdx && i < highs.length; i++) {
        if (exitDir >= 0) {
          maxFavorable = Math.max(maxFavorable, (highs[i] - currentClose) / currentClose);
          maxAdverse = Math.max(maxAdverse, (currentClose - lows[i]) / currentClose);
        } else {
          maxFavorable = Math.max(maxFavorable, (currentClose - lows[i]) / currentClose);
          maxAdverse = Math.max(maxAdverse, (highs[i] - currentClose) / currentClose);
        }
      }
      const cleanMoveRatio = (maxFavorable + maxAdverse) > 0 
        ? maxFavorable / (maxFavorable + maxAdverse) 
        : 0;
      
      // Trade-worthiness uses horizon-specific edge threshold
      const isTradeWorthy = edge > horizonMinEdge && cleanMoveRatio > 0.5 ? 1 : 0;
      tradeWorthy.push(isTradeWorthy);
      
      maxAbsReturn = Math.max(maxAbsReturn, Math.abs(logReturn));
    } else {
      rawReturns.push(NaN);
      costAdjustedEdges.push(NaN);
      directions.push(0);
      tradeWorthy.push(0);
    }
  }
  
  // Sample weighting: prioritize high-move samples in volatile regimes
  const moveWeight = Math.pow(maxAbsReturn / 0.01, 0.5);
  const volatilityMultiplier = volatility20 > 0.015 ? 1.5 : volatility20 < 0.005 ? 0.3 : 1.0;
  const rawWeight = moveWeight * volatilityMultiplier;
  const sampleWeight = Math.max(0.1, Math.min(5.0, rawWeight));
  
  return {
    rawReturns,
    costAdjustedEdges,
    directions,
    tradeWorthy,
    sampleWeight,
  };
}

export function getLabels(
  closes: number[],
  idx: number,
  horizons: number[]
): { returns: number[]; directions: number[] } {
  const returns: number[] = [];
  const directions: number[] = [];
  
  for (const h of horizons) {
    const futureIdx = idx + h;
    if (futureIdx < closes.length) {
      const currentClose = closes[idx];
      const futureClose = closes[futureIdx];
      const logReturn = Math.log(futureClose / currentClose);
      returns.push(logReturn);
      directions.push(logReturn > 0.0008 ? 1 : logReturn < -0.0008 ? -1 : 0);
    } else {
      returns.push(NaN);
      directions.push(0);
    }
  }
  
  return { returns, directions };
}
