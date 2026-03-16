import { db } from "./db";
import { candles as candlesTable } from "@shared/schema";
import { eq, and, desc } from "drizzle-orm";
import { TRADING_SYMBOLS } from "@shared/symbols";

export type ChopTier = "TRENDING" | "SOFT_CHOP" | "HARD_CHOP";

export interface MarketRegimeState {
  symbol: string;
  adx: number;
  chopIndex: number;
  bbw: number;
  tier: ChopTier;
  updatedAt: number;
}

interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
}

const ADX_HARD_BLOCK = 15;
const ADX_SOFT_ZONE = 25;
const CACHE_TTL_MS = 15 * 60 * 1000;

const regimeCache = new Map<string, MarketRegimeState>();

export function calcADX(candles: Candle[], period: number = 14): number {
  if (candles.length < 2 * period + 1) return 0;

  const plusDM: number[] = [];
  const minusDM: number[] = [];
  const trueRanges: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const h = candles[i].high;
    const l = candles[i].low;
    const ph = candles[i - 1].high;
    const pl = candles[i - 1].low;
    const pc = candles[i - 1].close;

    const upMove = h - ph;
    const downMove = pl - l;

    plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);
    trueRanges.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
  }

  let smoothTR = 0;
  let smoothPlusDM = 0;
  let smoothMinusDM = 0;

  for (let i = 0; i < period; i++) {
    smoothTR += trueRanges[i];
    smoothPlusDM += plusDM[i];
    smoothMinusDM += minusDM[i];
  }

  const dxSeries: number[] = [];

  for (let i = period; i < trueRanges.length; i++) {
    if (i === period) {
      // nothing — use the initial sums
    } else {
      smoothTR = smoothTR - smoothTR / period + trueRanges[i];
      smoothPlusDM = smoothPlusDM - smoothPlusDM / period + plusDM[i];
      smoothMinusDM = smoothMinusDM - smoothMinusDM / period + minusDM[i];
    }

    if (smoothTR === 0) {
      dxSeries.push(0);
      continue;
    }

    const pdi = (smoothPlusDM / smoothTR) * 100;
    const mdi = (smoothMinusDM / smoothTR) * 100;
    const diSum = pdi + mdi;

    if (diSum === 0) {
      dxSeries.push(0);
      continue;
    }

    dxSeries.push((Math.abs(pdi - mdi) / diSum) * 100);
  }

  if (dxSeries.length === 0) return 0;
  if (dxSeries.length < period) {
    return dxSeries.reduce((a, b) => a + b, 0) / dxSeries.length;
  }

  let adx = 0;
  for (let i = 0; i < period; i++) {
    adx += dxSeries[i];
  }
  adx /= period;

  for (let i = period; i < dxSeries.length; i++) {
    adx = (adx * (period - 1) + dxSeries[i]) / period;
  }

  return adx;
}

export function calcChopIndex(candles: Candle[], period: number = 14): number {
  if (candles.length < period + 1) return 50;
  const recent = candles.slice(-period);
  const prev = candles.slice(-(period + 1));

  const trs: number[] = [];
  for (let i = 1; i < prev.length; i++) {
    trs.push(
      Math.max(
        prev[i].high - prev[i].low,
        Math.abs(prev[i].high - prev[i - 1].close),
        Math.abs(prev[i].low - prev[i - 1].close)
      )
    );
  }

  const atrSum = trs.slice(0, period).reduce((a, b) => a + b, 0);
  const high14 = Math.max(...recent.map((c) => c.high));
  const low14 = Math.min(...recent.map((c) => c.low));
  const range = high14 - low14;

  if (range === 0) return 100;
  return Math.min(100, (100 * Math.log10(atrSum / range)) / Math.log10(period));
}

export function calcBBW(candles: Candle[], period: number = 20): number {
  if (candles.length < period) return 1;
  const closes = candles.slice(-period).map((c) => c.close);
  const mid = closes.reduce((a, b) => a + b, 0) / period;
  const std = Math.sqrt(
    closes.reduce((sum, x) => sum + (x - mid) ** 2, 0) / period
  );
  return mid > 0 ? ((2 * std) / mid) * 100 : 1;
}

export function determineTier(adx: number): ChopTier {
  if (adx < ADX_HARD_BLOCK) return "HARD_CHOP";
  if (adx < ADX_SOFT_ZONE) return "SOFT_CHOP";
  return "TRENDING";
}

export async function getSymbolRegimeState(
  symbol: string
): Promise<MarketRegimeState> {
  const cached = regimeCache.get(symbol);
  if (cached && Date.now() - cached.updatedAt < CACHE_TTL_MS) {
    return cached;
  }

  const rows = await db
    .select({
      open: candlesTable.open,
      high: candlesTable.high,
      low: candlesTable.low,
      close: candlesTable.close,
    })
    .from(candlesTable)
    .where(
      and(eq(candlesTable.symbol, symbol), eq(candlesTable.timeframe, "15m"))
    )
    .orderBy(desc(candlesTable.timestamp))
    .limit(50);

  const candles: Candle[] = rows
    .reverse()
    .map((r) => ({
      open: Number(r.open),
      high: Number(r.high),
      low: Number(r.low),
      close: Number(r.close),
    }));

  const adx = calcADX(candles);
  const chopIndex = calcChopIndex(candles);
  const bbw = calcBBW(candles);
  const tier = determineTier(adx);

  const state: MarketRegimeState = {
    symbol,
    adx,
    chopIndex,
    bbw,
    tier,
    updatedAt: Date.now(),
  };

  regimeCache.set(symbol, state);
  return state;
}

export async function getAllRegimeStates(): Promise<MarketRegimeState[]> {
  const results = await Promise.all(
    TRADING_SYMBOLS.map((sym) => getSymbolRegimeState(sym))
  );
  return results;
}

export function getChopGateDecision(
  tier: ChopTier,
  side: "LONG" | "SHORT",
  v5Score: number
): {
  blocked: boolean;
  reason: string;
  adjustedLeverageMult: number;
  adjustedThreshold: number;
} {
  if (tier === "HARD_CHOP") {
    return {
      blocked: true,
      reason: `CHOP_HARD_BLOCK: ADX < ${ADX_HARD_BLOCK} — no directional trend detected`,
      adjustedLeverageMult: 0,
      adjustedThreshold: 1,
    };
  }

  if (tier === "SOFT_CHOP") {
    const softThreshold = 0.62;
    if (v5Score < softThreshold) {
      return {
        blocked: true,
        reason: `CHOP_SOFT_BLOCK: ADX ${ADX_HARD_BLOCK}-${ADX_SOFT_ZONE}, v5_score ${v5Score.toFixed(3)} < raised threshold ${softThreshold}`,
        adjustedLeverageMult: 0.4,
        adjustedThreshold: softThreshold,
      };
    }
    return {
      blocked: false,
      reason: `CHOP_SOFT_PASS: v5_score ${v5Score.toFixed(3)} >= ${softThreshold} (throttled leverage)`,
      adjustedLeverageMult: 0.4,
      adjustedThreshold: softThreshold,
    };
  }

  return {
    blocked: false,
    reason: "TRENDING: no chop protection active",
    adjustedLeverageMult: 1.0,
    adjustedThreshold: 0,
  };
}

export function clearRegimeCache(): void {
  regimeCache.clear();
}
