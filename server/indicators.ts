import type { Candle } from "@shared/schema";

export interface IndicatorResult {
  name: string;
  value: number;
  signal: "bullish" | "bearish" | "neutral";
  strength: number;
  description: string;
}

export interface TechnicalIndicators {
  rsi: IndicatorResult;
  macd: IndicatorResult & { histogram: number; signal: "bullish" | "bearish" | "neutral"; macdLine: number; signalLine: number };
  bollingerBands: IndicatorResult & { upper: number; middle: number; lower: number; percentB: number };
  obv: IndicatorResult;
  vwap: IndicatorResult;
  atr: IndicatorResult;
  adx: IndicatorResult & { plusDI: number; minusDI: number };
  stochastic: IndicatorResult & { k: number; d: number };
  ema: { ema9: number; ema21: number; ema50: number; ema200: number };
  supportResistance: { supports: number[]; resistances: number[] };
  volumeProfile: { highVolumeZones: number[]; pocPrice: number };
}

export function calculateRSI(candles: Candle[], period: number = 14): IndicatorResult {
  if (candles.length < period + 1) {
    return { name: "RSI", value: 50, signal: "neutral", strength: 0, description: "Insufficient data" };
  }

  let gains = 0;
  let losses = 0;

  for (let i = candles.length - period; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;
    if (change > 0) gains += change;
    else losses -= change;
  }

  const avgGain = gains / period;
  const avgLoss = losses / period;
  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = 0;
  let description = `RSI at ${rsi.toFixed(1)}`;

  if (rsi < 30) {
    signal = "bullish";
    strength = (30 - rsi) / 30;
    description = `Oversold at ${rsi.toFixed(1)} - potential bounce`;
  } else if (rsi > 70) {
    signal = "bearish";
    strength = (rsi - 70) / 30;
    description = `Overbought at ${rsi.toFixed(1)} - potential pullback`;
  } else if (rsi > 50) {
    signal = "bullish";
    strength = (rsi - 50) / 40;
    description = `Bullish momentum at ${rsi.toFixed(1)}`;
  } else {
    signal = "bearish";
    strength = (50 - rsi) / 40;
    description = `Bearish momentum at ${rsi.toFixed(1)}`;
  }

  return { name: "RSI", value: rsi, signal, strength, description };
}

export function calculateMACD(candles: Candle[]): IndicatorResult & { histogram: number; macdLine: number; signalLine: number } {
  if (candles.length < 26) {
    return { name: "MACD", value: 0, signal: "neutral", strength: 0, description: "Insufficient data", histogram: 0, macdLine: 0, signalLine: 0 };
  }

  const closes = candles.map(c => c.close);
  
  const ema12 = calculateEMA(closes, 12);
  const ema26 = calculateEMA(closes, 26);
  const macdLine = ema12 - ema26;
  
  const macdHistory: number[] = [];
  for (let i = 25; i < closes.length; i++) {
    const shortEma = calculateEMAAt(closes, 12, i);
    const longEma = calculateEMAAt(closes, 26, i);
    macdHistory.push(shortEma - longEma);
  }
  
  const signalLine = macdHistory.length >= 9 
    ? calculateEMA(macdHistory, 9) 
    : macdLine;
  
  const histogram = macdLine - signalLine;

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = 0;
  let description = "MACD neutral";

  if (histogram > 0 && macdLine > signalLine) {
    signal = "bullish";
    strength = Math.min(1, Math.abs(histogram) / (closes[closes.length - 1] * 0.001));
    description = histogram > 0 && macdHistory.length > 1 && histogram > macdHistory[macdHistory.length - 2] - (macdHistory.length >= 11 ? calculateEMA(macdHistory.slice(0, -1), 9) : macdHistory[macdHistory.length - 2])
      ? "MACD bullish momentum increasing"
      : "MACD bullish crossover";
  } else if (histogram < 0 && macdLine < signalLine) {
    signal = "bearish";
    strength = Math.min(1, Math.abs(histogram) / (closes[closes.length - 1] * 0.001));
    description = "MACD bearish";
  }

  return { name: "MACD", value: macdLine, signal, strength, description, histogram, macdLine, signalLine };
}

export function calculateBollingerBands(candles: Candle[], period: number = 20, stdDev: number = 2): IndicatorResult & { upper: number; middle: number; lower: number; percentB: number } {
  if (candles.length < period) {
    const currentPrice = candles[candles.length - 1]?.close ?? 0;
    return { name: "Bollinger Bands", value: currentPrice, signal: "neutral", strength: 0, description: "Insufficient data", upper: currentPrice, middle: currentPrice, lower: currentPrice, percentB: 0.5 };
  }

  const closes = candles.slice(-period).map(c => c.close);
  const middle = closes.reduce((a, b) => a + b, 0) / period;
  
  const squaredDiffs = closes.map(c => Math.pow(c - middle, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
  const std = Math.sqrt(variance);
  
  const upper = middle + stdDev * std;
  const lower = middle - stdDev * std;
  
  const currentPrice = candles[candles.length - 1].close;
  const percentB = (currentPrice - lower) / (upper - lower);

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = 0;
  let description = "Price within bands";

  if (percentB < 0) {
    signal = "bullish";
    strength = Math.min(1, -percentB);
    description = "Price below lower band - oversold";
  } else if (percentB > 1) {
    signal = "bearish";
    strength = Math.min(1, percentB - 1);
    description = "Price above upper band - overbought";
  } else if (percentB < 0.2) {
    signal = "bullish";
    strength = (0.2 - percentB) / 0.2;
    description = "Price near lower band - potential bounce";
  } else if (percentB > 0.8) {
    signal = "bearish";
    strength = (percentB - 0.8) / 0.2;
    description = "Price near upper band - potential rejection";
  }

  return { name: "Bollinger Bands", value: percentB, signal, strength, description, upper, middle, lower, percentB };
}

export function calculateOBV(candles: Candle[]): IndicatorResult {
  if (candles.length < 2) {
    return { name: "OBV", value: 0, signal: "neutral", strength: 0, description: "Insufficient data" };
  }

  let obv = 0;
  const obvHistory: number[] = [0];

  for (let i = 1; i < candles.length; i++) {
    if (candles[i].close > candles[i - 1].close) {
      obv += candles[i].volume;
    } else if (candles[i].close < candles[i - 1].close) {
      obv -= candles[i].volume;
    }
    obvHistory.push(obv);
  }

  const recentObv = obvHistory.slice(-10);
  const obvTrend = recentObv[recentObv.length - 1] - recentObv[0];
  const priceTrend = candles[candles.length - 1].close - candles[candles.length - 10]?.close;

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = 0;
  let description = "OBV stable";

  if (obvTrend > 0 && priceTrend > 0) {
    signal = "bullish";
    strength = 0.6;
    description = "OBV confirms uptrend";
  } else if (obvTrend < 0 && priceTrend < 0) {
    signal = "bearish";
    strength = 0.6;
    description = "OBV confirms downtrend";
  } else if (obvTrend > 0 && priceTrend < 0) {
    signal = "bullish";
    strength = 0.8;
    description = "Bullish divergence - accumulation";
  } else if (obvTrend < 0 && priceTrend > 0) {
    signal = "bearish";
    strength = 0.8;
    description = "Bearish divergence - distribution";
  }

  return { name: "OBV", value: obv, signal, strength, description };
}

export function calculateVWAP(candles: Candle[]): IndicatorResult {
  if (candles.length < 1) {
    return { name: "VWAP", value: 0, signal: "neutral", strength: 0, description: "Insufficient data" };
  }

  let cumulativeTPV = 0;
  let cumulativeVolume = 0;

  for (const candle of candles) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    cumulativeTPV += typicalPrice * candle.volume;
    cumulativeVolume += candle.volume;
  }

  const vwap = cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : 0;
  const currentPrice = candles[candles.length - 1].close;
  const deviation = ((currentPrice - vwap) / vwap) * 100;

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = Math.min(1, Math.abs(deviation) / 2);
  let description = `Price at VWAP`;

  if (currentPrice > vwap) {
    signal = "bullish";
    description = `Price ${deviation.toFixed(2)}% above VWAP`;
  } else if (currentPrice < vwap) {
    signal = "bearish";
    description = `Price ${Math.abs(deviation).toFixed(2)}% below VWAP`;
  }

  return { name: "VWAP", value: vwap, signal, strength, description };
}

export function calculateATR(candles: Candle[], period: number = 14): IndicatorResult {
  if (candles.length < period + 1) {
    return { name: "ATR", value: 0, signal: "neutral", strength: 0, description: "Insufficient data" };
  }

  let atrSum = 0;
  for (let i = candles.length - period; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1]?.close ?? candles[i].open;
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    atrSum += tr;
  }
  
  const atr = atrSum / period;
  const currentPrice = candles[candles.length - 1].close;
  const volatilityPercent = (atr / currentPrice) * 100;

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = Math.min(1, volatilityPercent / 3);
  let description = `ATR: $${atr.toFixed(2)} (${volatilityPercent.toFixed(2)}% volatility)`;

  if (volatilityPercent > 2) {
    description = `High volatility: ${volatilityPercent.toFixed(2)}% - wider stops needed`;
  } else if (volatilityPercent < 0.5) {
    description = `Low volatility: ${volatilityPercent.toFixed(2)}% - breakout potential`;
  }

  return { name: "ATR", value: atr, signal, strength, description };
}

export function calculateADX(candles: Candle[], period: number = 14): IndicatorResult & { plusDI: number; minusDI: number } {
  if (candles.length < period * 2) {
    return { name: "ADX", value: 0, signal: "neutral", strength: 0, description: "Insufficient data", plusDI: 0, minusDI: 0 };
  }

  let plusDMSum = 0;
  let minusDMSum = 0;
  let trSum = 0;

  for (let i = candles.length - period; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevHigh = candles[i - 1].high;
    const prevLow = candles[i - 1].low;
    const prevClose = candles[i - 1].close;

    const plusDM = Math.max(0, high - prevHigh);
    const minusDM = Math.max(0, prevLow - low);
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));

    if (plusDM > minusDM) {
      plusDMSum += plusDM;
    } else if (minusDM > plusDM) {
      minusDMSum += minusDM;
    }
    trSum += tr;
  }

  const plusDI = trSum > 0 ? (plusDMSum / trSum) * 100 : 0;
  const minusDI = trSum > 0 ? (minusDMSum / trSum) * 100 : 0;
  const dx = plusDI + minusDI > 0 ? Math.abs(plusDI - minusDI) / (plusDI + minusDI) * 100 : 0;
  const adx = dx;

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = adx / 100;
  let description = `ADX: ${adx.toFixed(1)} - `;

  if (adx > 40) {
    description += "Strong trend";
    signal = plusDI > minusDI ? "bullish" : "bearish";
  } else if (adx > 25) {
    description += "Moderate trend";
    signal = plusDI > minusDI ? "bullish" : "bearish";
  } else if (adx > 20) {
    description += "Weak trend";
  } else {
    description += "No trend (ranging)";
  }

  return { name: "ADX", value: adx, signal, strength, description, plusDI, minusDI };
}

export function calculateStochastic(candles: Candle[], kPeriod: number = 14, dPeriod: number = 3): IndicatorResult & { k: number; d: number } {
  if (candles.length < kPeriod) {
    return { name: "Stochastic", value: 50, signal: "neutral", strength: 0, description: "Insufficient data", k: 50, d: 50 };
  }

  const recentCandles = candles.slice(-kPeriod);
  const highest = Math.max(...recentCandles.map(c => c.high));
  const lowest = Math.min(...recentCandles.map(c => c.low));
  const currentClose = candles[candles.length - 1].close;

  const k = highest !== lowest ? ((currentClose - lowest) / (highest - lowest)) * 100 : 50;
  
  const kValues: number[] = [];
  for (let i = kPeriod; i <= candles.length; i++) {
    const slice = candles.slice(i - kPeriod, i);
    const h = Math.max(...slice.map(c => c.high));
    const l = Math.min(...slice.map(c => c.low));
    const c = slice[slice.length - 1].close;
    kValues.push(h !== l ? ((c - l) / (h - l)) * 100 : 50);
  }
  
  const d = kValues.slice(-dPeriod).reduce((a, b) => a + b, 0) / dPeriod;

  let signal: "bullish" | "bearish" | "neutral" = "neutral";
  let strength = 0;
  let description = "Stochastic neutral";

  if (k < 20 && d < 20) {
    signal = "bullish";
    strength = (20 - Math.min(k, d)) / 20;
    description = "Oversold - potential bounce";
  } else if (k > 80 && d > 80) {
    signal = "bearish";
    strength = (Math.max(k, d) - 80) / 20;
    description = "Overbought - potential pullback";
  } else if (k > d && k < 80) {
    signal = "bullish";
    strength = 0.4;
    description = "Bullish momentum";
  } else if (k < d && k > 20) {
    signal = "bearish";
    strength = 0.4;
    description = "Bearish momentum";
  }

  return { name: "Stochastic", value: k, signal, strength, description, k, d };
}

export function calculateEMAs(candles: Candle[]): { ema9: number; ema21: number; ema50: number; ema200: number } {
  const closes = candles.map(c => c.close);
  return {
    ema9: closes.length >= 9 ? calculateEMA(closes, 9) : closes[closes.length - 1],
    ema21: closes.length >= 21 ? calculateEMA(closes, 21) : closes[closes.length - 1],
    ema50: closes.length >= 50 ? calculateEMA(closes, 50) : closes[closes.length - 1],
    ema200: closes.length >= 200 ? calculateEMA(closes, 200) : closes[closes.length - 1],
  };
}

export function findSupportResistance(candles: Candle[]): { supports: number[]; resistances: number[] } {
  if (candles.length < 20) {
    return { supports: [], resistances: [] };
  }

  const pivots: { price: number; type: "high" | "low" }[] = [];
  
  for (let i = 2; i < candles.length - 2; i++) {
    const isSwingHigh = candles[i].high > candles[i - 1].high && 
                        candles[i].high > candles[i - 2].high &&
                        candles[i].high > candles[i + 1].high &&
                        candles[i].high > candles[i + 2].high;
    
    const isSwingLow = candles[i].low < candles[i - 1].low && 
                       candles[i].low < candles[i - 2].low &&
                       candles[i].low < candles[i + 1].low &&
                       candles[i].low < candles[i + 2].low;
    
    if (isSwingHigh) pivots.push({ price: candles[i].high, type: "high" });
    if (isSwingLow) pivots.push({ price: candles[i].low, type: "low" });
  }

  const currentPrice = candles[candles.length - 1].close;
  const supports = pivots
    .filter(p => p.type === "low" && p.price < currentPrice)
    .map(p => p.price)
    .sort((a, b) => b - a)
    .slice(0, 3);
  
  const resistances = pivots
    .filter(p => p.type === "high" && p.price > currentPrice)
    .map(p => p.price)
    .sort((a, b) => a - b)
    .slice(0, 3);

  return { supports, resistances };
}

export function calculateVolumeProfile(candles: Candle[]): { highVolumeZones: number[]; pocPrice: number } {
  if (candles.length < 10) {
    return { highVolumeZones: [], pocPrice: candles[candles.length - 1]?.close ?? 0 };
  }

  const priceVolume: Map<number, number> = new Map();
  const priceStep = (Math.max(...candles.map(c => c.high)) - Math.min(...candles.map(c => c.low))) / 50;

  for (const candle of candles) {
    const avgPrice = (candle.high + candle.low) / 2;
    const bucket = Math.round(avgPrice / priceStep) * priceStep;
    priceVolume.set(bucket, (priceVolume.get(bucket) ?? 0) + candle.volume);
  }

  const sortedByVolume = Array.from(priceVolume.entries()).sort((a, b) => b[1] - a[1]);
  const pocPrice = sortedByVolume[0]?.[0] ?? candles[candles.length - 1].close;
  const highVolumeZones = sortedByVolume.slice(0, 5).map(([price]) => price);

  return { highVolumeZones, pocPrice };
}

export function getAllIndicators(candles: Candle[]): TechnicalIndicators {
  return {
    rsi: calculateRSI(candles),
    macd: calculateMACD(candles),
    bollingerBands: calculateBollingerBands(candles),
    obv: calculateOBV(candles),
    vwap: calculateVWAP(candles),
    atr: calculateATR(candles),
    adx: calculateADX(candles),
    stochastic: calculateStochastic(candles),
    ema: calculateEMAs(candles),
    supportResistance: findSupportResistance(candles),
    volumeProfile: calculateVolumeProfile(candles),
  };
}

function calculateEMA(data: number[], period: number): number {
  if (data.length === 0) return 0;
  if (data.length < period) return data[data.length - 1];
  
  const multiplier = 2 / (period + 1);
  let ema = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  
  for (let i = period; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

function calculateEMAAt(data: number[], period: number, endIndex: number): number {
  if (endIndex < period) return data[endIndex];
  
  const slice = data.slice(0, endIndex + 1);
  return calculateEMA(slice, period);
}

export function calculateMultiTimeframeScore(mtf: { m5: Candle[]; m15: Candle[]; h1: Candle[]; h4: Candle[] }): {
  score: number;
  direction: "bullish" | "bearish" | "neutral";
  alignment: number;
  details: { timeframe: string; trend: "up" | "down" | "neutral"; weight: number }[];
} {
  const details: { timeframe: string; trend: "up" | "down" | "neutral"; weight: number }[] = [];
  
  const analyzeTrend = (candles: Candle[]): "up" | "down" | "neutral" => {
    if (candles.length < 20) return "neutral";
    const ema20 = calculateEMA(candles.map(c => c.close), 20);
    const currentPrice = candles[candles.length - 1].close;
    const priceChange = (candles[candles.length - 1].close - candles[candles.length - 10].close) / candles[candles.length - 10].close;
    
    if (currentPrice > ema20 && priceChange > 0.002) return "up";
    if (currentPrice < ema20 && priceChange < -0.002) return "down";
    return "neutral";
  };

  const m5Trend = analyzeTrend(mtf.m5);
  const m15Trend = analyzeTrend(mtf.m15);
  const h1Trend = analyzeTrend(mtf.h1);
  const h4Trend = analyzeTrend(mtf.h4);

  details.push({ timeframe: "5m", trend: m5Trend, weight: 0.15 });
  details.push({ timeframe: "15m", trend: m15Trend, weight: 0.25 });
  details.push({ timeframe: "1h", trend: h1Trend, weight: 0.30 });
  details.push({ timeframe: "4h", trend: h4Trend, weight: 0.30 });

  let bullishScore = 0;
  let bearishScore = 0;

  for (const d of details) {
    if (d.trend === "up") bullishScore += d.weight;
    else if (d.trend === "down") bearishScore += d.weight;
  }

  const score = bullishScore - bearishScore;
  const alignment = Math.abs(score);
  const direction = score > 0.2 ? "bullish" : score < -0.2 ? "bearish" : "neutral";

  return { score, direction, alignment, details };
}
