import type { Candle } from "@shared/schema";

export interface FeatureVector {
  timestamp: number;
  returns1: number;
  returns2: number;
  returns4: number;
  returns8: number;
  ema20: number;
  ema50: number;
  ema20Slope: number;
  ema50Slope: number;
  emaDistance: number;
  breakoutDistanceHigh: number;
  breakoutDistanceLow: number;
  efficiencyRatio: number;
  atr14: number;
  volatility: number;
  bollingerWidth: number;
  volatilityRegime: "low" | "medium" | "high";
  rsi14: number;
  macd: number;
  macdSignal: number;
  macdHist: number;
  adx: number;
  plusDi: number;
  minusDi: number;
  stochK: number;
  stochD: number;
  obv: number;
  obvSlope: number;
  kalmanFast: number;
  kalmanSlow: number;
  kalmanSpread: number;
  kalmanRegime: "bull" | "bear" | "chop";
  pricePosition: number;
  trendStrength: number;
  momentum: number;
  volumeRatio: number;
  priceVelocity: number;
  priceAcceleration: number;
  embedding: number[];
}

function ema(data: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const result: number[] = [];
  let prevEma = data[0];
  
  for (let i = 0; i < data.length; i++) {
    if (i === 0) {
      result.push(data[0]);
    } else {
      prevEma = data[i] * k + prevEma * (1 - k);
      result.push(prevEma);
    }
  }
  return result;
}

function sma(data: number[], period: number): number[] {
  const result: number[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(data[i]);
    } else {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
  }
  return result;
}

function calculateATR(candles: Candle[], period: number = 14): number[] {
  const trs: number[] = [];
  
  for (let i = 0; i < candles.length; i++) {
    if (i === 0) {
      trs.push(candles[i].high - candles[i].low);
    } else {
      const tr = Math.max(
        candles[i].high - candles[i].low,
        Math.abs(candles[i].high - candles[i - 1].close),
        Math.abs(candles[i].low - candles[i - 1].close)
      );
      trs.push(tr);
    }
  }
  
  return ema(trs, period);
}

function calculateRSI(closes: number[], period: number = 14): number[] {
  const result: number[] = [];
  let avgGain = 0;
  let avgLoss = 0;
  
  for (let i = 0; i < closes.length; i++) {
    if (i === 0) {
      result.push(50);
      continue;
    }
    
    const change = closes[i] - closes[i - 1];
    const gain = Math.max(0, change);
    const loss = Math.max(0, -change);
    
    if (i <= period) {
      avgGain = (avgGain * (i - 1) + gain) / i;
      avgLoss = (avgLoss * (i - 1) + loss) / i;
    } else {
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
    }
    
    if (avgLoss === 0) {
      result.push(100);
    } else {
      const rs = avgGain / avgLoss;
      result.push(100 - (100 / (1 + rs)));
    }
  }
  return result;
}

function calculateMACD(closes: number[]): { macd: number[]; signal: number[]; histogram: number[] } {
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);
  const macdLine = ema12.map((v, i) => v - ema26[i]);
  const signalLine = ema(macdLine, 9);
  const histogram = macdLine.map((v, i) => v - signalLine[i]);
  
  return { macd: macdLine, signal: signalLine, histogram };
}

function calculateBollingerBands(closes: number[], period: number = 20): { upper: number[]; middle: number[]; lower: number[]; width: number[] } {
  const middle = sma(closes, period);
  const upper: number[] = [];
  const lower: number[] = [];
  const width: number[] = [];
  
  for (let i = 0; i < closes.length; i++) {
    if (i < period - 1) {
      upper.push(closes[i]);
      lower.push(closes[i]);
      width.push(0);
    } else {
      const slice = closes.slice(i - period + 1, i + 1);
      const mean = middle[i];
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
      const std = Math.sqrt(variance);
      upper.push(mean + 2 * std);
      lower.push(mean - 2 * std);
      width.push((upper[i] - lower[i]) / middle[i]);
    }
  }
  
  return { upper, middle, lower, width };
}

function calculateStochastic(candles: Candle[], kPeriod: number = 14, dPeriod: number = 3): { k: number[]; d: number[] } {
  const k: number[] = [];
  
  for (let i = 0; i < candles.length; i++) {
    if (i < kPeriod - 1) {
      k.push(50);
    } else {
      const slice = candles.slice(i - kPeriod + 1, i + 1);
      const highestHigh = Math.max(...slice.map(c => c.high));
      const lowestLow = Math.min(...slice.map(c => c.low));
      const range = highestHigh - lowestLow;
      if (range === 0) {
        k.push(50);
      } else {
        k.push(((candles[i].close - lowestLow) / range) * 100);
      }
    }
  }
  
  const d = sma(k, dPeriod);
  return { k, d };
}

function calculateADX(candles: Candle[], period: number = 14): { adx: number[]; plusDi: number[]; minusDi: number[] } {
  const plusDm: number[] = [];
  const minusDm: number[] = [];
  const tr: number[] = [];
  
  for (let i = 0; i < candles.length; i++) {
    if (i === 0) {
      plusDm.push(0);
      minusDm.push(0);
      tr.push(candles[i].high - candles[i].low);
    } else {
      const upMove = candles[i].high - candles[i - 1].high;
      const downMove = candles[i - 1].low - candles[i].low;
      
      plusDm.push(upMove > downMove && upMove > 0 ? upMove : 0);
      minusDm.push(downMove > upMove && downMove > 0 ? downMove : 0);
      
      tr.push(Math.max(
        candles[i].high - candles[i].low,
        Math.abs(candles[i].high - candles[i - 1].close),
        Math.abs(candles[i].low - candles[i - 1].close)
      ));
    }
  }
  
  const smoothedPlusDm = ema(plusDm, period);
  const smoothedMinusDm = ema(minusDm, period);
  const smoothedTr = ema(tr, period);
  
  const plusDi = smoothedPlusDm.map((v, i) => smoothedTr[i] === 0 ? 0 : (v / smoothedTr[i]) * 100);
  const minusDi = smoothedMinusDm.map((v, i) => smoothedTr[i] === 0 ? 0 : (v / smoothedTr[i]) * 100);
  
  const dx = plusDi.map((v, i) => {
    const sum = v + minusDi[i];
    return sum === 0 ? 0 : (Math.abs(v - minusDi[i]) / sum) * 100;
  });
  
  const adx = ema(dx, period);
  
  return { adx, plusDi, minusDi };
}

function calculateOBV(candles: Candle[]): number[] {
  const obv: number[] = [];
  
  for (let i = 0; i < candles.length; i++) {
    if (i === 0) {
      obv.push(candles[i].volume);
    } else {
      if (candles[i].close > candles[i - 1].close) {
        obv.push(obv[i - 1] + candles[i].volume);
      } else if (candles[i].close < candles[i - 1].close) {
        obv.push(obv[i - 1] - candles[i].volume);
      } else {
        obv.push(obv[i - 1]);
      }
    }
  }
  return obv;
}

function kalmanFilter(data: number[], processNoise: number = 0.01): number[] {
  const result: number[] = [];
  let x = data[0];
  let P = 1;
  const Q = processNoise;
  const R = 0.1;
  
  for (const z of data) {
    const xPrior = x;
    const pPrior = P + Q;
    const K = pPrior / (pPrior + R);
    x = xPrior + K * (z - xPrior);
    P = (1 - K) * pPrior;
    result.push(x);
  }
  return result;
}

function calculateEfficiencyRatio(candles: Candle[], period: number = 10): number[] {
  const result: number[] = [];
  
  for (let i = 0; i < candles.length; i++) {
    if (i < period) {
      result.push(0.5);
    } else {
      const netChange = Math.abs(candles[i].close - candles[i - period].close);
      let totalChange = 0;
      for (let j = i - period + 1; j <= i; j++) {
        totalChange += Math.abs(candles[j].close - candles[j - 1].close);
      }
      result.push(totalChange === 0 ? 0.5 : netChange / totalChange);
    }
  }
  return result;
}

export function computeFeatures(candles: Candle[]): FeatureVector[] {
  if (candles.length < 50) {
    console.warn("Need at least 50 candles for feature computation");
    return [];
  }
  
  const closes = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  const volumes = candles.map(c => c.volume);
  
  const ema20Arr = ema(closes, 20);
  const ema50Arr = ema(closes, 50);
  const atr14Arr = calculateATR(candles, 14);
  const rsi14Arr = calculateRSI(closes, 14);
  const { macd, signal, histogram } = calculateMACD(closes);
  const { upper, lower, width } = calculateBollingerBands(closes, 20);
  const { k: stochK, d: stochD } = calculateStochastic(candles, 14, 3);
  const { adx, plusDi, minusDi } = calculateADX(candles, 14);
  const obvArr = calculateOBV(candles);
  const effRatioArr = calculateEfficiencyRatio(candles, 10);
  
  const kalmanFastArr = kalmanFilter(closes, 0.02);
  const kalmanSlowArr = kalmanFilter(closes, 0.005);
  
  const features: FeatureVector[] = [];
  const startIdx = Math.max(50, candles.length > 250 ? 250 : 50);
  
  for (let i = startIdx; i < candles.length; i++) {
    const price = closes[i];
    const avgVol = volumes.slice(i - 20, i).reduce((a, b) => a + b, 0) / 20;
    
    const high20 = Math.max(...highs.slice(i - 20, i + 1));
    const low20 = Math.min(...lows.slice(i - 20, i + 1));
    
    const atrVal = atr14Arr[i];
    const volRegime: "low" | "medium" | "high" = 
      atrVal / price < 0.002 ? "low" : 
      atrVal / price > 0.005 ? "high" : "medium";
    
    const kalmanSpread = kalmanFastArr[i] - kalmanSlowArr[i];
    const kalmanRegime: "bull" | "bear" | "chop" = 
      kalmanSpread > atrVal * 0.5 ? "bull" :
      kalmanSpread < -atrVal * 0.5 ? "bear" : "chop";
    
    const embedding = [
      (closes[i] - closes[i - 1]) / closes[i - 1],
      (closes[i] - closes[i - 4]) / closes[i - 4],
      rsi14Arr[i] / 100,
      histogram[i] / price,
      adx[i] / 100,
      stochK[i] / 100,
      effRatioArr[i],
      kalmanSpread / atrVal,
    ];
    
    features.push({
      timestamp: candles[i].timestamp,
      returns1: (closes[i] - closes[i - 1]) / closes[i - 1],
      returns2: (closes[i] - closes[i - 2]) / closes[i - 2],
      returns4: (closes[i] - closes[i - 4]) / closes[i - 4],
      returns8: (closes[i] - closes[i - 8]) / closes[i - 8],
      ema20: ema20Arr[i],
      ema50: ema50Arr[i],
      ema20Slope: (ema20Arr[i] - ema20Arr[i - 1]) / ema20Arr[i - 1],
      ema50Slope: (ema50Arr[i] - ema50Arr[i - 1]) / ema50Arr[i - 1],
      emaDistance: (price - ema20Arr[i]) / atrVal,
      breakoutDistanceHigh: (price - high20) / atrVal,
      breakoutDistanceLow: (price - low20) / atrVal,
      efficiencyRatio: effRatioArr[i],
      atr14: atrVal,
      volatility: atrVal / price,
      bollingerWidth: width[i],
      volatilityRegime: volRegime,
      rsi14: rsi14Arr[i],
      macd: macd[i],
      macdSignal: signal[i],
      macdHist: histogram[i],
      adx: adx[i],
      plusDi: plusDi[i],
      minusDi: minusDi[i],
      stochK: stochK[i],
      stochD: stochD[i],
      obv: obvArr[i],
      obvSlope: (obvArr[i] - obvArr[i - 5]) / Math.abs(obvArr[i - 5] || 1),
      kalmanFast: kalmanFastArr[i],
      kalmanSlow: kalmanSlowArr[i],
      kalmanSpread,
      kalmanRegime,
      pricePosition: (price - low20) / (high20 - low20 || 1),
      trendStrength: adx[i],
      momentum: rsi14Arr[i] - 50,
      volumeRatio: volumes[i] / avgVol,
      priceVelocity: (closes[i] - closes[i - 3]) / (3 * atrVal),
      priceAcceleration: ((closes[i] - closes[i - 3]) - (closes[i - 3] - closes[i - 6])) / (3 * atrVal),
      embedding,
    });
  }
  
  return features;
}

export function getLatestFeatures(candles: Candle[]): FeatureVector | null {
  const features = computeFeatures(candles);
  return features.length > 0 ? features[features.length - 1] : null;
}
