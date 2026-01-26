import type { Candle } from "@shared/schema";

export interface CandlestickPattern {
  name: string;
  type: "bullish" | "bearish" | "neutral";
  strength: number;
  description: string;
}

export function detectCandlestickPatterns(candles: Candle[]): CandlestickPattern[] {
  if (candles.length < 5) return [];
  
  const patterns: CandlestickPattern[] = [];
  const c = candles[candles.length - 1];
  const c1 = candles[candles.length - 2];
  const c2 = candles[candles.length - 3];
  
  const bodySize = Math.abs(c.close - c.open);
  const totalRange = c.high - c.low;
  const upperWick = c.high - Math.max(c.open, c.close);
  const lowerWick = Math.min(c.open, c.close) - c.low;
  const isBullish = c.close > c.open;
  const isBearish = c.close < c.open;
  
  const avgBody = candles.slice(-10).reduce((sum, candle) => 
    sum + Math.abs(candle.close - candle.open), 0) / 10;
  const avgRange = candles.slice(-10).reduce((sum, candle) => 
    sum + (candle.high - candle.low), 0) / 10;
  
  if (bodySize < avgBody * 0.3 && totalRange > 0) {
    if (lowerWick > bodySize * 2 && upperWick < bodySize) {
      patterns.push({
        name: "Hammer",
        type: "bullish",
        strength: 0.7,
        description: "Bullish reversal - long lower shadow, small body at top"
      });
    } else if (upperWick > bodySize * 2 && lowerWick < bodySize) {
      patterns.push({
        name: "Shooting Star",
        type: "bearish",
        strength: 0.7,
        description: "Bearish reversal - long upper shadow, small body at bottom"
      });
    } else if (upperWick > bodySize && lowerWick > bodySize) {
      patterns.push({
        name: "Doji",
        type: "neutral",
        strength: 0.5,
        description: "Indecision - open and close nearly equal"
      });
    }
  }
  
  if (bodySize > avgBody * 1.5) {
    if (isBullish) {
      patterns.push({
        name: "Marubozu Bullish",
        type: "bullish",
        strength: 0.8,
        description: "Strong buying pressure - large bullish body, minimal wicks"
      });
    } else {
      patterns.push({
        name: "Marubozu Bearish",
        type: "bearish",
        strength: 0.8,
        description: "Strong selling pressure - large bearish body, minimal wicks"
      });
    }
  }
  
  const c1Body = Math.abs(c1.close - c1.open);
  const c1Bullish = c1.close > c1.open;
  const c1Bearish = c1.close < c1.open;
  
  if (c1Bearish && isBullish && bodySize > c1Body && 
      c.open < c1.close && c.close > c1.open) {
    patterns.push({
      name: "Bullish Engulfing",
      type: "bullish",
      strength: 0.85,
      description: "Strong bullish reversal - current candle engulfs previous"
    });
  }
  
  if (c1Bullish && isBearish && bodySize > c1Body && 
      c.open > c1.close && c.close < c1.open) {
    patterns.push({
      name: "Bearish Engulfing",
      type: "bearish",
      strength: 0.85,
      description: "Strong bearish reversal - current candle engulfs previous"
    });
  }
  
  if (c1Bearish && isBullish && c.close > (c1.open + c1.close) / 2) {
    patterns.push({
      name: "Piercing Line",
      type: "bullish",
      strength: 0.7,
      description: "Bullish reversal - closes above midpoint of previous bearish"
    });
  }
  
  if (c1Bullish && isBearish && c.close < (c1.open + c1.close) / 2) {
    patterns.push({
      name: "Dark Cloud Cover",
      type: "bearish",
      strength: 0.7,
      description: "Bearish reversal - closes below midpoint of previous bullish"
    });
  }
  
  if (candles.length >= 3) {
    const c2Bearish = c2.close < c2.open;
    const c2Bullish = c2.close > c2.open;
    
    if (c2Bearish && Math.abs(c1.close - c1.open) < avgBody * 0.3 && isBullish && c.close > c2.open) {
      patterns.push({
        name: "Morning Star",
        type: "bullish",
        strength: 0.9,
        description: "Strong bullish reversal - three candle pattern"
      });
    }
    
    if (c2Bullish && Math.abs(c1.close - c1.open) < avgBody * 0.3 && isBearish && c.close < c2.open) {
      patterns.push({
        name: "Evening Star",
        type: "bearish",
        strength: 0.9,
        description: "Strong bearish reversal - three candle pattern"
      });
    }
    
    if (c2Bearish && c1Bearish && isBullish && c1.close < c2.close && c.close > c1.open) {
      patterns.push({
        name: "Three Inside Up",
        type: "bullish",
        strength: 0.8,
        description: "Bullish confirmation pattern"
      });
    }
    
    if (c2Bullish && c1Bullish && isBearish && c1.close > c2.close && c.close < c1.open) {
      patterns.push({
        name: "Three Inside Down",
        type: "bearish",
        strength: 0.8,
        description: "Bearish confirmation pattern"
      });
    }
  }
  
  const recentLows = candles.slice(-5).map(x => x.low);
  const recentHighs = candles.slice(-5).map(x => x.high);
  const supportLevel = Math.min(...recentLows);
  const resistanceLevel = Math.max(...recentHighs);
  
  if (c.low <= supportLevel * 1.002 && isBullish) {
    patterns.push({
      name: "Support Bounce",
      type: "bullish",
      strength: 0.65,
      description: "Price bounced off recent support level"
    });
  }
  
  if (c.high >= resistanceLevel * 0.998 && isBearish) {
    patterns.push({
      name: "Resistance Rejection",
      type: "bearish",
      strength: 0.65,
      description: "Price rejected at recent resistance level"
    });
  }
  
  return patterns;
}

export interface VolumeProfile {
  buyVolume: number;
  sellVolume: number;
  volumeRatio: number;
  volumeTrend: "increasing" | "decreasing" | "stable";
  volumeAnomaly: boolean;
  climaxVolume: boolean;
}

export function analyzeVolumeProfile(candles: Candle[]): VolumeProfile {
  if (candles.length < 20) {
    return {
      buyVolume: 0,
      sellVolume: 0,
      volumeRatio: 1,
      volumeTrend: "stable",
      volumeAnomaly: false,
      climaxVolume: false,
    };
  }
  
  let buyVol = 0;
  let sellVol = 0;
  
  for (const c of candles.slice(-10)) {
    if (c.close > c.open) {
      buyVol += c.volume;
    } else {
      sellVol += c.volume;
    }
  }
  
  const recentAvg = candles.slice(-5).reduce((s, c) => s + c.volume, 0) / 5;
  const olderAvg = candles.slice(-20, -10).reduce((s, c) => s + c.volume, 0) / 10;
  
  const volumeTrend: "increasing" | "decreasing" | "stable" = 
    recentAvg > olderAvg * 1.3 ? "increasing" :
    recentAvg < olderAvg * 0.7 ? "decreasing" : "stable";
  
  const stdDev = Math.sqrt(
    candles.slice(-20).reduce((sum, c) => sum + Math.pow(c.volume - olderAvg, 2), 0) / 20
  );
  
  const lastVol = candles[candles.length - 1].volume;
  const volumeAnomaly = lastVol > olderAvg + 2 * stdDev;
  const climaxVolume = lastVol > olderAvg * 3;
  
  return {
    buyVolume: buyVol,
    sellVolume: sellVol,
    volumeRatio: sellVol > 0 ? buyVol / sellVol : 2,
    volumeTrend,
    volumeAnomaly,
    climaxVolume,
  };
}

export interface FeatureVector {
  timestamp: number;
  
  // OHLCV data - essential for neural networks to see raw price action
  price: number;         // Current close price
  open: number;          // Current candle open
  high: number;          // Current candle high
  low: number;           // Current candle low
  close: number;         // Current candle close
  volume: number;        // Current candle volume
  normalizedPrice: number;    // Price normalized by ATR (price / ATR)
  normalizedVolume: number;   // Volume normalized by average (volume / avgVolume)
  candleBody: number;         // Body size as % of range: |close - open| / (high - low)
  candleRange: number;        // Range as % of price: (high - low) / close
  
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
  
  // Cross-asset features - ETH, SOL, BNB relative to BTC
  ethBtcCorrelation: number;     // 20-period rolling correlation
  solBtcCorrelation: number;     // 20-period rolling correlation  
  bnbBtcCorrelation: number;     // 20-period rolling correlation
  ethRelativeStrength: number;   // ETH return / BTC return (1 = equal, >1 = ETH outperforming)
  solRelativeStrength: number;   // SOL return / BTC return
  bnbRelativeStrength: number;   // BNB return / BTC return
  ethMomentumDivergence: number; // ETH momentum - BTC momentum (positive = ETH leading)
  solMomentumDivergence: number; // SOL momentum - BTC momentum
  bnbMomentumDivergence: number; // BNB momentum - BTC momentum
  cryptoSectorMomentum: number;  // Average altcoin momentum vs BTC (market breadth)
  
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

// RESEARCH-BACKED: Kalman Filter with optimized parameters
// Q=0.1 (process noise), R=0.1 (measurement noise), P=1000 (initial uncertainty)
// These parameters provide responsive yet stable trend estimation for crypto markets
function kalmanFilter(data: number[], processNoise: number = 0.1): number[] {
  const result: number[] = [];
  let x = data[0];
  let P = 1000;  // High initial uncertainty - key research finding
  const Q = processNoise;  // Default Q=0.1 for responsive tracking
  const R = 0.1;  // Measurement noise - well-calibrated for crypto
  
  for (const z of data) {
    // Prediction step
    const xPrior = x;
    const pPrior = P + Q;
    
    // Update step with Kalman gain
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
  
  // RESEARCH-BACKED: Fast Kalman (Q=0.1) for responsive signals, Slow (Q=0.02) for trend
  const kalmanFastArr = kalmanFilter(closes, 0.1);   // Fast: Q=0.1 for responsive tracking
  const kalmanSlowArr = kalmanFilter(closes, 0.02); // Slow: Q=0.02 for trend filtering
  
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
    const effRatio = effRatioArr[i];
    const kalmanSlope = i > 0 ? (kalmanFastArr[i] - kalmanFastArr[i - 1]) / price : 0;
    
    let kalmanRegime: "bull" | "bear" | "chop" = "chop";
    const spreadThreshold = atrVal * 0.25;
    const strongSpreadThreshold = atrVal * 0.4;
    
    if (kalmanSpread > strongSpreadThreshold) {
      kalmanRegime = "bull";
    } else if (kalmanSpread < -strongSpreadThreshold) {
      kalmanRegime = "bear";
    } else if (kalmanSpread > spreadThreshold && effRatio > 0.30 && kalmanSlope > 0) {
      kalmanRegime = "bull";
    } else if (kalmanSpread < -spreadThreshold && effRatio > 0.30 && kalmanSlope < 0) {
      kalmanRegime = "bear";
    }
    
    // EXPANDED EMBEDDING: 24 features for robust pattern matching
    // Based on research: 100-200 dimensions optimal, 20+ minimum for trading
    const embedding = [
      // Returns at multiple lookbacks (4 features)
      (closes[i] - closes[i - 1]) / closes[i - 1],
      (closes[i] - closes[i - 2]) / closes[i - 2],
      (closes[i] - closes[i - 4]) / closes[i - 4],
      (closes[i] - closes[i - 8]) / closes[i - 8],
      
      // Momentum indicators (4 features)
      rsi14Arr[i] / 100,
      stochK[i] / 100,
      stochD[i] / 100,
      (rsi14Arr[i] - 50) / 50,  // RSI deviation from neutral
      
      // Trend indicators (4 features)
      adx[i] / 100,
      (plusDi[i] - minusDi[i]) / 100,  // DI spread
      histogram[i] / (price * 0.01),   // Normalized MACD histogram
      (macd[i] - signal[i]) / (price * 0.01),  // MACD-Signal divergence
      
      // Volatility features (4 features)
      effRatioArr[i],
      width[i] * 10,  // Bollinger width scaled
      atrVal / price * 100,  // Normalized ATR
      kalmanSpread / atrVal,  // Kalman spread relative to ATR
      
      // Volume features (3 features)
      Math.log1p(volumes[i] / avgVol),  // Log volume ratio
      (obvArr[i] - obvArr[i - 5]) / Math.max(1, Math.abs(obvArr[i - 5])),  // OBV slope
      volumes[i] > avgVol * 1.5 ? 1 : 0,  // High volume flag
      
      // Price position (3 features)
      (price - low20) / (high20 - low20 || 1),  // Position in range
      (price - ema20Arr[i]) / atrVal,  // Distance from EMA20
      (price - ema50Arr[i]) / atrVal,  // Distance from EMA50
      
      // Regime indicators (2 features)
      kalmanRegime === "bull" ? 1 : kalmanRegime === "bear" ? -1 : 0,
      volRegime === "high" ? 1 : volRegime === "low" ? -1 : 0,
    ];
    
    // OHLCV normalized features for neural networks
    const currentCandle = candles[i];
    const candleRange = currentCandle.high - currentCandle.low;
    const candleBodyVal = Math.abs(currentCandle.close - currentCandle.open);
    
    features.push({
      timestamp: candles[i].timestamp,
      
      // Raw OHLCV data
      price: price,
      open: currentCandle.open,
      high: currentCandle.high,
      low: currentCandle.low,
      close: currentCandle.close,
      volume: currentCandle.volume,
      
      // Normalized OHLCV for neural networks
      normalizedPrice: price / atrVal,
      normalizedVolume: volumes[i] / avgVol,
      candleBody: candleRange > 0 ? candleBodyVal / candleRange : 0,
      candleRange: candleRange / price,
      
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
      
      // Cross-asset features - defaults to 0 when cross-asset data not available
      // These are computed by enrichFeaturesWithCrossAsset() when data is present
      ethBtcCorrelation: 0,
      solBtcCorrelation: 0,
      bnbBtcCorrelation: 0,
      ethRelativeStrength: 0,
      solRelativeStrength: 0,
      bnbRelativeStrength: 0,
      ethMomentumDivergence: 0,
      solMomentumDivergence: 0,
      bnbMomentumDivergence: 0,
      cryptoSectorMomentum: 0,
      
      embedding: sanitizeArray(embedding),
    });
  }
  
  // Sanitize all feature vectors to prevent NaN/Infinity propagation
  return features.map(sanitizeFeatureVector);
}

/**
 * FEATURE SANITIZATION: Prevent NaN/Infinity from corrupting ML predictions
 * Replaces invalid values with safe defaults to ensure system stability
 */
function sanitizeValue(val: number, defaultVal: number = 0): number {
  if (!Number.isFinite(val)) return defaultVal;
  return val;
}

function sanitizeArray(arr: number[]): number[] {
  return arr.map(v => sanitizeValue(v, 0));
}

function sanitizeFeatureVector(f: FeatureVector): FeatureVector {
  // CRITICAL: Sanitize price first since other defaults depend on it
  const safePrice = sanitizeValue(f.price, 50000);  // Reasonable BTC default
  const safeAtr = sanitizeValue(f.atr14, safePrice * 0.01);  // Default 1% ATR
  
  return {
    ...f,
    price: safePrice,
    open: sanitizeValue(f.open, safePrice),
    high: sanitizeValue(f.high, safePrice),
    low: sanitizeValue(f.low, safePrice),
    close: sanitizeValue(f.close, safePrice),
    volume: sanitizeValue(f.volume, 0),
    normalizedPrice: sanitizeValue(f.normalizedPrice, 1),
    normalizedVolume: sanitizeValue(f.normalizedVolume, 1),
    candleBody: sanitizeValue(f.candleBody, 0),
    candleRange: sanitizeValue(f.candleRange, 0),
    returns1: sanitizeValue(f.returns1, 0),
    returns2: sanitizeValue(f.returns2, 0),
    returns4: sanitizeValue(f.returns4, 0),
    returns8: sanitizeValue(f.returns8, 0),
    ema20: sanitizeValue(f.ema20, safePrice),
    ema50: sanitizeValue(f.ema50, safePrice),
    ema20Slope: sanitizeValue(f.ema20Slope, 0),
    ema50Slope: sanitizeValue(f.ema50Slope, 0),
    emaDistance: sanitizeValue(f.emaDistance, 0),
    breakoutDistanceHigh: sanitizeValue(f.breakoutDistanceHigh, 0),
    breakoutDistanceLow: sanitizeValue(f.breakoutDistanceLow, 0),
    efficiencyRatio: sanitizeValue(f.efficiencyRatio, 0.5),
    atr14: safeAtr,
    volatility: sanitizeValue(f.volatility, 0),
    bollingerWidth: sanitizeValue(f.bollingerWidth, 0),
    rsi14: sanitizeValue(f.rsi14, 50),
    macd: sanitizeValue(f.macd, 0),
    macdSignal: sanitizeValue(f.macdSignal, 0),
    macdHist: sanitizeValue(f.macdHist, 0),
    adx: sanitizeValue(f.adx, 25),
    plusDi: sanitizeValue(f.plusDi, 25),
    minusDi: sanitizeValue(f.minusDi, 25),
    stochK: sanitizeValue(f.stochK, 50),
    stochD: sanitizeValue(f.stochD, 50),
    obv: sanitizeValue(f.obv, 0),
    obvSlope: sanitizeValue(f.obvSlope, 0),
    kalmanFast: sanitizeValue(f.kalmanFast, safePrice),
    kalmanSlow: sanitizeValue(f.kalmanSlow, safePrice),
    kalmanSpread: sanitizeValue(f.kalmanSpread, 0),
    pricePosition: sanitizeValue(f.pricePosition, 0.5),
    trendStrength: sanitizeValue(f.trendStrength, 25),
    momentum: sanitizeValue(f.momentum, 0),
    volumeRatio: sanitizeValue(f.volumeRatio, 1),
    priceVelocity: sanitizeValue(f.priceVelocity, 0),
    priceAcceleration: sanitizeValue(f.priceAcceleration, 0),
    ethBtcCorrelation: sanitizeValue(f.ethBtcCorrelation, 0),
    solBtcCorrelation: sanitizeValue(f.solBtcCorrelation, 0),
    bnbBtcCorrelation: sanitizeValue(f.bnbBtcCorrelation, 0),
    ethRelativeStrength: sanitizeValue(f.ethRelativeStrength, 1),
    solRelativeStrength: sanitizeValue(f.solRelativeStrength, 1),
    bnbRelativeStrength: sanitizeValue(f.bnbRelativeStrength, 1),
    ethMomentumDivergence: sanitizeValue(f.ethMomentumDivergence, 0),
    solMomentumDivergence: sanitizeValue(f.solMomentumDivergence, 0),
    bnbMomentumDivergence: sanitizeValue(f.bnbMomentumDivergence, 0),
    cryptoSectorMomentum: sanitizeValue(f.cryptoSectorMomentum, 0),
    embedding: sanitizeArray(f.embedding),
  };
}

/**
 * DAIN-style Adaptive Normalization for Non-Stationary Market Data
 * Uses rolling statistics to normalize features based on recent history only
 * This prevents data leakage by never using future data for normalization
 */
export class AdaptiveNormalizer {
  private rollingMean: Map<string, number[]> = new Map();
  private rollingStd: Map<string, number[]> = new Map();
  private readonly windowSize: number;
  private readonly clipValue: number;
  
  constructor(windowSize: number = 100, clipValue: number = 3.0) {
    this.windowSize = windowSize;
    this.clipValue = clipValue; // Clip z-scores beyond this value
  }
  
  /**
   * Update rolling statistics with a new observation
   * Returns normalized value using only past data
   */
  updateAndNormalize(feature: string, value: number): number {
    if (!isFinite(value) || isNaN(value)) return 0;
    
    // Get or initialize rolling windows
    if (!this.rollingMean.has(feature)) {
      this.rollingMean.set(feature, []);
      this.rollingStd.set(feature, []);
    }
    
    const meanWindow = this.rollingMean.get(feature)!;
    const stdWindow = this.rollingStd.get(feature)!;
    
    // Calculate current statistics BEFORE adding new value (forward-only)
    let mean = 0;
    let std = 1;
    
    if (meanWindow.length >= 10) { // Minimum samples for stable statistics
      mean = meanWindow.reduce((a, b) => a + b, 0) / meanWindow.length;
      const variance = meanWindow.reduce((a, b) => a + (b - mean) ** 2, 0) / meanWindow.length;
      std = Math.sqrt(variance) || 1; // Prevent division by zero
    }
    
    // Add new value to rolling window
    meanWindow.push(value);
    if (meanWindow.length > this.windowSize) {
      meanWindow.shift();
    }
    
    // Z-score normalization with clipping
    let normalized = (value - mean) / std;
    normalized = Math.max(-this.clipValue, Math.min(this.clipValue, normalized));
    
    return normalized;
  }
  
  /**
   * Normalize a full feature vector using adaptive statistics
   * Each feature is normalized independently based on its own history
   */
  normalizeFeatureVector(features: FeatureVector): FeatureVector {
    return {
      ...features,
      // Normalize momentum indicators (they can have varying scales)
      rsi14: this.updateAndNormalize('rsi14', features.rsi14),
      macd: this.updateAndNormalize('macd', features.macd),
      macdSignal: this.updateAndNormalize('macdSignal', features.macdSignal),
      macdHist: this.updateAndNormalize('macdHist', features.macdHist),
      stochK: this.updateAndNormalize('stochK', features.stochK),
      stochD: this.updateAndNormalize('stochD', features.stochD),
      momentum: this.updateAndNormalize('momentum', features.momentum),
      
      // Normalize volatility indicators
      atr14: this.updateAndNormalize('atr14', features.atr14),
      volatility: this.updateAndNormalize('volatility', features.volatility),
      bollingerWidth: this.updateAndNormalize('bollingerWidth', features.bollingerWidth),
      
      // Normalize trend indicators
      adx: this.updateAndNormalize('adx', features.adx),
      plusDi: this.updateAndNormalize('plusDi', features.plusDi),
      minusDi: this.updateAndNormalize('minusDi', features.minusDi),
      trendStrength: this.updateAndNormalize('trendStrength', features.trendStrength),
      
      // Normalize returns (already percentage-based but can have outliers)
      returns1: this.updateAndNormalize('returns1', features.returns1),
      returns2: this.updateAndNormalize('returns2', features.returns2),
      returns4: this.updateAndNormalize('returns4', features.returns4),
      returns8: this.updateAndNormalize('returns8', features.returns8),
      
      // Normalize cross-asset features
      ethBtcCorrelation: this.updateAndNormalize('ethBtcCorrelation', features.ethBtcCorrelation),
      solBtcCorrelation: this.updateAndNormalize('solBtcCorrelation', features.solBtcCorrelation),
      bnbBtcCorrelation: this.updateAndNormalize('bnbBtcCorrelation', features.bnbBtcCorrelation),
      ethRelativeStrength: this.updateAndNormalize('ethRelativeStrength', features.ethRelativeStrength),
      solRelativeStrength: this.updateAndNormalize('solRelativeStrength', features.solRelativeStrength),
      bnbRelativeStrength: this.updateAndNormalize('bnbRelativeStrength', features.bnbRelativeStrength),
      ethMomentumDivergence: this.updateAndNormalize('ethMomentumDivergence', features.ethMomentumDivergence),
      solMomentumDivergence: this.updateAndNormalize('solMomentumDivergence', features.solMomentumDivergence),
      bnbMomentumDivergence: this.updateAndNormalize('bnbMomentumDivergence', features.bnbMomentumDivergence),
      cryptoSectorMomentum: this.updateAndNormalize('cryptoSectorMomentum', features.cryptoSectorMomentum),
      
      // Keep price-based features as-is (already normalized by ATR or are raw prices needed for context)
      // Embedding is not normalized (it's already in a learned latent space)
    };
  }
  
  /**
   * Get current statistics for debugging/monitoring
   */
  getStats(): { feature: string; mean: number; std: number; samples: number }[] {
    const stats: { feature: string; mean: number; std: number; samples: number }[] = [];
    
    for (const [feature, values] of this.rollingMean.entries()) {
      if (values.length > 0) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
        stats.push({
          feature,
          mean: Math.round(mean * 1000) / 1000,
          std: Math.round(Math.sqrt(variance) * 1000) / 1000,
          samples: values.length,
        });
      }
    }
    
    return stats;
  }
  
  /**
   * Reset all statistics (e.g., when switching to new training data)
   */
  reset(): void {
    this.rollingMean.clear();
    this.rollingStd.clear();
    console.log('[Adaptive Normalizer] Statistics reset');
  }
}

// Global instance for consistent normalization across components
export const adaptiveNormalizer = new AdaptiveNormalizer(100, 3.0);

/**
 * Compute rolling correlation between two price series
 */
function computeCorrelation(series1: number[], series2: number[], period: number = 20): number {
  if (series1.length < period || series2.length < period) return 0;
  
  const recent1 = series1.slice(-period);
  const recent2 = series2.slice(-period);
  
  const mean1 = recent1.reduce((a, b) => a + b, 0) / period;
  const mean2 = recent2.reduce((a, b) => a + b, 0) / period;
  
  let numerator = 0;
  let sumSq1 = 0;
  let sumSq2 = 0;
  
  for (let i = 0; i < period; i++) {
    const diff1 = recent1[i] - mean1;
    const diff2 = recent2[i] - mean2;
    numerator += diff1 * diff2;
    sumSq1 += diff1 * diff1;
    sumSq2 += diff2 * diff2;
  }
  
  const denominator = Math.sqrt(sumSq1) * Math.sqrt(sumSq2);
  return denominator > 0 ? numerator / denominator : 0;
}

/**
 * Enrich BTC features with cross-asset correlation and relative strength data
 * @param btcFeatures - BTC feature vectors to enrich
 * @param btcCandles - BTC candle data
 * @param ethCandles - ETH candle data (must be aligned by timestamp with BTC)
 * @param solCandles - SOL candle data (must be aligned by timestamp with BTC)
 * @param bnbCandles - BNB candle data (must be aligned by timestamp with BTC)
 */
export function enrichFeaturesWithCrossAsset(
  btcFeatures: FeatureVector[],
  btcCandles: Candle[],
  ethCandles: Candle[],
  solCandles: Candle[],
  bnbCandles: Candle[]
): FeatureVector[] {
  if (!ethCandles.length || !solCandles.length || !bnbCandles.length) {
    console.log("[Cross-Asset] No altcoin data available, using defaults");
    return btcFeatures;
  }
  
  // Create timestamp-indexed maps for fast lookup
  const ethByTime = new Map(ethCandles.map(c => [c.timestamp, c]));
  const solByTime = new Map(solCandles.map(c => [c.timestamp, c]));
  const bnbByTime = new Map(bnbCandles.map(c => [c.timestamp, c]));
  
  const btcReturns: number[] = [];
  const ethReturns: number[] = [];
  const solReturns: number[] = [];
  const bnbReturns: number[] = [];
  
  // Build aligned return series
  for (let i = 1; i < btcCandles.length; i++) {
    const ts = btcCandles[i].timestamp;
    const prevTs = btcCandles[i - 1].timestamp;
    
    const eth = ethByTime.get(ts);
    const prevEth = ethByTime.get(prevTs);
    const sol = solByTime.get(ts);
    const prevSol = solByTime.get(prevTs);
    const bnb = bnbByTime.get(ts);
    const prevBnb = bnbByTime.get(prevTs);
    
    if (eth && prevEth && sol && prevSol && bnb && prevBnb) {
      btcReturns.push((btcCandles[i].close - btcCandles[i - 1].close) / btcCandles[i - 1].close);
      ethReturns.push((eth.close - prevEth.close) / prevEth.close);
      solReturns.push((sol.close - prevSol.close) / prevSol.close);
      bnbReturns.push((bnb.close - prevBnb.close) / prevBnb.close);
    }
  }
  
  if (btcReturns.length < 20) {
    console.log(`[Cross-Asset] Insufficient aligned data (${btcReturns.length} points), using defaults`);
    return btcFeatures;
  }
  
  // Compute cross-asset features for each BTC feature vector
  const period = 20;
  const momentumPeriod = 10;
  
  return btcFeatures.map((feature, idx) => {
    // Find corresponding index in returns array
    const returnIdx = Math.min(idx, btcReturns.length - 1);
    
    if (returnIdx < period) {
      return feature; // Not enough history yet
    }
    
    // Correlations over trailing 20 periods
    const ethCorr = computeCorrelation(
      btcReturns.slice(0, returnIdx + 1),
      ethReturns.slice(0, returnIdx + 1),
      period
    );
    const solCorr = computeCorrelation(
      btcReturns.slice(0, returnIdx + 1),
      solReturns.slice(0, returnIdx + 1),
      period
    );
    const bnbCorr = computeCorrelation(
      btcReturns.slice(0, returnIdx + 1),
      bnbReturns.slice(0, returnIdx + 1),
      period
    );
    
    // Relative strength: sum of returns over momentum period
    const btcMom = btcReturns.slice(returnIdx - momentumPeriod + 1, returnIdx + 1).reduce((a, b) => a + b, 0);
    const ethMom = ethReturns.slice(returnIdx - momentumPeriod + 1, returnIdx + 1).reduce((a, b) => a + b, 0);
    const solMom = solReturns.slice(returnIdx - momentumPeriod + 1, returnIdx + 1).reduce((a, b) => a + b, 0);
    const bnbMom = bnbReturns.slice(returnIdx - momentumPeriod + 1, returnIdx + 1).reduce((a, b) => a + b, 0);
    
    // Relative strength (1.0 = equal, >1 = altcoin outperforming)
    const safeBtcMom = Math.abs(btcMom) > 0.0001 ? btcMom : 0.0001;
    const ethRS = ethMom / safeBtcMom;
    const solRS = solMom / safeBtcMom;
    const bnbRS = bnbMom / safeBtcMom;
    
    // Momentum divergence (positive = altcoin leading)
    const ethDivergence = ethMom - btcMom;
    const solDivergence = solMom - btcMom;
    const bnbDivergence = bnbMom - btcMom;
    
    // Sector momentum (average altcoin momentum vs BTC - market breadth)
    const avgAltMom = (ethMom + solMom + bnbMom) / 3;
    const sectorMomentum = avgAltMom - btcMom;
    
    return {
      ...feature,
      ethBtcCorrelation: ethCorr,
      solBtcCorrelation: solCorr,
      bnbBtcCorrelation: bnbCorr,
      ethRelativeStrength: Math.max(-5, Math.min(5, ethRS)), // Clamp to [-5, 5]
      solRelativeStrength: Math.max(-5, Math.min(5, solRS)),
      bnbRelativeStrength: Math.max(-5, Math.min(5, bnbRS)),
      ethMomentumDivergence: ethDivergence * 100, // Scale for better neural network learning
      solMomentumDivergence: solDivergence * 100,
      bnbMomentumDivergence: bnbDivergence * 100,
      cryptoSectorMomentum: sectorMomentum * 100,
    };
  });
}

export function getLatestFeatures(candles: Candle[]): FeatureVector | null {
  const features = computeFeatures(candles);
  return features.length > 0 ? features[features.length - 1] : null;
}

export interface TimeframeAggregation {
  timeframe: string;
  multiplier: number;
  candles: Candle[];
}

export function aggregateToTimeframe(candles: Candle[], multiplier: number): Candle[] {
  if (candles.length < multiplier) return [];
  
  const aggregated: Candle[] = [];
  const remainder = candles.length % multiplier;
  const startIndex = remainder;
  const count = Math.floor(candles.length / multiplier);
  
  for (let i = 0; i < count; i++) {
    const sliceStart = startIndex + i * multiplier;
    const sliceEnd = sliceStart + multiplier;
    const slice = candles.slice(sliceStart, sliceEnd);
    if (slice.length === 0) continue;
    
    aggregated.push({
      timestamp: slice[0].timestamp,
      open: slice[0].open,
      high: Math.max(...slice.map(c => c.high)),
      low: Math.min(...slice.map(c => c.low)),
      close: slice[slice.length - 1].close,
      volume: slice.reduce((sum, c) => sum + c.volume, 0),
    });
  }
  
  return aggregated;
}

export interface MultiTimeframePattern {
  timeframe: string;
  patterns: CandlestickPattern[];
  trend: "bullish" | "bearish" | "neutral";
  strength: number;
}

export interface MultiTimeframeCorrelation {
  timeframes: MultiTimeframePattern[];
  overallSignal: "bullish" | "bearish" | "neutral";
  confluence: number;
  alignedTimeframes: number;
  divergence: boolean;
  description: string;
}

export function analyzeMultiTimeframePatterns(candles: Candle[]): MultiTimeframeCorrelation {
  const timeframeMultipliers = [
    { name: "5m", mult: 1 },
    { name: "15m", mult: 3 },
    { name: "1h", mult: 12 },
    { name: "4h", mult: 48 },
  ];
  
  const timeframes: MultiTimeframePattern[] = [];
  let bullishCount = 0;
  let bearishCount = 0;
  let totalStrength = 0;
  
  for (const tf of timeframeMultipliers) {
    const aggregated = tf.mult === 1 ? candles : aggregateToTimeframe(candles, tf.mult);
    
    if (aggregated.length < 10) continue;
    
    const patterns = detectCandlestickPatterns(aggregated);
    const volumeProfile = analyzeVolumeProfile(aggregated);
    
    const bullishPatterns = patterns.filter(p => p.type === "bullish");
    const bearishPatterns = patterns.filter(p => p.type === "bearish");
    
    let trend: "bullish" | "bearish" | "neutral" = "neutral";
    let strength = 0;
    
    if (bullishPatterns.length > bearishPatterns.length) {
      trend = "bullish";
      strength = bullishPatterns.reduce((s, p) => s + p.strength, 0) / bullishPatterns.length;
      bullishCount++;
    } else if (bearishPatterns.length > bullishPatterns.length) {
      trend = "bearish";
      strength = bearishPatterns.reduce((s, p) => s + p.strength, 0) / bearishPatterns.length;
      bearishCount++;
    } else if (volumeProfile.volumeRatio > 1.2) {
      trend = "bullish";
      strength = 0.6;
      bullishCount++;
    } else if (volumeProfile.volumeRatio < 0.8) {
      trend = "bearish";
      strength = 0.6;
      bearishCount++;
    }
    
    const recentCandles = aggregated.slice(-5);
    if (recentCandles.length >= 2) {
      const priceChange = (recentCandles[recentCandles.length - 1].close - recentCandles[0].close) / recentCandles[0].close;
      if (priceChange > 0.005 && trend === "neutral") {
        trend = "bullish";
        strength = 0.5;
        bullishCount++;
      } else if (priceChange < -0.005 && trend === "neutral") {
        trend = "bearish";
        strength = 0.5;
        bearishCount++;
      }
    }
    
    totalStrength += strength;
    
    timeframes.push({
      timeframe: tf.name,
      patterns,
      trend,
      strength,
    });
  }
  
  const alignedTimeframes = Math.max(bullishCount, bearishCount);
  const totalTimeframes = timeframes.length;
  const confluence = totalTimeframes > 0 ? alignedTimeframes / totalTimeframes : 0;
  const divergence = bullishCount > 0 && bearishCount > 0 && Math.abs(bullishCount - bearishCount) <= 1;
  
  let overallSignal: "bullish" | "bearish" | "neutral" = "neutral";
  if (bullishCount > bearishCount && confluence >= 0.5) {
    overallSignal = "bullish";
  } else if (bearishCount > bullishCount && confluence >= 0.5) {
    overallSignal = "bearish";
  }
  
  const avgStrength = totalTimeframes > 0 ? totalStrength / totalTimeframes : 0;
  
  let description = "";
  if (confluence >= 0.75) {
    description = `Strong ${overallSignal} confluence across ${alignedTimeframes}/${totalTimeframes} timeframes`;
  } else if (divergence) {
    description = `Timeframe divergence detected - mixed signals across timeframes`;
  } else if (confluence >= 0.5) {
    description = `Moderate ${overallSignal} bias with ${(confluence * 100).toFixed(0)}% confluence`;
  } else {
    description = `No clear multi-timeframe alignment`;
  }
  
  return {
    timeframes,
    overallSignal,
    confluence,
    alignedTimeframes,
    divergence,
    description,
  };
}

/**
 * SHARED ATR-PERCENTILE REGIME CLASSIFIER
 * Used by signal-engine, strategy-learner, and paper engine for consistency
 * 
 * Classifies market regime based on ATR percentile ranking:
 * - 75th+ percentile = shock (extreme volatility)
 * - 50-75th percentile = trending (high volatility, directional)
 * - 25-50th percentile = ranging (moderate volatility)
 * - Below 25th percentile = quiet (low volatility, compression)
 * - Default = chop (no clear pattern)
 */
export type MarketRegime = "trend_up" | "trend_down" | "shock" | "quiet" | "ranging" | "chop";

export interface RegimeAnalysis {
  regime: MarketRegime;
  atrPercentile: number;
  efficiencyRatio: number;
  isTrending: boolean;
  reasoning: string;
}

export function classifyRegime(candles: Candle[], idx?: number): RegimeAnalysis {
  const targetIdx = idx !== undefined ? idx : candles.length - 1;
  if (targetIdx < 20 || candles.length < 21) {
    return {
      regime: "chop",
      atrPercentile: 50,
      efficiencyRatio: 0,
      isTrending: false,
      reasoning: "Insufficient data for regime classification",
    };
  }
  
  const lookback = 20;
  const atrLookback = 100;
  const slice = candles.slice(Math.max(0, targetIdx - lookback), targetIdx + 1);
  const closes = slice.map(c => c.close);
  
  // Calculate current ATR
  let atrSum = 0;
  for (let i = Math.max(1, targetIdx - 13); i <= targetIdx; i++) {
    const candle = candles[i];
    const prevClose = candles[i - 1].close;
    const tr = Math.max(
      candle.high - candle.low,
      Math.abs(candle.high - prevClose),
      Math.abs(candle.low - prevClose)
    );
    atrSum += tr;
  }
  const currentATR = atrSum / 14;
  const currentPrice = closes[closes.length - 1];
  const atrPercent = (currentATR / currentPrice) * 100;
  
  // Calculate historical ATR values for percentile ranking
  const atrHistory: number[] = [];
  const histStart = Math.max(14, targetIdx - atrLookback);
  for (let i = histStart; i <= targetIdx; i++) {
    let histAtrSum = 0;
    for (let j = Math.max(1, i - 13); j <= i; j++) {
      const candle = candles[j];
      const prevClose = candles[j - 1].close;
      const tr = Math.max(
        candle.high - candle.low,
        Math.abs(candle.high - prevClose),
        Math.abs(candle.low - prevClose)
      );
      histAtrSum += tr;
    }
    const histATR = histAtrSum / 14;
    const histPrice = candles[i].close;
    atrHistory.push((histATR / histPrice) * 100);
  }
  
  // Calculate ATR percentile
  const atrPercentile = atrHistory.length > 0 
    ? (atrHistory.filter(a => a < atrPercent).length / atrHistory.length) * 100
    : 50;
  
  // Calculate efficiency ratio for trend detection
  const netMove = Math.abs(closes[closes.length - 1] - closes[0]);
  const totalMove = closes.slice(1).reduce((sum, c, i) => sum + Math.abs(c - closes[i]), 0);
  const efficiencyRatio = totalMove > 0 ? netMove / totalMove : 0;
  
  // Determine trend direction
  const returns = closes.slice(1).map((c, i) => (c - closes[i]) / closes[i]);
  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const isBullish = avgReturn > 0 && closes[closes.length - 1] > closes[0];
  const isBearish = avgReturn < 0 && closes[closes.length - 1] < closes[0];
  
  // REGIME CLASSIFICATION
  let regime: MarketRegime;
  let reasoning: string;
  
  if (atrPercentile >= 75) {
    regime = "shock";
    reasoning = `ATR at ${atrPercentile.toFixed(0)}th percentile - extreme volatility`;
  } else if (atrPercentile >= 50 && efficiencyRatio >= 0.3) {
    if (isBullish) {
      regime = "trend_up";
      reasoning = `ATR ${atrPercentile.toFixed(0)}th pct, ER ${(efficiencyRatio * 100).toFixed(0)}% - bullish trend`;
    } else if (isBearish) {
      regime = "trend_down";
      reasoning = `ATR ${atrPercentile.toFixed(0)}th pct, ER ${(efficiencyRatio * 100).toFixed(0)}% - bearish trend`;
    } else {
      regime = "chop";
      reasoning = `ATR ${atrPercentile.toFixed(0)}th pct but no clear direction`;
    }
  } else if (atrPercentile >= 25 && efficiencyRatio < 0.3) {
    regime = "ranging";
    reasoning = `ATR ${atrPercentile.toFixed(0)}th pct, low ER - ranging market`;
  } else if (atrPercentile < 25) {
    regime = "quiet";
    reasoning = `ATR at ${atrPercentile.toFixed(0)}th percentile - low volatility compression`;
  } else {
    regime = "chop";
    reasoning = `No clear regime pattern detected`;
  }
  
  return {
    regime,
    atrPercentile,
    efficiencyRatio,
    isTrending: regime === "trend_up" || regime === "trend_down",
    reasoning,
  };
}

/**
 * Get regime-adaptive risk/reward parameters
 * Returns appropriate stop and TP multipliers based on current regime
 */
export function getRegimeRiskParams(regime: MarketRegime): {
  stopMultiplier: number;
  rrRatio: number;
  maxPositionSize: number;
  reasoning: string;
} {
  switch (regime) {
    case "trend_up":
    case "trend_down":
      return {
        stopMultiplier: 1.0,    // Full ATR for stop
        rrRatio: 2.0,           // 2:1 reward-to-risk
        maxPositionSize: 0.15,  // 15% max position
        reasoning: "Trending - wide stops, let winners run",
      };
    case "shock":
      return {
        stopMultiplier: 0.7,    // Tighter stops in volatile conditions
        rrRatio: 1.2,           // Quick exits
        maxPositionSize: 0.05,  // Reduced position size
        reasoning: "Shock - reduced exposure, quick exits",
      };
    case "quiet":
      return {
        stopMultiplier: 0.8,    // Moderate stops
        rrRatio: 2.0,           // Wide target for breakout
        maxPositionSize: 0.10,  // Moderate position
        reasoning: "Quiet - await breakout, wide targets",
      };
    case "ranging":
      return {
        stopMultiplier: 0.75,   // Tighter for mean reversion
        rrRatio: 1.5,           // Moderate targets
        maxPositionSize: 0.10,  // Moderate position
        reasoning: "Ranging - quick profits, mean reversion",
      };
    case "chop":
    default:
      return {
        stopMultiplier: 0.6,    // Very tight stops
        rrRatio: 1.3,           // Conservative targets
        maxPositionSize: 0.03,  // Minimal position
        reasoning: "Chop - minimal exposure recommended",
      };
  }
}
