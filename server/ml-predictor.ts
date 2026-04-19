import type { FeatureVector } from "./feature-engine";
import { findSimilarPatterns, computePatternStats, getPatternConfidence, type PatternStats } from "./pattern-memory";
import { analyzeMarket as openAIAnalyze, generateAISignal as openAISignal } from "./ai-analysis";
import type { Candle, FuturesData, AIAnalysis, AISignal } from "@shared/schema";

export interface MLPrediction {
  probUp: number;
  probDown: number;
  probChop: number;
  expectedMove: number;
  confidence: number;
  direction: "LONG" | "SHORT" | "HOLD";
  model: string;
}

export interface EnsemblePrediction {
  probUp: number;
  probDown: number;
  probChop: number;
  expectedMove: number;
  confidence: number;
  direction: "LONG" | "SHORT" | "HOLD";
  models: {
    rulebased: MLPrediction;
    pattern: MLPrediction;
    ai: MLPrediction | null;
    gpu: MLPrediction | null;
  };
  consensus: number;
  gpuUncertainty?: number;
}

// GPU Model Prediction Storage - received from external GPU trainer
export interface GPUPrediction {
  symbol: string;
  timestamp: number;
  returnH1: number;      // Predicted return at horizon 1 (15m)
  returnH2: number;      // Predicted return at horizon 2 (60m)
  returnH3: number;      // Predicted return at horizon 3 (240m)
  quantile10: number;    // 10th percentile (downside risk)
  quantile50: number;    // Median prediction
  quantile90: number;    // 90th percentile (upside potential)
  directionalProb: number; // P(positive return)
  modelId: string;
  confidence: number;
}

// In-memory storage for GPU predictions (updated by /api/gpu-export/predictions endpoint)
const gpuPredictionCache = new Map<string, GPUPrediction>();
const GPU_PREDICTION_TTL_MS = 5 * 60 * 1000; // 5 minutes TTL

export function updateGPUPrediction(prediction: GPUPrediction): void {
  const key = `${prediction.symbol}_${prediction.modelId}`;
  gpuPredictionCache.set(key, prediction);
  console.log(`[GPU] Updated prediction for ${key}: H1=${(prediction.returnH1*100).toFixed(2)}%, conf=${(prediction.confidence*100).toFixed(0)}%`);
  // V7 Path A signal evaluation hook (no-op for non-V7 symbols)
  import("./paper/v7_path_a").then(m => m.onV7Prediction(prediction)).catch(e => {
    console.error("[V7] onV7Prediction failed:", e);
  });
}

export function getLatestGPUPrediction(symbol: string = "BTCUSDT"): GPUPrediction | null {
  // Find most recent prediction for this symbol
  let latest: GPUPrediction | null = null;
  const now = Date.now();
  const keysToDelete: string[] = [];
  
  gpuPredictionCache.forEach((pred, key) => {
    if (key.startsWith(symbol)) {
      // Check TTL
      if (now - pred.timestamp > GPU_PREDICTION_TTL_MS) {
        keysToDelete.push(key);
        return;
      }
      if (!latest || pred.timestamp > latest.timestamp) {
        latest = pred;
      }
    }
  });
  
  // Clean up expired entries
  keysToDelete.forEach(key => gpuPredictionCache.delete(key));
  
  return latest;
}

// Convert GPU prediction to MLPrediction format for ensemble integration
function gpuToMLPrediction(gpuPred: GPUPrediction, currentPrice: number): MLPrediction {
  // Sanitize quantile values to prevent NaN/undefined issues
  const q10 = isFinite(gpuPred.quantile10) ? gpuPred.quantile10 : 0;
  const q50 = isFinite(gpuPred.quantile50) ? gpuPred.quantile50 : gpuPred.returnH1 || 0;
  const q90 = isFinite(gpuPred.quantile90) ? gpuPred.quantile90 : 0;
  
  // Use quantile spread to determine uncertainty (with guard against divide-by-zero)
  const quantileSpread = q90 - q10;
  const uncertainty = Math.abs(q50) > 0.0001 ? quantileSpread / Math.abs(q50) : 1.0;
  
  // Convert directional probability to action probabilities
  const dirProb = gpuPred.directionalProb;
  
  // If directional prob is close to 0.5, favor HOLD
  const edgeFromNeutral = Math.abs(dirProb - 0.5) * 2;  // 0 to 1 scale
  
  let pLong = 0;
  let pShort = 0;
  let pHold = 0;
  
  if (edgeFromNeutral < 0.15) {
    // Low directional edge - favor HOLD
    pHold = 0.6;
    pLong = 0.2;
    pShort = 0.2;
  } else if (dirProb > 0.5) {
    // Bullish signal
    pLong = Math.min(0.7, 0.3 + dirProb * 0.5);
    pShort = 0.1;
    pHold = 1 - pLong - pShort;
  } else {
    // Bearish signal
    pShort = Math.min(0.7, 0.3 + (1 - dirProb) * 0.5);
    pLong = 0.1;
    pHold = 1 - pLong - pShort;
  }
  
  // Adjust for high uncertainty - increase HOLD probability
  if (uncertainty > 1.5) {
    const uncertaintyPenalty = Math.min(0.3, (uncertainty - 1.5) * 0.2);
    pHold = Math.min(0.8, pHold + uncertaintyPenalty);
    pLong *= (1 - uncertaintyPenalty);
    pShort *= (1 - uncertaintyPenalty);
  }
  
  // Normalize
  const total = pLong + pShort + pHold;
  pLong /= total;
  pShort /= total;
  pHold /= total;
  
  // Expected move from quantile median (quantile50 is a return, not price)
  // Multiply by currentPrice to get expected price move
  const expectedMove = gpuPred.quantile50 * currentPrice;
  
  // Determine direction based on probabilities
  let direction: "LONG" | "SHORT" | "HOLD" = "HOLD";
  let confidence = pHold;
  
  if (pLong > pShort + 0.1 && pLong > pHold && edgeFromNeutral > 0.2) {
    direction = "LONG";
    confidence = pLong * gpuPred.confidence;
  } else if (pShort > pLong + 0.1 && pShort > pHold && edgeFromNeutral > 0.2) {
    direction = "SHORT";
    confidence = pShort * gpuPred.confidence;
  }
  
  return {
    probUp: pLong,
    probDown: pShort,
    probChop: pHold,
    expectedMove,
    confidence,
    direction,
    model: "gpu_neural_net",
  };
}

// ACTION-BASED MODEL: Each model outputs P(LONG), P(SHORT), P(HOLD)
// HOLD is a valid action when signals are neutral or conflicting
function ruleBasedPredict(feature: FeatureVector): MLPrediction {
  let bullScore = 0;
  let bearScore = 0;
  let holdScore = 0;  // Renamed from chopScore to holdScore for action-based thinking
  
  // RSI signals - with proper HOLD zone
  if (feature.rsi14 < 25) bullScore += 2.5;  // Strong oversold
  else if (feature.rsi14 < 35) bullScore += 1.0;  // Mild oversold
  else if (feature.rsi14 > 75) bearScore += 2.5;  // Strong overbought
  else if (feature.rsi14 > 65) bearScore += 1.0;  // Mild overbought
  else holdScore += 2.0;  // Neutral zone = HOLD
  
  // MACD signals - require meaningful momentum
  const macdStrength = Math.abs(feature.macdHist) / (Math.abs(feature.macd) + 0.01);
  if (feature.macdHist > 0 && macdStrength > 0.15) bullScore += 1.5;
  else if (feature.macdHist < 0 && macdStrength > 0.15) bearScore += 1.5;
  else holdScore += 1.0;  // Weak MACD = no edge
  
  // Kalman regime - crucial for HOLD decision
  if (feature.kalmanRegime === "bull") bullScore += 2;
  else if (feature.kalmanRegime === "bear") bearScore += 2;
  else holdScore += 3;  // CHOP regime = strongly favor HOLD
  
  // ADX trend strength - low ADX means HOLD
  if (feature.adx > 30) {
    if (feature.plusDi > feature.minusDi) bullScore += 1.5;
    else bearScore += 1.5;
  } else if (feature.adx > 20) {
    // Moderate trend - weak signal
    if (feature.plusDi > feature.minusDi) bullScore += 0.5;
    else bearScore += 0.5;
  } else {
    holdScore += 2.5;  // Low ADX = no trend = HOLD
  }
  
  // Stochastic - only at extremes
  if (feature.stochK < 15) bullScore += 1.5;
  else if (feature.stochK > 85) bearScore += 1.5;
  else holdScore += 0.5;
  
  // Efficiency ratio - choppy market = HOLD
  if (feature.efficiencyRatio > 0.65) {
    if (feature.returns4 > 0.002) bullScore += 1;
    else if (feature.returns4 < -0.002) bearScore += 1;
  } else if (feature.efficiencyRatio < 0.35) {
    holdScore += 2;  // Very choppy = strong HOLD
  } else {
    holdScore += 0.5;
  }
  
  // EMA distance - for mean reversion at extremes only
  if (feature.emaDistance > 1.5) bearScore += 0.5;
  else if (feature.emaDistance < -1.5) bullScore += 0.5;
  
  const total = bullScore + bearScore + holdScore;
  
  // Convert scores to ACTION probabilities (not direction probabilities)
  const pLong = bullScore / total;
  const pShort = bearScore / total;
  const pHold = holdScore / total;
  
  const expectedMove = (pLong - pShort) * feature.atr14 * 2;
  
  // ACTION-BASED DIRECTION: HOLD is a valid output when it has highest probability
  let direction: "LONG" | "SHORT" | "HOLD" = "HOLD";
  let confidence = 0;
  
  // Require clear edge for action - otherwise HOLD
  const actionThreshold = 0.40;  // Need 40% probability for action
  const edgeThreshold = 0.12;    // Need 12% edge over next best action
  
  if (pLong > actionThreshold && pLong > pShort + edgeThreshold && pLong > pHold) {
    direction = "LONG";
    confidence = pLong;
  } else if (pShort > actionThreshold && pShort > pLong + edgeThreshold && pShort > pHold) {
    direction = "SHORT";
    confidence = pShort;
  } else {
    // Not enough edge - HOLD
    direction = "HOLD";
    confidence = pHold;
  }
  
  return {
    probUp: pLong,
    probDown: pShort,
    probChop: pHold,
    expectedMove,
    confidence,
    direction,
    model: "rule_based",
  };
}

// ACTION-BASED PATTERN MODEL: HOLD when pattern EV < 0 or insufficient history
// PRECISION FIX: Use threshold 0.82 to match Pattern Memory's MIN_SIMILARITY_THRESHOLD
// Lower thresholds (0.6) include weak matches causing false positives
async function patternBasedPredict(feature: FeatureVector): Promise<MLPrediction> {
  try {
    // P0-1 FIX: Pass feature.timestamp to prevent backtest leakage
    const matches = await findSimilarPatterns(feature.embedding, 50, 0.82, undefined, feature.timestamp);
    const stats = computePatternStats(matches);
    const totalMatches = matches.length;
    
    // Insufficient pattern history - strong HOLD signal
    if (totalMatches < 15) {
      return {
        probUp: 0.15,
        probDown: 0.15,
        probChop: 0.70,  // High HOLD probability when we don't have enough data
        expectedMove: 0,
        confidence: 0.70,  // Confident in HOLD, not in direction
        direction: "HOLD",
        model: "pattern_memory",
      };
    }
    
    // Calculate EV for each action from historical pattern outcomes
    let longEV = 0;
    let shortEV = 0;
    let longWins = 0;
    let shortWins = 0;
    let longCount = 0;
    let shortCount = 0;
    
    for (const match of matches) {
      const weight = match.similarity;
      const ret = match.forwardReturn8;
      
      // Calculate EV if we had taken LONG at this pattern
      longEV += weight * ret;  // Positive return = LONG wins
      if (ret > 0.001) longWins += weight;
      longCount += weight;
      
      // Calculate EV if we had taken SHORT at this pattern  
      shortEV += weight * (-ret);  // Negative return = SHORT wins
      if (ret < -0.001) shortWins += weight;
      shortCount += weight;
    }
    
    // Normalize EVs
    const avgLongEV = longCount > 0 ? longEV / longCount : 0;
    const avgShortEV = shortCount > 0 ? shortEV / shortCount : 0;
    
    // Win rates for each direction
    const longWinRate = longCount > 0 ? longWins / longCount : 0;
    const shortWinRate = shortCount > 0 ? shortWins / shortCount : 0;
    
    // STANDARDIZED: 0.10% round-trip trading costs (aligned with all systems)
    const tradingCosts = 0.0010;  // 0.10% round trip (maker fees + slippage + funding)
    
    // ACTION-BASED: Use EV directly for action selection
    const netLongEV = avgLongEV - tradingCosts;
    const netShortEV = avgShortEV - tradingCosts;
    
    // EV thresholds for action
    const evThreshold = 0.0005;  // 0.05% minimum EV after costs
    const winRateThreshold = 0.45;  // 45% minimum historical win rate
    
    // Directly select action based on max EV (not probability normalization)
    let direction: "LONG" | "SHORT" | "HOLD" = "HOLD";
    let confidence = 0;
    
    // Find max EV action
    const maxEV = Math.max(netLongEV, netShortEV, 0);  // 0 = HOLD EV
    
    if (maxEV <= 0 || maxEV < evThreshold) {
      // No action has positive EV - HOLD is best
      direction = "HOLD";
      confidence = 0.7;  // High confidence in HOLD when no edge exists
    } else if (netLongEV >= netShortEV && netLongEV > evThreshold && longWinRate > winRateThreshold) {
      // LONG has highest positive EV and meets win rate threshold
      direction = "LONG";
      confidence = Math.min(0.85, 0.5 + netLongEV * 10 + longWinRate * 0.3);
    } else if (netShortEV > netLongEV && netShortEV > evThreshold && shortWinRate > winRateThreshold) {
      // SHORT has highest positive EV and meets win rate threshold
      direction = "SHORT";
      confidence = Math.min(0.85, 0.5 + netShortEV * 10 + shortWinRate * 0.3);
    } else {
      // EV positive but win rate too low - HOLD
      direction = "HOLD";
      confidence = 0.6;
    }
    
    // Convert EV to probabilities for ensemble aggregation
    // Use sigmoid-like transformation of EV
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x * 100));
    const pLong = netLongEV > 0 && longWinRate > winRateThreshold ? sigmoid(netLongEV) : 0.1;
    const pShort = netShortEV > 0 && shortWinRate > winRateThreshold ? sigmoid(netShortEV) : 0.1;
    const pHold = Math.max(0.2, 1 - pLong - pShort);
    
    // Normalize probabilities
    const total = pLong + pShort + pHold;
    const normPLong = pLong / total;
    const normPShort = pShort / total;
    const normPHold = pHold / total;
    
    const expectedMove = stats.avgReturn8 * feature.kalmanFast;
    
    return {
      probUp: normPLong,
      probDown: normPShort,
      probChop: normPHold,
      expectedMove,
      confidence,
      direction,
      model: "pattern_memory",
    };
  } catch (error) {
    console.error("Pattern prediction error:", error);
    return {
      probUp: 0.15,
      probDown: 0.15,
      probChop: 0.70,
      expectedMove: 0,
      confidence: 0.70,
      direction: "HOLD",
      model: "pattern_memory",
    };
  }
}

async function aiBasedPredict(
  candles: Candle[],
  feature: FeatureVector,
  futuresData: FuturesData
): Promise<MLPrediction | null> {
  try {
    type SignalType = "bullish" | "bearish" | "neutral";
    const rsiSignal: SignalType = feature.rsi14 < 30 ? "bullish" : feature.rsi14 > 70 ? "bearish" : "neutral";
    const macdSignal: SignalType = feature.macdHist > 0 ? "bullish" : "bearish";
    const obvSignal: SignalType = feature.obvSlope > 0 ? "bullish" : "bearish";
    const adxSignal: SignalType = feature.adx > 25 ? "bullish" : "neutral";
    const stochSignal: SignalType = feature.stochK < 20 ? "bullish" : feature.stochK > 80 ? "bearish" : "neutral";
    const emaSignal: SignalType = feature.ema20Slope > 0 ? "bullish" : "bearish";
    const volSignal: SignalType = feature.volumeRatio > 1.2 ? "bullish" : "neutral";
    
    const indicators = {
      rsi: { name: "RSI", value: feature.rsi14, signal: rsiSignal, strength: Math.abs(feature.rsi14 - 50) / 50, description: "" },
      macd: { name: "MACD", value: feature.macd, signal: macdSignal, strength: 0.5, description: "", histogram: feature.macdHist, macdLine: feature.macd, signalLine: feature.macdSignal },
      bollingerBands: { name: "BB", value: 0, signal: "neutral" as SignalType, strength: 0.5, description: "", upper: 0, middle: 0, lower: 0, percentB: 0.5 },
      obv: { name: "OBV", value: feature.obv, signal: obvSignal, strength: 0.5, description: "" },
      vwap: { name: "VWAP", value: 0, signal: "neutral" as SignalType, strength: 0.5, description: "" },
      atr: { name: "ATR", value: feature.atr14, signal: "neutral" as SignalType, strength: 0.5, description: "" },
      adx: { name: "ADX", value: feature.adx, signal: adxSignal, strength: feature.adx / 100, description: "", plusDI: feature.plusDi, minusDI: feature.minusDi },
      stochastic: { name: "Stoch", value: feature.stochK, signal: stochSignal, strength: 0.5, description: "", k: feature.stochK, d: feature.stochD },
      ema: { name: "EMA", value: feature.ema20, signal: emaSignal, strength: 0.5, description: "", ema9: feature.ema20, ema21: feature.ema20, ema50: feature.ema50, ema200: feature.ema50 },
      supportResistance: { name: "S/R", value: 0, signal: "neutral" as SignalType, strength: 0.5, description: "", supports: [0], resistances: [0] },
      volumeProfile: { name: "Vol", value: feature.volumeRatio, signal: volSignal, strength: 0.5, description: "", highVolumeZones: [0], pocPrice: 0 },
    };
    
    const whaleActivity = {
      largeBuys: 0,
      largeSells: 0,
      netFlow: 0,
      whaleActivity: "neutral" as const,
    };
    
    const mtfScore = {
      direction: "neutral" as const,
      score: 0,
      alignment: 0.5,
      details: [
        { timeframe: "5m", trend: "neutral" as const, weight: 1 },
        { timeframe: "15m", trend: "neutral" as const, weight: 2 },
        { timeframe: "1h", trend: "neutral" as const, weight: 3 },
        { timeframe: "4h", trend: "neutral" as const, weight: 4 },
      ],
    };
    
    const aiSignal = await openAISignal(candles, indicators, futuresData, whaleActivity, mtfScore);
    
    if (!aiSignal) return null;
    
    return {
      probUp: aiSignal.direction === "LONG" ? aiSignal.confidence : aiSignal.direction === "SHORT" ? 0.2 : 0.4,
      probDown: aiSignal.direction === "SHORT" ? aiSignal.confidence : aiSignal.direction === "LONG" ? 0.2 : 0.4,
      probChop: aiSignal.direction === "HOLD" ? aiSignal.confidence : 0.2,
      expectedMove: (aiSignal.takeProfit1 || 0) - (aiSignal.entryPrice || 0),
      confidence: aiSignal.confidence,
      direction: aiSignal.direction,
      model: "openai",
    };
  } catch (error) {
    console.error("AI prediction error:", error);
    return null;
  }
}

function computeConfidence(
  probUp: number,
  probDown: number,
  probChop: number,
  expectedMove: number,
  currentPrice: number,
  costs: number,
  patternMaturity: number,
  modelConfidences: number[],
  modelDirections: ("LONG" | "SHORT" | "HOLD")[],
  adx: number
): { confidence: number; components: ConfidenceComponents } {
  const baseConfidence = Math.max(probUp, probDown);
  
  const regimeClarity = 1 - probChop;
  
  const edge = Math.abs(expectedMove) / currentPrice;
  let edgePenalty = 1.0;
  if (edge <= costs) {
    edgePenalty = Math.max(0.5, 0.5 + (edge / costs) * 0.5);
  } else {
    edgePenalty = Math.min(1.2, 1 + (edge - costs) / costs * 0.1);
  }
  
  const patternFactor = 0.6 + patternMaturity * 0.4;
  
  // Confidence agreement (variance of confidence scores)
  let confidenceAgreement = 1.0;
  if (modelConfidences.length > 1) {
    const mean = modelConfidences.reduce((a, b) => a + b, 0) / modelConfidences.length;
    const variance = modelConfidences.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / modelConfidences.length;
    const stdDev = Math.sqrt(variance);
    confidenceAgreement = Math.max(0.7, 1 - stdDev * 0.5);
  }
  
  // DIRECTION AGREEMENT - Strongly penalize when models disagree on LONG vs SHORT
  // This is the critical precision fix: opposing directions = high uncertainty
  let directionAgreement = 1.0;
  if (modelDirections.length > 1) {
    const longCount = modelDirections.filter(d => d === "LONG").length;
    const shortCount = modelDirections.filter(d => d === "SHORT").length;
    
    // If any models say LONG and others say SHORT, apply severe penalty
    if (longCount > 0 && shortCount > 0) {
      // Opposing signals: the more balanced the split, the worse the penalty
      const opposition = Math.min(longCount, shortCount);
      const totalDirectional = longCount + shortCount;
      const oppositionRatio = opposition / totalDirectional;  // 0.5 = perfect split
      
      // Severe penalty for directional disagreement (0.3 to 0.7 range)
      directionAgreement = 0.3 + (1 - oppositionRatio) * 0.4;
    }
  }
  
  // Combined agreement: min of confidence and direction agreement
  const agreement = Math.min(confidenceAgreement, directionAgreement);
  
  const trendStrength = Math.min(1.0, adx / 40);
  const trendFactor = 0.8 + trendStrength * 0.2;
  
  const weightedConfidence = (
    baseConfidence * 0.35 +
    regimeClarity * 0.20 +
    edgePenalty * 0.10 +
    patternFactor * 0.10 +
    agreement * 0.15 +  // Increased weight for direction agreement
    trendFactor * 0.10
  );
  
  const confidence = Math.max(0.20, Math.min(0.85, weightedConfidence));
  
  return {
    confidence,
    components: {
      base: baseConfidence,
      regimeClarity,
      edgePenalty,
      patternFactor,
      agreement,
      trendFactor,
    }
  };
}

export interface ConfidenceComponents {
  base: number;
  regimeClarity: number;
  edgePenalty: number;
  patternFactor: number;
  agreement: number;
  trendFactor: number;
}

// ACTION-BASED ENSEMBLE: Aggregate EV per action, select max EV, HOLD if max EV <= 0
export async function getEnsemblePrediction(
  candles: Candle[],
  feature: FeatureVector,
  futuresData: FuturesData,
  includeAI: boolean = true,
  symbol: string = "BTCUSDT"  // Added symbol parameter for GPU predictions
): Promise<EnsemblePrediction> {
  const rulePrediction = ruleBasedPredict(feature);
  const patternPrediction = await patternBasedPredict(feature);
  const aiPrediction = includeAI ? await aiBasedPredict(candles, feature, futuresData) : null;
  
  // Get GPU prediction for the specified symbol
  const gpuRawPrediction = getLatestGPUPrediction(symbol);
  // Use close price for expected move calculation (quantiles are returns, not prices)
  const latestPrice = candles.length > 0 ? candles[candles.length - 1].close : feature.kalmanFast || 50000;
  const gpuPrediction = gpuRawPrediction ? gpuToMLPrediction(gpuRawPrediction, latestPrice) : null;
  
  // Model weights - GPU gets significant weight when available (trained on larger dataset)
  // If GPU available: rule=0.25, pattern=0.25, ai=0.20, gpu=0.30
  // If GPU not available: rule=0.35, pattern=0.35, ai=0.30 (original)
  const weights = gpuPrediction ? {
    rulebased: 0.25,
    pattern: 0.25,
    ai: 0.20,
    gpu: 0.30,
  } : {
    rulebased: 0.35,
    pattern: 0.35,
    ai: 0.30,
    gpu: 0,
  };
  
  // STANDARDIZED: 0.10% round-trip trading costs (maker fees + slippage + funding)
  const tradingCosts = 0.0010;  // Aligned with pattern-memory and signal-engine
  const currentPrice = feature.kalmanFast || 1;
  
  // STEP 1: Aggregate weighted action probabilities from all models
  // probUp = P(LONG action is profitable)
  // probDown = P(SHORT action is profitable)  
  // probChop = P(HOLD action is best)
  let totalWeight = weights.rulebased + weights.pattern;
  let pLong = rulePrediction.probUp * weights.rulebased + patternPrediction.probUp * weights.pattern;
  let pShort = rulePrediction.probDown * weights.rulebased + patternPrediction.probDown * weights.pattern;
  let pHold = rulePrediction.probChop * weights.rulebased + patternPrediction.probChop * weights.pattern;
  let expectedMove = rulePrediction.expectedMove * weights.rulebased + patternPrediction.expectedMove * weights.pattern;
  
  if (aiPrediction) {
    totalWeight += weights.ai;
    pLong += aiPrediction.probUp * weights.ai;
    pShort += aiPrediction.probDown * weights.ai;
    pHold += aiPrediction.probChop * weights.ai;
    expectedMove += aiPrediction.expectedMove * weights.ai;
  }
  
  // Add GPU prediction if available
  if (gpuPrediction && weights.gpu > 0) {
    totalWeight += weights.gpu;
    pLong += gpuPrediction.probUp * weights.gpu;
    pShort += gpuPrediction.probDown * weights.gpu;
    pHold += gpuPrediction.probChop * weights.gpu;
    expectedMove += gpuPrediction.expectedMove * weights.gpu;
  }
  
  // Normalize probabilities
  pLong /= totalWeight;
  pShort /= totalWeight;
  pHold /= totalWeight;
  expectedMove /= totalWeight;
  
  // STEP 2: Calculate Expected Value (EV) for each action
  // EV(LONG) = P(price goes up) * avg_up_return - P(price goes down) * avg_down_return - costs
  // Simplified: EV(LONG) = pLong * expectedMove - costs (if expectedMove is positive)
  
  const atr = feature.atr14;
  const avgWinSize = atr * 1.2 / currentPrice;  // Expected winner size as %
  const avgLossSize = atr * 0.8 / currentPrice; // Expected loser size as %
  
  const evLong = pLong * avgWinSize - (1 - pLong) * avgLossSize - tradingCosts;
  const evShort = pShort * avgWinSize - (1 - pShort) * avgLossSize - tradingCosts;
  const evHold = 0;  // HOLD has zero EV but also zero risk
  
  // STEP 3: Count model votes for consensus tracking
  const votes = [rulePrediction.direction, patternPrediction.direction];
  if (aiPrediction) votes.push(aiPrediction.direction);
  if (gpuPrediction) votes.push(gpuPrediction.direction);
  
  const longVotes = votes.filter(v => v === "LONG").length;
  const shortVotes = votes.filter(v => v === "SHORT").length;
  const holdVotes = votes.filter(v => v === "HOLD").length;
  
  // STEP 4: Select action with HIGHEST EV (action-based, not direction-based)
  // This is the institutional approach: only trade when EV is positive
  let direction: "LONG" | "SHORT" | "HOLD";
  let consensus = 0;
  
  // Find max EV action
  const maxEV = Math.max(evLong, evShort, evHold);
  
  // Minimum EV threshold to take action (must cover costs with margin)
  const minActionEV = tradingCosts * 0.5;  // Need at least 50% of costs as expected profit
  
  if (maxEV <= 0 || maxEV < minActionEV) {
    // No action has positive EV - HOLD
    direction = "HOLD";
    consensus = holdVotes / votes.length;
  } else if (evLong >= evShort && evLong > minActionEV) {
    // LONG has highest positive EV
    direction = "LONG";
    consensus = longVotes / votes.length;
  } else if (evShort > evLong && evShort > minActionEV) {
    // SHORT has highest positive EV
    direction = "SHORT";
    consensus = shortVotes / votes.length;
  } else {
    // Edge case - default to HOLD
    direction = "HOLD";
    consensus = holdVotes / votes.length;
  }
  
  // STEP 5: Additional regime-based HOLD enforcement
  // Even with positive EV, avoid trading in very choppy conditions
  const isChopRegime = feature.kalmanRegime === "chop";
  const isLowADX = feature.adx < 20;
  
  if (direction !== "HOLD" && isChopRegime && isLowADX) {
    // Chop regime with weak trend - reduce confidence or switch to HOLD
    if (maxEV < tradingCosts * 1.5) {
      direction = "HOLD";
      consensus = 0.5;
    }
  }
  
  // STEP 6: GPT veto - if AI strongly says HOLD and we're about to trade
  if (aiPrediction && aiPrediction.direction === "HOLD" && direction !== "HOLD") {
    // AI is saying HOLD but ensemble wants to trade
    // Only override if AI confidence is high and our EV is marginal
    if (aiPrediction.confidence > 0.6 && maxEV < tradingCosts * 2) {
      direction = "HOLD";
      consensus = 0.5;
    }
  }
  
  // Compute confidence for the chosen action
  const patternMaturity = patternPrediction.confidence > 0.5 ? Math.min(1.0, patternPrediction.confidence) : 0.3;
  const modelConfidences = [rulePrediction.confidence, patternPrediction.confidence];
  if (aiPrediction) modelConfidences.push(aiPrediction.confidence);
  if (gpuPrediction) modelConfidences.push(gpuPrediction.confidence);
  
  // Pass model directions for direction-agreement penalty
  const { confidence } = computeConfidence(
    pLong, pShort, pHold, expectedMove, currentPrice, tradingCosts, patternMaturity, modelConfidences, votes, feature.adx
  );
  
  // Calculate GPU uncertainty from quantile spread if available
  let gpuUncertainty: number | undefined;
  if (gpuRawPrediction) {
    const quantileSpread = gpuRawPrediction.quantile90 - gpuRawPrediction.quantile10;
    gpuUncertainty = quantileSpread / (Math.abs(gpuRawPrediction.quantile50) + 0.0001);
  }
  
  return {
    probUp: pLong,
    probDown: pShort,
    probChop: pHold,
    expectedMove,
    confidence,
    direction,
    models: {
      rulebased: rulePrediction,
      pattern: patternPrediction,
      ai: aiPrediction,
      gpu: gpuPrediction,
    },
    consensus,
    gpuUncertainty,
  };
}
