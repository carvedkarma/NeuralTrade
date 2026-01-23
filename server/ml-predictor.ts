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
  };
  consensus: number;
}

function ruleBasedPredict(feature: FeatureVector): MLPrediction {
  let bullScore = 0;
  let bearScore = 0;
  let chopScore = 0;
  
  if (feature.rsi14 < 30) bullScore += 2;
  else if (feature.rsi14 > 70) bearScore += 2;
  else chopScore += 1;
  
  if (feature.macdHist > 0 && feature.macdHist > feature.macd * 0.1) bullScore += 1.5;
  else if (feature.macdHist < 0 && feature.macdHist < feature.macd * 0.1) bearScore += 1.5;
  
  if (feature.kalmanRegime === "bull") bullScore += 2;
  else if (feature.kalmanRegime === "bear") bearScore += 2;
  else chopScore += 2;
  
  if (feature.adx > 25) {
    if (feature.plusDi > feature.minusDi) bullScore += 1.5;
    else bearScore += 1.5;
  } else {
    chopScore += 1.5;
  }
  
  if (feature.stochK < 20) bullScore += 1;
  else if (feature.stochK > 80) bearScore += 1;
  
  if (feature.efficiencyRatio > 0.6) {
    if (feature.returns4 > 0) bullScore += 1;
    else bearScore += 1;
  } else {
    chopScore += 1;
  }
  
  if (feature.emaDistance > 1) bearScore += 0.5;
  else if (feature.emaDistance < -1) bullScore += 0.5;
  
  const total = bullScore + bearScore + chopScore;
  const probUp = bullScore / total;
  const probDown = bearScore / total;
  const probChop = chopScore / total;
  
  const expectedMove = (probUp - probDown) * feature.atr14 * 2;
  
  let direction: "LONG" | "SHORT" | "HOLD" = "HOLD";
  let confidence = 0;
  
  if (probUp > 0.5 && probUp > probDown + 0.15) {
    direction = "LONG";
    confidence = probUp;
  } else if (probDown > 0.5 && probDown > probUp + 0.15) {
    direction = "SHORT";
    confidence = probDown;
  } else {
    confidence = probChop;
  }
  
  return {
    probUp,
    probDown,
    probChop,
    expectedMove,
    confidence,
    direction,
    model: "rule_based",
  };
}

async function patternBasedPredict(feature: FeatureVector): Promise<MLPrediction> {
  try {
    const matches = await findSimilarPatterns(feature.embedding, 50, 0.6);
    const stats = computePatternStats(matches);
    const { direction, confidence, reasoning } = getPatternConfidence(stats);
    
    const probUp = direction === "LONG" ? stats.winRate : (1 - stats.winRate) * 0.5;
    const probDown = direction === "SHORT" ? stats.winRate : (1 - stats.winRate) * 0.5;
    const probChop = 1 - probUp - probDown;
    
    return {
      probUp,
      probDown,
      probChop,
      expectedMove: stats.avgReturn8 * feature.kalmanFast,
      confidence,
      direction,
      model: "pattern_memory",
    };
  } catch (error) {
    console.error("Pattern prediction error:", error);
    return {
      probUp: 0.33,
      probDown: 0.33,
      probChop: 0.34,
      expectedMove: 0,
      confidence: 0,
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
    const indicators = {
      rsi: { name: "RSI", value: feature.rsi14, signal: feature.rsi14 < 30 ? "bullish" : feature.rsi14 > 70 ? "bearish" : "neutral" as const, strength: Math.abs(feature.rsi14 - 50) / 50, description: "" },
      macd: { name: "MACD", value: feature.macd, signal: feature.macdHist > 0 ? "bullish" : "bearish" as const, strength: 0.5, description: "" },
      bollingerBands: { name: "BB", value: 0, signal: "neutral" as const, strength: 0.5, description: "" },
      obv: { name: "OBV", value: feature.obv, signal: feature.obvSlope > 0 ? "bullish" : "bearish" as const, strength: 0.5, description: "" },
      vwap: { name: "VWAP", value: 0, signal: "neutral" as const, strength: 0.5, description: "" },
      atr: { name: "ATR", value: feature.atr14, signal: "neutral" as const, strength: 0.5, description: "" },
      adx: { name: "ADX", value: feature.adx, signal: feature.adx > 25 ? "bullish" : "neutral" as const, strength: feature.adx / 100, description: "" },
      stochastic: { name: "Stoch", value: feature.stochK, signal: feature.stochK < 20 ? "bullish" : feature.stochK > 80 ? "bearish" : "neutral" as const, strength: 0.5, description: "" },
    };
    
    const aiSignal = await openAISignal(candles, indicators, futuresData);
    
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

export async function getEnsemblePrediction(
  candles: Candle[],
  feature: FeatureVector,
  futuresData: FuturesData,
  includeAI: boolean = true
): Promise<EnsemblePrediction> {
  const rulePrediction = ruleBasedPredict(feature);
  const patternPrediction = await patternBasedPredict(feature);
  const aiPrediction = includeAI ? await aiBasedPredict(candles, feature, futuresData) : null;
  
  const weights = {
    rulebased: 0.35,
    pattern: 0.35,
    ai: 0.30,
  };
  
  let totalWeight = weights.rulebased + weights.pattern;
  let probUp = rulePrediction.probUp * weights.rulebased + patternPrediction.probUp * weights.pattern;
  let probDown = rulePrediction.probDown * weights.rulebased + patternPrediction.probDown * weights.pattern;
  let probChop = rulePrediction.probChop * weights.rulebased + patternPrediction.probChop * weights.pattern;
  let expectedMove = rulePrediction.expectedMove * weights.rulebased + patternPrediction.expectedMove * weights.pattern;
  
  if (aiPrediction) {
    totalWeight += weights.ai;
    probUp += aiPrediction.probUp * weights.ai;
    probDown += aiPrediction.probDown * weights.ai;
    probChop += aiPrediction.probChop * weights.ai;
    expectedMove += aiPrediction.expectedMove * weights.ai;
  }
  
  probUp /= totalWeight;
  probDown /= totalWeight;
  probChop /= totalWeight;
  expectedMove /= totalWeight;
  
  const votes = [rulePrediction.direction, patternPrediction.direction];
  if (aiPrediction) votes.push(aiPrediction.direction);
  
  const longVotes = votes.filter(v => v === "LONG").length;
  const shortVotes = votes.filter(v => v === "SHORT").length;
  const holdVotes = votes.filter(v => v === "HOLD").length;
  
  let direction: "LONG" | "SHORT" | "HOLD" = "HOLD";
  let consensus = 0;
  
  if (longVotes > shortVotes && longVotes > holdVotes) {
    direction = "LONG";
    consensus = longVotes / votes.length;
  } else if (shortVotes > longVotes && shortVotes > holdVotes) {
    direction = "SHORT";
    consensus = shortVotes / votes.length;
  } else {
    consensus = holdVotes / votes.length;
  }
  
  const confidence = (rulePrediction.confidence + patternPrediction.confidence + (aiPrediction?.confidence || 0)) / 
    (aiPrediction ? 3 : 2);
  
  return {
    probUp,
    probDown,
    probChop,
    expectedMove,
    confidence,
    direction,
    models: {
      rulebased: rulePrediction,
      pattern: patternPrediction,
      ai: aiPrediction,
    },
    consensus,
  };
}
