import type { Candle, FuturesData, Signal } from "@shared/schema";
import type { FeatureVector } from "./feature-engine";
import { getEnsemblePrediction, type EnsemblePrediction } from "./ml-predictor";
import { findSimilarPatterns, computePatternStats, type PatternMatch } from "./pattern-memory";
import { getSentimentData, interpretFearGreed } from "./sentiment-api";
import { strategyLearner, type CombinedIntelligence } from "./strategy-learner";

function checkNewsFilter(newsScore: number, fearGreedValue: number): { shouldVeto: boolean; reason: string | null } {
  if (fearGreedValue >= 85 || fearGreedValue <= 10) {
    return { 
      shouldVeto: true, 
      reason: `Extreme sentiment (F&G: ${fearGreedValue}) - macro blackout period` 
    };
  }
  if (Math.abs(newsScore) >= 0.8) {
    return { 
      shouldVeto: true, 
      reason: `Extreme news bias (${(newsScore * 100).toFixed(0)}%) - wait for stabilization` 
    };
  }
  return { shouldVeto: false, reason: null };
}

export interface ExpansionGate {
  impulseCandle: boolean;
  atrExpansion: boolean;
  rangeBreak: boolean;
  confirmed: boolean;
  details: string;
}

export type EdgeBucket = "none" | "weak" | "moderate" | "strong";

export interface ShotPlan {
  signal: "LONG" | "SHORT" | "HOLD";
  confidence: number;
  regime: "trend_up" | "trend_down" | "chop" | "shock";
  strategy: string;
  entryZone: { low: number; high: number } | null;
  stopLoss: number | null;
  takeProfit1: number | null;
  takeProfit2: number | null;
  trailingStop: number | null;
  riskReward: number;
  expectedHoldTime: string;
  estimatedCosts: number;
  edge: number;
  edgeBucket: EdgeBucket;
  edgeMultiple: number;
  probUp: number;
  probDown: number;
  probChop: number;
  expectedMove: number;
  reasons: string[];
  vetoReasons: string[];
  patternMatches: PatternMatch[];
  mlPredictions: EnsemblePrediction;
  expansionGate: ExpansionGate;
  combinedIntelligence?: CombinedIntelligence;
}

const FEES = 0.0004;
const SLIPPAGE = 0.0002;
const FUNDING_RISK = 0.0001;
const EDGE_MULTIPLE_MIN = 1.5;

function estimateCosts(holdCandles: number): number {
  const fundingPeriods = Math.ceil(holdCandles / 32);
  return FEES * 2 + SLIPPAGE * 2 + FUNDING_RISK * fundingPeriods;
}

function classifyEdgeBucket(edge: number, costs: number): EdgeBucket {
  if (edge <= 0 || costs <= 0) return "none";
  const multiple = edge / costs;
  if (multiple < 1.0) return "none";
  if (multiple < 1.5) return "weak";
  if (multiple < 2.5) return "moderate";
  return "strong";
}

function checkExpansionGate(
  candles: Candle[],
  atr14: number,
  direction: "LONG" | "SHORT" | "HOLD"
): ExpansionGate {
  if (candles.length < 20 || direction === "HOLD") {
    return {
      impulseCandle: false,
      atrExpansion: false,
      rangeBreak: false,
      confirmed: false,
      details: "Insufficient data or HOLD signal"
    };
  }

  const lastCandle = candles[candles.length - 1];
  
  const impulseThreshold = 0.6 * atr14;
  const candleBody = Math.abs(lastCandle.close - lastCandle.open);
  const impulseCandle = candleBody >= impulseThreshold;
  
  const currentRange = lastCandle.high - lastCandle.low;
  const atrExpansion = currentRange >= atr14 * 1.15;
  
  const rangeLookback = candles.slice(-12, -1);
  let rangeBreak = false;
  
  if (direction === "SHORT") {
    const rangeLow = Math.min(...rangeLookback.map(c => c.low));
    rangeBreak = lastCandle.close < rangeLow;
  } else if (direction === "LONG") {
    const rangeHigh = Math.max(...rangeLookback.map(c => c.high));
    rangeBreak = lastCandle.close > rangeHigh;
  }
  
  const volumeLookback = candles.slice(-20, -1);
  const volumes = volumeLookback.map(c => c.volume).sort((a, b) => a - b);
  const medianVolume = volumes[Math.floor(volumes.length / 2)];
  const volumeConfirms = lastCandle.volume > medianVolume;
  
  const rangeBreakWithVolume = rangeBreak && volumeConfirms;
  
  const confirmed = impulseCandle || atrExpansion || rangeBreakWithVolume;
  
  const details: string[] = [];
  if (impulseCandle) details.push(`Impulse: body ${candleBody.toFixed(0)} >= 0.6*ATR(${impulseThreshold.toFixed(0)})`);
  if (atrExpansion) details.push(`ATR exp: range ${currentRange.toFixed(0)} >= 1.15*ATR(${(atr14 * 1.15).toFixed(0)})`);
  if (rangeBreakWithVolume) details.push(`Range break + vol > median`);
  if (!confirmed) details.push("Waiting for expansion confirmation");

  return {
    impulseCandle,
    atrExpansion,
    rangeBreak: rangeBreakWithVolume,
    confirmed,
    details: details.join("; ")
  };
}

function calculateRiskReward(entry: number, stop: number, tp: number): number {
  const risk = Math.abs(entry - stop);
  const reward = Math.abs(tp - entry);
  return risk === 0 ? 0 : reward / risk;
}

function getRegime(feature: FeatureVector): "trend_up" | "trend_down" | "chop" | "shock" {
  if (feature.volatilityRegime === "high" && feature.adx > 40) {
    return "shock";
  }
  if (feature.kalmanRegime === "bull" && feature.adx > 25) {
    return "trend_up";
  }
  if (feature.kalmanRegime === "bear" && feature.adx > 25) {
    return "trend_down";
  }
  return "chop";
}

function selectStrategy(regime: string, feature: FeatureVector): string {
  if (regime === "shock") {
    return "Shock Recovery";
  }
  if (regime === "chop") {
    return "Mean Reversion";
  }
  if (feature.efficiencyRatio > 0.6) {
    return "Kalman Trend Follow";
  }
  if (Math.abs(feature.kalmanSpread) < feature.atr14 * 0.3) {
    return "Kalman Retest";
  }
  return "Kalman Trend";
}

function getEstimatedHoldTime(regime: string, feature: FeatureVector): string {
  if (regime === "shock") return "1-2 candles (15-30min)";
  if (regime === "chop") return "2-4 candles (30min-1h)";
  if (feature.volatilityRegime === "high") return "2-4 candles (30min-1h)";
  if (feature.adx > 40) return "4-8 candles (1-2h)";
  return "4-6 candles (1-1.5h)";
}

export async function generateShotPlan(
  candles: Candle[],
  feature: FeatureVector,
  futuresData: FuturesData,
  includeAI: boolean = true
): Promise<ShotPlan> {
  const currentPrice = candles[candles.length - 1].close;
  const regime = getRegime(feature);
  const strategy = selectStrategy(regime, feature);
  const holdTimeStr = getEstimatedHoldTime(regime, feature);
  
  const [ensemble, sentiment] = await Promise.all([
    getEnsemblePrediction(candles, feature, futuresData, includeAI),
    getSentimentData()
  ]);
  
  const patternMatches = await findSimilarPatterns(feature.embedding, 20, 0.65);
  const patternStats = computePatternStats(patternMatches);
  
  const reasons: string[] = [];
  const vetoReasons: string[] = [];
  
  const isChopRegime = regime === "chop";
  
  if (isChopRegime) {
    vetoReasons.push("CHOP GATE: Kalman regime is chop - no trade allowed");
  }
  
  const holdCandles = regime === "shock" ? 2 : regime === "chop" ? 3 : 6;
  const costs = estimateCosts(holdCandles);
  
  const fearGreedValue = sentiment.fearGreed?.value ?? 50;
  const newsFilter = checkNewsFilter(sentiment.newsScore, fearGreedValue);
  if (newsFilter.shouldVeto && newsFilter.reason) {
    vetoReasons.push(newsFilter.reason);
  }
  
  if (ensemble.probChop > 0.55 && !isChopRegime) {
    vetoReasons.push(`High chop probability: ${(ensemble.probChop * 100).toFixed(1)}%`);
  }
  if (Math.abs(ensemble.expectedMove) < costs * 3) {
    vetoReasons.push(`Edge too small: Expected ${(ensemble.expectedMove / currentPrice * 100).toFixed(2)}% vs costs ${(costs * 100).toFixed(2)}%`);
  }
  if (feature.volatilityRegime === "high" && regime === "chop") {
    vetoReasons.push("High volatility in choppy regime - dangerous");
  }
  if (ensemble.consensus < 0.5) {
    vetoReasons.push(`Low model consensus: ${(ensemble.consensus * 100).toFixed(0)}%`);
  }
  if (ensemble.confidence < 0.65) {
    vetoReasons.push(`Low confidence: ${(ensemble.confidence * 100).toFixed(0)}% (need 65%+)`);
  }
  if (patternMatches.length < 10) {
    vetoReasons.push(`Insufficient pattern matches: ${patternMatches.length} (need 10+)`);
  }
  
  if (ensemble.direction === "LONG") {
    if (feature.rsi14 < 40) reasons.push("RSI oversold, bounce potential");
    if (feature.kalmanRegime === "bull") reasons.push("Kalman filter bullish regime");
    if (feature.macdHist > 0) reasons.push("MACD histogram positive");
    if (feature.stochK < 30) reasons.push("Stochastic oversold");
    if (patternStats.winRate > 0.55) reasons.push(`Pattern history: ${(patternStats.winRate * 100).toFixed(0)}% win rate`);
  } else if (ensemble.direction === "SHORT") {
    if (feature.rsi14 > 60) reasons.push("RSI overbought, pullback potential");
    if (feature.kalmanRegime === "bear") reasons.push("Kalman filter bearish regime");
    if (feature.macdHist < 0) reasons.push("MACD histogram negative");
    if (feature.stochK > 70) reasons.push("Stochastic overbought");
    if (patternStats.winRate > 0.55) reasons.push(`Pattern history: ${(patternStats.winRate * 100).toFixed(0)}% win rate`);
  }
  
  const baseShouldTrade = vetoReasons.length === 0 && 
                       ensemble.confidence >= 0.65 && 
                       ensemble.consensus >= 0.5 &&
                       patternMatches.length >= 10 &&
                       reasons.length >= 1;
  
  let combinedIntelligence: CombinedIntelligence | undefined;
  let shouldTrade = baseShouldTrade;
  
  try {
    combinedIntelligence = await strategyLearner.getCombinedIntelligence(
      candles,
      feature,
      ensemble.direction,
      ensemble.confidence
    );
    
    if (combinedIntelligence.vetoes.length > 0) {
      combinedIntelligence.vetoes.forEach(v => {
        if (!vetoReasons.includes(v)) vetoReasons.push(v);
      });
    }
    
    combinedIntelligence.reasoning.forEach(r => {
      if (!reasons.includes(r)) reasons.push(r);
    });
    
    if (baseShouldTrade && combinedIntelligence.finalSignal === "HOLD") {
      shouldTrade = false;
      if (!vetoReasons.some(v => v.includes("Strategy Learner"))) {
        vetoReasons.push("Combined Intelligence recommends HOLD");
      }
    }
    
    if (baseShouldTrade && combinedIntelligence.strategyEV > 0) {
      reasons.push(`Strategy EV: +${(combinedIntelligence.strategyEV * 100).toFixed(2)}%`);
    }
  } catch (e) {
    console.warn("[Signal Engine] Combined intelligence failed:", e);
  }
  
  let entryZone: { low: number; high: number } | null = null;
  let stopLoss: number | null = null;
  let takeProfit1: number | null = null;
  let takeProfit2: number | null = null;
  let trailingStop: number | null = null;
  
  if (shouldTrade && ensemble.direction !== "HOLD") {
    const atr = feature.atr14;
    
    if (ensemble.direction === "LONG") {
      entryZone = {
        low: currentPrice - atr * 0.25,
        high: currentPrice + atr * 0.1,
      };
      stopLoss = currentPrice - atr * 1.5;
      takeProfit1 = currentPrice + atr * 1.5;
      takeProfit2 = currentPrice + atr * 3;
      trailingStop = feature.kalmanFast;
    } else {
      entryZone = {
        low: currentPrice - atr * 0.1,
        high: currentPrice + atr * 0.25,
      };
      stopLoss = currentPrice + atr * 1.5;
      takeProfit1 = currentPrice - atr * 1.5;
      takeProfit2 = currentPrice - atr * 3;
      trailingStop = feature.kalmanFast;
    }
  }
  
  const riskReward = entryZone && stopLoss && takeProfit1 
    ? calculateRiskReward(currentPrice, stopLoss, takeProfit1) 
    : 0;
  
  const edge = Math.abs(ensemble.expectedMove) / currentPrice - costs;
  const edgeBucket = classifyEdgeBucket(edge, costs);
  const edgeMultiple = costs > 0 ? edge / costs : 0;
  
  const expansionGate = checkExpansionGate(candles, feature.atr14, ensemble.direction);
  
  const displayConfidence = shouldTrade 
    ? ensemble.confidence 
    : ensemble.confidence * 0.7;
  
  return {
    signal: shouldTrade ? ensemble.direction : "HOLD",
    confidence: displayConfidence,
    regime,
    strategy,
    entryZone,
    stopLoss,
    takeProfit1,
    takeProfit2,
    trailingStop,
    riskReward,
    expectedHoldTime: holdTimeStr,
    estimatedCosts: costs,
    edge,
    edgeBucket,
    edgeMultiple,
    probUp: ensemble.probUp,
    probDown: ensemble.probDown,
    probChop: ensemble.probChop,
    expectedMove: ensemble.expectedMove,
    reasons: shouldTrade ? reasons : [],
    vetoReasons: shouldTrade ? [] : vetoReasons,
    patternMatches,
    mlPredictions: ensemble,
    expansionGate,
    combinedIntelligence,
  };
}

export function formatShotPlanForDisplay(plan: ShotPlan): string {
  let output = `\n=== SIGNAL: ${plan.signal} ===\n`;
  output += `Confidence: ${(plan.confidence * 100).toFixed(1)}%\n`;
  output += `Regime: ${plan.regime}\n`;
  output += `Strategy: ${plan.strategy}\n\n`;
  
  if (plan.entryZone) {
    output += `Entry Zone: $${plan.entryZone.low.toFixed(2)} - $${plan.entryZone.high.toFixed(2)}\n`;
    output += `Stop Loss: $${plan.stopLoss?.toFixed(2)}\n`;
    output += `Take Profit 1: $${plan.takeProfit1?.toFixed(2)}\n`;
    output += `Take Profit 2: $${plan.takeProfit2?.toFixed(2)}\n`;
    output += `Risk/Reward: ${plan.riskReward.toFixed(2)}\n\n`;
  }
  
  output += `Expected Hold: ${plan.expectedHoldTime}\n`;
  output += `Estimated Costs: ${(plan.estimatedCosts * 100).toFixed(3)}%\n`;
  output += `Edge: ${(plan.edge * 100).toFixed(3)}%\n\n`;
  
  output += `Probabilities:\n`;
  output += `  Up: ${(plan.probUp * 100).toFixed(1)}%\n`;
  output += `  Down: ${(plan.probDown * 100).toFixed(1)}%\n`;
  output += `  Chop: ${(plan.probChop * 100).toFixed(1)}%\n\n`;
  
  if (plan.reasons.length > 0) {
    output += `Reasons:\n`;
    plan.reasons.forEach((r, i) => output += `  ${i + 1}. ${r}\n`);
  }
  
  if (plan.vetoReasons.length > 0) {
    output += `\nVeto Reasons (why HOLD):\n`;
    plan.vetoReasons.forEach((r, i) => output += `  ${i + 1}. ${r}\n`);
  }
  
  return output;
}
